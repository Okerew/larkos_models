from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_char_p,
    POINTER,
)

import numpy as np

from modules.config import (
    MAX_NEURONS, INPUT_SIZE,
    MEMORY_VECTOR_SIZE,
    FEATURE_VECTOR_SIZE,
    CONTEXT_VECTOR_SIZE,
    MEMORY_CAPACITY,
)


class MemoryEntry(Structure):
    _fields_ = [
        ("vector",     c_float * MEMORY_VECTOR_SIZE),
        ("importance", c_float),
        ("timestamp",  c_uint),
    ]


class MemoryLevel(Structure):
    _fields_ = [
        ("entries",              POINTER(MemoryEntry)),
        ("importance_threshold", c_float),
        ("size",                 c_uint),
        ("capacity",             c_uint),
    ]


class MemoryHierarchy(Structure):
    _fields_ = [
        ("short_term",              MemoryLevel),
        ("medium_term",             MemoryLevel),
        ("long_term",               MemoryLevel),
        ("consolidation_threshold", c_float),
        ("abstraction_threshold",   c_float),
        ("total_capacity",          c_uint),
    ]


class MemorySystem(Structure):
    _fields_ = [
        ("hierarchy", MemoryHierarchy),
        ("head",      c_uint),
        ("size",      c_uint),
        ("capacity",  c_uint),
        ("entries",   POINTER(MemoryEntry)),
    ]


class WorkingMemoryEntry(Structure):
    _fields_ = [
        ("features",          POINTER(c_float)),
        ("abstraction_level", c_float),
        ("context_vector",    POINTER(c_float)),
        ("depth",             c_uint),
    ]


class FocusBuffer(Structure):
    _fields_ = [
        ("entries",             POINTER(WorkingMemoryEntry)),
        ("size",                c_uint),
        ("capacity",            c_uint),
        ("attention_threshold", c_float),
    ]


class ActiveBuffer(Structure):
    _fields_ = [
        ("entries",          POINTER(WorkingMemoryEntry)),
        ("size",             c_uint),
        ("capacity",         c_uint),
        ("activation_decay", c_float),
    ]


class SemanticCluster(Structure):
    _fields_ = [
        ("vector",    POINTER(c_float)),
        ("size",      c_uint),
        ("coherence", c_float),
        ("activation", POINTER(c_float)),
    ]


class ClusterSet(Structure):
    _fields_ = [
        ("clusters",          POINTER(SemanticCluster)),
        ("num_clusters",      c_uint),
        ("similarity_matrix", POINTER(c_float)),
    ]


class WorkingMemorySystem(Structure):
    _fields_ = [
        ("focus",          FocusBuffer),
        ("active",         ActiveBuffer),
        ("clusters",       ClusterSet),
        ("global_context", POINTER(c_float)),
    ]


FeatureMatrix = (c_float * MEMORY_VECTOR_SIZE) * FEATURE_VECTOR_SIZE

# Hard cap on how many entries per level serialize_state will marshal
# into Python dicts. With MEMORY_CAPACITY at 1M a full serialization
# would build ~30k+ dicts of 288 floats PER CALL (journal joins, API
# status, output prompt, cognitive_fuse sampling all read it), so the
# entry list is truncated while size/capacity stay truthful. Consumers
# that only need counts use serialize_stats instead.
MAX_SERIALIZED_ENTRIES = 10_000


def bind(lib: CDLL):
    lib.createMemorySystem.argtypes = [c_uint]
    lib.createMemorySystem.restype  = POINTER(MemorySystem)

    lib.createWorkingMemorySystem.argtypes = [c_uint]
    lib.createWorkingMemorySystem.restype  = POINTER(WorkingMemorySystem)

    lib.addMemory.argtypes = [
        POINTER(MemorySystem),
        POINTER(WorkingMemorySystem),
        POINTER(c_uint),
        POINTER(c_float),
        c_uint,
        POINTER(FeatureMatrix),
    ]
    lib.addMemory.restype = None

    lib.consolidateMemory.argtypes = [POINTER(MemorySystem)]
    lib.consolidateMemory.restype  = None

    lib.decayMemorySystem.argtypes = [POINTER(MemorySystem)]
    lib.decayMemorySystem.restype  = None

    lib.addToDirectMemory.argtypes = [
        POINTER(MemorySystem),
        POINTER(MemoryEntry),
    ]
    lib.addToDirectMemory.restype = None

    lib.consolidateToLongTermMemory.argtypes = [
        POINTER(WorkingMemorySystem),
        POINTER(MemorySystem),
        c_uint,
    ]
    lib.consolidateToLongTermMemory.restype = None

    lib.freeMemorySystem.argtypes = [POINTER(MemorySystem)]
    lib.freeMemorySystem.restype  = None

    lib.saveMemorySystem.argtypes = [POINTER(MemorySystem), c_char_p]
    lib.saveMemorySystem.restype  = None

    lib.loadMemorySystem.argtypes = [c_char_p]
    lib.loadMemorySystem.restype  = POINTER(MemorySystem)


def serialize_level(entries_ptr, size: int) -> list:
    if not entries_ptr:
        return []
    out = []
    safe_size = min(size, MEMORY_CAPACITY)
    for i in range(safe_size):
        e = entries_ptr[i]
        out.append({
            "importance": float(e.importance),
            "timestamp":  int(e.timestamp),
            "vector": [
                float(e.vector[j])
                for j in range(MEMORY_VECTOR_SIZE)
            ],
        })
    return out


def _iter_levels(ms):
    return (
        ("short_term",  ms.hierarchy.short_term),
        ("medium_term", ms.hierarchy.medium_term),
        ("long_term",   ms.hierarchy.long_term),
    )


def _level_count(level) -> int:
    return max(
        0, min(
            int(level.size), int(level.capacity),
            MAX_SERIALIZED_ENTRIES,
        )
    )


def entry_meta(mem_sys) -> list:
    """[(level, timestamp, importance)] for every serialized entry,
    same per-level cap as serialize_state but with the 288-float
    vectors skipped - tier joins, next_timestamp and status ranking
    run on every API call and never touch the vectors."""
    if not mem_sys:
        return []
    out = []
    for name, level in _iter_levels(mem_sys.contents):
        for i in range(_level_count(level)):
            e = level.entries[i]
            out.append((name, int(e.timestamp), float(e.importance)))
    return out


def entry_views(mem_sys) -> list:
    """Same walk as entry_meta but the vector rides along as a
    zero-copy ndarray view into the C memory, so state-mode recall
    can score entries without building per-entry Python dicts."""
    if not mem_sys:
        return []
    out = []
    for name, level in _iter_levels(mem_sys.contents):
        for i in range(_level_count(level)):
            e = level.entries[i]
            out.append((
                name, int(e.timestamp), float(e.importance),
                np.ctypeslib.as_array(e.vector),
            ))
    return out


def serialize_stats(mem_sys) -> dict:
    """Sizes-and-capacities-only view of the memory system.

    Same top-level shape as serialize_state minus the entry lists, so
    per-step consumers (build_input_tensor churn channels, tier
    counts) never pay the per-entry marshalling cost. At 1M capacity
    that cost dominates the step otherwise.
    """
    if not mem_sys:
        return {"size": 0, "capacity": 0,
                "short_term": {"size": 0, "capacity": 0},
                "medium_term": {"size": 0, "capacity": 0},
                "long_term": {"size": 0, "capacity": 0}}
    ms = mem_sys.contents

    def _level(level):
        return {
            "size":     int(level.size),
            "capacity": int(level.capacity),
        }

    return {
        "size":        int(ms.size),
        "capacity":    int(ms.capacity),
        "short_term":  _level(ms.hierarchy.short_term),
        "medium_term": _level(ms.hierarchy.medium_term),
        "long_term":   _level(ms.hierarchy.long_term),
    }


def serialize_state(mem_sys) -> dict:
    if not mem_sys:
        return {"size": 0, "capacity": 0,
                "short_term": {"size": 0, "capacity": 0, "entries": []},
                "medium_term": {"size": 0, "capacity": 0, "entries": []},
                "long_term": {"size": 0, "capacity": 0, "entries": []}}
    ms = mem_sys.contents

    def _level(level):
        size = int(level.size)
        cap  = int(level.capacity)
        n = max(0, min(size, cap, MAX_SERIALIZED_ENTRIES))
        return {
            "size":     size,
            "capacity": cap,
            "entries":  serialize_level(level.entries, n),
        }

    return {
        "size":        int(ms.size),
        "capacity":    int(ms.capacity),
        "short_term":  _level(ms.hierarchy.short_term),
        "medium_term": _level(ms.hierarchy.medium_term),
        "long_term":   _level(ms.hierarchy.long_term),
    }

