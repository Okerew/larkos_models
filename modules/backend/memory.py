from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_char_p,
    POINTER,
)

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
    out = []
    for i in range(size):
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


def serialize_state(mem_sys) -> dict:
    ms = mem_sys.contents

    def _level(level):
        size = int(level.size)
        cap  = int(level.capacity)
        n = max(0, min(size, cap))
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

