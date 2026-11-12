# Hygiene ops for the C-side memory system: re-encode and remove.
# Additive module - no changes to structs, file formats, or existing
# functions. Ops act on tier arrays in place, keyed by timestamp
# (the same integer key that joins C entries to journal rows).
#
# Semantics:
#   - reencode: rewrites the vector of every entry carrying the
#     timestamp, zeroing importance first (reconsolidation: the slot
#     keeps its place in the tier but never inherits the old rank;
#     callers restore weight by memorizing corrected text as a fresh
#     entry). Timestamp preserved. Same-slot rewrite, no append.
#   - remove: deletes every entry carrying the timestamp (duplicate
#     keys occur after training reloads), shifting the tier array
#     left so no holes remain, zeroing the vacated tail slot, and
#     decrementing the tier size. Flat-array bookkeeping untouched.
#
# Neither op writes memory.bin. memorize/train flush anyway; call
# backend.save_memory() right after a hygiene mutation that must be
# durable immediately.

import ctypes

import numpy as np

from modules.backend.memory import MemoryEntry
from modules.config import MEMORY_VECTOR_SIZE

_TIER_NAMES = ("short_term", "medium_term", "long_term")


def _levels(mem_sys):
    h = mem_sys.contents.hierarchy
    return (
        ("short_term", h.short_term),
        ("medium_term", h.medium_term),
        ("long_term", h.long_term),
    )


def _usable(level):
    return max(0, min(int(level.size), int(level.capacity)))


def locate(mem_sys, ts):
    # All slots carrying the timestamp, any tier. Duplicates are
    # legal and reported, never guessed around.
    out = []
    for name, level in _levels(mem_sys):
        for i in range(_usable(level)):
            if int(level.entries[i].timestamp) == ts:
                out.append({
                    "tier": name,
                    "index": i,
                    "importance": float(level.entries[i].importance),
                })
    return out


def reencode(mem_sys, ts, vector):
    # Rewrite every slot with this timestamp in place. Raises
    # ValueError on a wrong-length or non-finite vector BEFORE any
    # write; the first failed readback returns status=error and
    # leaves the remaining slots untouched.
    v = np.asarray(vector, dtype=np.float32).ravel()
    if v.size != MEMORY_VECTOR_SIZE:
        raise ValueError(
            "vector must have %d floats, got %d" % (MEMORY_VECTOR_SIZE, v.size)
        )
    if not np.isfinite(v).all():
        raise ValueError("vector must be finite")
    hits = locate(mem_sys, ts)
    if not hits:
        return {"status": "not_found", "timestamp": int(ts), "reencoded": []}
    levels = dict(_levels(mem_sys))
    done = []
    for hit in hits:
        level = levels[hit["tier"]]
        e = level.entries[hit["index"]]
        arr = np.ctypeslib.as_array(e.vector)
        old_norm = float(np.sqrt((arr.astype(np.float64) ** 2).sum()))
        e.importance = 0.0
        arr[:] = v
        back = np.ctypeslib.as_array(level.entries[hit["index"]].vector)
        if not np.allclose(back, v, atol=1e-6):
            return {
                "status": "error",
                "detail": "readback mismatch at %s[%d]"
                % (hit["tier"], hit["index"]),
                "reencoded": done,
            }
        done.append({
            "tier": hit["tier"],
            "index": hit["index"],
            "old_importance": hit["importance"],
            "old_norm": round(old_norm, 6),
        })
    return {"status": "reencoded", "timestamp": int(ts), "reencoded": done}


def remove(mem_sys, ts):
    # Delete every slot with this timestamp from every tier, highest
    # index first so shifts never disturb pending targets.
    sizes_before = {name: _usable(level) for name, level in _levels(mem_sys)}
    removed = []
    for name, level in _levels(mem_sys):
        size = _usable(level)
        idxs = [
            i for i in range(size)
            if int(level.entries[i].timestamp) == ts
        ]
        if not idxs:
            continue
        esize = ctypes.sizeof(MemoryEntry)
        for i in sorted(idxs, reverse=True):
            if i < size - 1:
                dst = ctypes.addressof(level.entries[i])
                src = ctypes.addressof(level.entries[i + 1])
                ctypes.memmove(dst, src, (size - 1 - i) * esize)
            tail = level.entries[size - 1]
            tarr = np.ctypeslib.as_array(tail.vector)
            tarr[:] = 0.0
            tail.importance = 0.0
            tail.timestamp = 0
            size -= 1
        level.size = size
        removed.append({
            "tier": name,
            "indices": sorted(idxs),
            "size_after": size,
        })
    if not removed:
        return {"status": "not_found", "timestamp": int(ts), "removed": []}
    return {
        "status": "removed",
        "timestamp": int(ts),
        "removed": removed,
        "sizes_before": sizes_before,
    }
