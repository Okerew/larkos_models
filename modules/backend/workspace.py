"""ctypes bindings for the C-side workspace pool (WM runtime).

Struct layout must stay in lockstep with neural_web.c / include/nw.h.
All C workspace types are plain value structs (no inner pointers), so
Python can walk the pool directly and the whole block round-trips
through saveWorkspacePool / loadWorkspacePool.
"""

from ctypes import (
    CDLL, Structure,
    c_bool, c_char, c_char_p, c_float, c_int, c_uint,
    POINTER,
)

from modules.backend.memory import MemorySystem, WorkingMemorySystem
from modules.config import (
    MEMORY_VECTOR_SIZE,
    CONTEXT_VECTOR_SIZE,
    WM_LABEL_SIZE,
    WM_TEXT_SIZE,
    WM_HISTORY_SIZE,
    WM_KIND_NAMES,
    WM_MAX_LINKS,
    WM_MAX_SLOTS,
    WM_POOL_CAPACITY,
    WM_TASK_CAP
)

class WMSlot(Structure):
    _fields_ = [
        ("label", c_char * WM_LABEL_SIZE),
        ("text", c_char * WM_TEXT_SIZE),
        ("vector", c_float * MEMORY_VECTOR_SIZE),
        ("importance", c_float),
        ("step_tag", c_uint),
        ("used", c_bool),
    ]


class WMLink(Structure):
    _fields_ = [
        ("a", c_uint),
        ("b", c_uint),
        ("type", c_char * WM_LABEL_SIZE),
        ("weight", c_float),
        ("used", c_bool),
    ]


class WMSlotSnapshot(Structure):
    _fields_ = [
        ("step", c_uint),
        ("label", c_char * WM_LABEL_SIZE),
        ("vector", c_float * MEMORY_VECTOR_SIZE),
        ("importance", c_float),
    ]


class Workspace(Structure):
    _fields_ = [
        ("workspace_id", c_uint),
        ("kind", c_uint),
        ("active", c_bool),
        ("epoch", c_uint),
        ("created_step", c_uint),
        ("last_used_step", c_uint),
        ("context_vector", c_float * CONTEXT_VECTOR_SIZE),
        ("slots", WMSlot * WM_MAX_SLOTS),
        ("num_slots", c_uint),
        ("links", WMLink * WM_MAX_LINKS),
        ("num_links", c_uint),
        ("history", WMSlotSnapshot * WM_HISTORY_SIZE),
        ("history_count", c_uint),
        ("history_head", c_uint),
    ]


class WorkspacePool(Structure):
    _fields_ = [
        ("workspaces", Workspace * WM_POOL_CAPACITY),
        ("next_id", c_uint),
        ("epoch", c_uint),
    ]


def bind(lib: CDLL):
    lib.createWorkspacePool.argtypes = []
    lib.createWorkspacePool.restype = POINTER(WorkspacePool)

    lib.freeWorkspacePool.argtypes = [POINTER(WorkspacePool)]
    lib.freeWorkspacePool.restype = None

    lib.createWorkspace.argtypes = [
        POINTER(WorkspacePool), c_uint, POINTER(c_float),
        POINTER(MemorySystem), c_uint,
    ]
    lib.createWorkspace.restype = c_int

    lib.destroyWorkspace.argtypes = [
        POINTER(WorkspacePool), c_uint,
    ]
    lib.destroyWorkspace.restype = c_int

    lib.writeSlot.argtypes = [
        POINTER(WorkspacePool), c_uint, c_char_p, c_char_p,
        POINTER(c_float), c_float, c_uint,
    ]
    lib.writeSlot.restype = c_int

    lib.transformSlot.argtypes = [
        POINTER(WorkspacePool), c_uint, c_char_p,
        POINTER(c_float), c_float, c_uint,
    ]
    lib.transformSlot.restype = c_int

    lib.linkSlots.argtypes = [
        POINTER(WorkspacePool), c_uint, c_char_p, c_char_p,
        c_char_p, c_float,
    ]
    lib.linkSlots.restype = c_int

    lib.wmRetrieve.argtypes = [
        POINTER(WorkspacePool), c_uint, POINTER(c_float), c_int,
        POINTER(WorkingMemorySystem), c_uint, POINTER(c_int),
        POINTER(c_float), c_int,
    ]
    lib.wmRetrieve.restype = c_int

    lib.wmCommitSlot.argtypes = [
        POINTER(WorkspacePool), c_uint, c_char_p,
        POINTER(MemorySystem), c_uint, c_float,
    ]
    lib.wmCommitSlot.restype = c_int

    lib.wmValidateWorkspace.argtypes = [
        POINTER(WorkspacePool), c_uint, POINTER(MemorySystem),
        POINTER(c_int), POINTER(c_int), POINTER(c_float),
        POINTER(c_float), c_int,
    ]
    lib.wmValidateWorkspace.restype = c_int

    lib.wmDiscardSlot.argtypes = [
        POINTER(WorkspacePool), c_uint, c_char_p,
        POINTER(c_float), c_uint,
    ]
    lib.wmDiscardSlot.restype = c_int

    lib.replayWorkspace.argtypes = [
        POINTER(WorkspacePool), c_uint, c_uint,
        POINTER(WMSlotSnapshot), c_int,
    ]
    lib.replayWorkspace.restype = c_int

    lib.validateWorkspacePool.argtypes = [POINTER(WorkspacePool)]
    lib.validateWorkspacePool.restype = c_bool

    lib.saveWorkspacePool.argtypes = [
        POINTER(WorkspacePool), c_char_p,
    ]
    lib.saveWorkspacePool.restype = None

    lib.loadWorkspacePool.argtypes = [c_char_p]
    lib.loadWorkspacePool.restype = POINTER(WorkspacePool)


def serialize_workspace(ws: Workspace) -> dict:
    slots = []
    labels = {}
    for i in range(WM_MAX_SLOTS):
        s = ws.slots[i]
        if not s.used:
            continue
        label = s.label.decode(errors="replace").rstrip("\x00")
        labels[i] = label
        slots.append({
            "label": label,
            "text": s.text.decode(errors="replace").rstrip("\x00"),
            "importance": round(float(s.importance), 6),
            "step_tag": int(s.step_tag),
        })
    links = []
    for i in range(WM_MAX_LINKS):
        l = ws.links[i]
        if not l.used:
            continue
        links.append({
            "a": labels.get(int(l.a), f"slot_{int(l.a)}"),
            "b": labels.get(int(l.b), f"slot_{int(l.b)}"),
            "type": l.type.decode(errors="replace").rstrip("\x00"),
            "weight": round(float(l.weight), 6),
        })
    return {
        "workspace_id": int(ws.workspace_id),
        "kind": WM_KIND_NAMES.get(int(ws.kind), str(int(ws.kind))),
        "epoch": int(ws.epoch),
        "created_step": int(ws.created_step),
        "last_used_step": int(ws.last_used_step),
        "num_slots": len(slots),
        "num_links": len(links),
        "history_count": int(ws.history_count),
        "slots": slots,
        "links": links,
    }


def serialize_pool(pool_ptr) -> dict:
    if not pool_ptr:
        return {"workspaces": [], "epoch": 0}
    pool = pool_ptr.contents
    workspaces = [
        serialize_workspace(pool.workspaces[i])
        for i in range(WM_POOL_CAPACITY)
        if pool.workspaces[i].active
    ]
    tasks = sum(1 for w in workspaces if w["kind"] == "task")
    return {
        "workspaces": workspaces,
        "epoch": int(pool.epoch),
        "task_count": tasks,
        "task_cap": WM_TASK_CAP,
        "pool_capacity": WM_POOL_CAPACITY,
    }


def slot_labels(pool_ptr, workspace_id: int) -> list[tuple]:
    """[(slot_index, label, importance)] for used slots - slot_index is
    the raw C array position wmValidateWorkspace reports against."""
    if not pool_ptr:
        return []
    pool = pool_ptr.contents
    for i in range(WM_POOL_CAPACITY):
        ws = pool.workspaces[i]
        if not ws.active or int(ws.workspace_id) != int(workspace_id):
            continue
        out = []
        for j in range(WM_MAX_SLOTS):
            s = ws.slots[j]
            if s.used:
                out.append((
                    j,
                    s.label.decode(errors="replace").rstrip("\x00"),
                    float(s.importance),
                ))
        return out
    return []
