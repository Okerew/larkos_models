from ctypes import (
    CDLL, Structure,
    c_float, c_int,
    POINTER,
)
import modules.backend.memory as memory

from modules.config import MAX_NEURONS, INPUT_SIZE


class NetworkStateSnapshot(Structure):
    _fields_ = [
        ("states",          c_float * MAX_NEURONS),
        ("outputs",         c_float * MAX_NEURONS),
        ("inputs",          c_float * INPUT_SIZE),
        ("step",            c_int),
        ("current_memory",  memory.MemoryEntry),
    ]


def bind(lib: CDLL):
    lib.captureNetworkState.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(NetworkStateSnapshot),
        POINTER(c_float),
        c_int,
    ]
    lib.captureNetworkState.restype = None

    lib.saveNetworkStates.argtypes = [
        POINTER(NetworkStateSnapshot),
        c_int,
    ]
    lib.saveNetworkStates.restype = None

    lib.loadNetworkStates.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
    ]
    lib.loadNetworkStates.restype = c_int


def capture_current_memory(
    snapshot: NetworkStateSnapshot,
    mem_sys,
) -> None:
    m = mem_sys.contents
    idx = (m.head - 1 + m.capacity) % m.capacity
    snapshot.current_memory = m.entries[idx]


def serialize_snapshot(snap: NetworkStateSnapshot) -> dict:
    return {
        "step":    int(snap.step),
        "states":  [float(snap.states[i])  for i in range(MAX_NEURONS)],
        "outputs": [float(snap.outputs[i]) for i in range(MAX_NEURONS)],
        "inputs":  [float(snap.inputs[i])  for i in range(INPUT_SIZE)],
        "current_memory": {
            "importance": float(snap.current_memory.importance),
            "timestamp":  int(snap.current_memory.timestamp),
            "vector": [
                float(snap.current_memory.vector[i])
                for i in range(memory.MEMORY_VECTOR_SIZE)
            ],
        },
    }


def serialize_history(
    history: list[NetworkStateSnapshot],
    total_steps: int,
) -> list[dict]:
    return [serialize_snapshot(history[i]) for i in range(total_steps)]
