from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_int,
    POINTER,
)

from modules.config import NUM_PATHS, MAX_DECISION_STEPS


class DecisionPath(Structure):
    _fields_ = [
        ("states",      POINTER(c_float)),
        ("weights",     POINTER(c_float)),
        ("connections", POINTER(c_uint)),
        ("score",       c_float),
        ("num_steps",   c_int),
    ]


def bind(lib: CDLL, max_neurons: int):
    lib.simulateFutureStates.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        c_int,
        c_int,
    ]
    lib.simulateFutureStates.restype = None

    lib.generatePotentialTargets.argtypes = [
        c_int,
        POINTER(c_float),
        POINTER(c_float),
        c_int,
        POINTER(c_float),
        c_float,
    ]
    lib.generatePotentialTargets.restype = POINTER(c_float)

    lib.evaluateFutureOutcome.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int,
    ]
    lib.evaluateFutureOutcome.restype = c_float

    lib.selectOptimalDecisionPath.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        c_int,
        POINTER(c_float),
        POINTER(c_float),
        c_int,
        POINTER(c_float),
        c_float,
    ]
    lib.selectOptimalDecisionPath.restype = None

    lib.generateDecisionPath.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        c_int,
        c_float,
    ]
    lib.generateDecisionPath.restype = DecisionPath

    lib.selectBestPath.argtypes = [POINTER(DecisionPath), c_int]
    lib.selectBestPath.restype  = DecisionPath

    lib.applyDecisionPath.argtypes = [
        DecisionPath,
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint),
        c_float,
    ]
    lib.applyDecisionPath.restype = None

    lib.selectOptimalMetaDecisionPath.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        c_int,
        POINTER(c_float),
        POINTER(c_float),
    ]
    lib.selectOptimalMetaDecisionPath.restype = None


def serialize_decision_path(path: DecisionPath, num_steps: int) -> dict:
    return {
        "score":     float(path.score),
        "num_steps": int(path.num_steps),
        "states": [
            float(path.states[i])
            for i in range(num_steps)
            if path.states
        ],
        "weights": [
            float(path.weights[i])
            for i in range(num_steps)
            if path.weights
        ],
    }
