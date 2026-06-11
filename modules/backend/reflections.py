from ctypes import (
    CDLL, Structure,
    c_float, c_int, c_bool, c_uint, c_char,
    POINTER, c_void_p,
)

from modules.config import REASONING_SIZE, HISTORY_SIZE


class ReflectionMetrics(Structure):
    _fields_ = [
        ("confidence_score",        c_float),
        ("coherence_score",         c_float),
        ("novelty_score",           c_float),
        ("consistency_score",       c_float),
        ("reasoning",               c_char * REASONING_SIZE),
        ("potentially_confabulated",c_bool),
    ]


class ReflectionHistory(Structure):
    _fields_ = [
        ("historical_confidence",  c_float * HISTORY_SIZE),
        ("historical_coherence",   c_float * HISTORY_SIZE),
        ("historical_consistency", c_float * HISTORY_SIZE),
        ("history_index",          c_int),
        ("confidence_threshold",   c_float),
        ("coherence_threshold",    c_float),
        ("consistency_threshold",  c_float),
    ]


class ReflectionParameters(Structure):
    _fields_ = [
        ("current_adaptation_rate", c_float),
        ("input_noise_scale",       c_float),
        ("weight_noise_scale",      c_float),
        ("plasticity",              c_float),
        ("noise_tolerance",         c_float),
        ("learning_rate",           c_float),
    ]


def bind(lib: CDLL, max_neurons: int, max_connections: int):
    lib.initializeReflectionSystem.argtypes = []
    lib.initializeReflectionSystem.restype  = POINTER(ReflectionHistory)

    lib.initializeReflectionParameters.argtypes = []
    lib.initializeReflectionParameters.restype  = (
        POINTER(ReflectionParameters)
    )

    lib.performSelfReflection.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        POINTER(ReflectionHistory),
        c_int,
    ]
    lib.performSelfReflection.restype = ReflectionMetrics

    lib.integrateReflectionSystem.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_int,
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(ReflectionParameters),
    ]
    lib.integrateReflectionSystem.restype = None


def serialize_history(hist_ptr) -> dict:
    h = hist_ptr.contents
    return {
        "history_index":          int(h.history_index),
        "confidence_threshold":   float(h.confidence_threshold),
        "coherence_threshold":    float(h.coherence_threshold),
        "consistency_threshold":  float(h.consistency_threshold),
        "historical_confidence":  [
            float(h.historical_confidence[i])
            for i in range(HISTORY_SIZE)
        ],
        "historical_coherence":   [
            float(h.historical_coherence[i])
            for i in range(HISTORY_SIZE)
        ],
        "historical_consistency": [
            float(h.historical_consistency[i])
            for i in range(HISTORY_SIZE)
        ],
    }


def serialize_params(params_ptr) -> dict:
    p = params_ptr.contents
    return {
        "current_adaptation_rate": float(p.current_adaptation_rate),
        "input_noise_scale":       float(p.input_noise_scale),
        "weight_noise_scale":      float(p.weight_noise_scale),
        "plasticity":              float(p.plasticity),
        "noise_tolerance":         float(p.noise_tolerance),
        "learning_rate":           float(p.learning_rate),
    }


def serialize_metrics(metrics: ReflectionMetrics) -> dict:
    return {
        "confidence_score":         float(metrics.confidence_score),
        "coherence_score":          float(metrics.coherence_score),
        "novelty_score":            float(metrics.novelty_score),
        "consistency_score":        float(metrics.consistency_score),
        "potentially_confabulated": bool(metrics.potentially_confabulated),
    }
