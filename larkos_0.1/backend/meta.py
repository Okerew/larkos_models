from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_int,
    POINTER,
)

from modules.config import HISTORY_LENGTH


class MetaController(Structure):
    _fields_ = [
        ("meta_learning_rate",          c_float),
        ("exploration_factor",          c_float),
        ("region_importance_scores",    POINTER(c_float)),
        ("learning_efficiency_history", POINTER(c_float)),
        ("num_regions",                 c_int),
    ]


class MetacognitionMetrics(Structure):
    _fields_ = [
        ("confidence_level",   c_float),
        ("adaptation_rate",    c_float),
        ("cognitive_load",     c_float),
        ("error_awareness",    c_float),
        ("context_relevance",  c_float),
        ("performance_history", c_float * HISTORY_LENGTH),
    ]


class MetaLearningState(Structure):
    _fields_ = [
        ("learning_efficiency", c_float),
        ("exploration_rate",    c_float),
        ("stability_index",     c_float),
        ("priority_weights",    POINTER(c_float)),
        ("current_phase",       c_int),
    ]


class NetworkPerformanceMetrics(Structure):
    _fields_ = [
        ("region_performance_scores",  POINTER(c_float)),
        ("region_error_rates",         POINTER(c_float)),
        ("region_output_variance",     POINTER(c_float)),
        ("num_regions",                c_int),
    ]


class DecisionPath(Structure):
    _fields_ = [
        ("states",      POINTER(c_float)),
        ("weights",     POINTER(c_float)),
        ("connections", POINTER(c_uint)),
        ("score",       c_float),
        ("num_steps",   c_int),
    ]


def bind(lib: CDLL, max_neurons: int):
    lib.initializeMetaController.argtypes = [c_int]
    lib.initializeMetaController.restype  = POINTER(MetaController)

    lib.initializeMetacognitionMetrics.argtypes = []
    lib.initializeMetacognitionMetrics.restype  = \
        POINTER(MetacognitionMetrics)

    lib.initializeMetaLearningState.argtypes = [c_int]
    lib.initializeMetaLearningState.restype  = POINTER(MetaLearningState)

    lib.computePerformanceVariance.argtypes = [POINTER(c_float), c_int]
    lib.computePerformanceVariance.restype  = c_float

    lib.updateMetacognitionMetrics.argtypes = [
        POINTER(MetacognitionMetrics),
        POINTER(MetaController),
        POINTER(NetworkPerformanceMetrics),
    ]
    lib.updateMetacognitionMetrics.restype = None

    lib.updateMetaControllerPriorities.argtypes = [
        POINTER(MetaController),
        POINTER(NetworkPerformanceMetrics),
        POINTER(MetacognitionMetrics),
    ]
    lib.updateMetaControllerPriorities.restype = None

    lib.applyMetaControllerAdaptations.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(MetaController),
        c_int,
    ]
    lib.applyMetaControllerAdaptations.restype = None

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

    lib.evaluatePathQuality.argtypes = [
        DecisionPath,
        POINTER(MetaLearningState),
        POINTER(MetacognitionMetrics),
    ]
    lib.evaluatePathQuality.restype = c_float

    lib.updateMetaLearningState.argtypes = [
        POINTER(MetaLearningState),
        DecisionPath,
        POINTER(MetacognitionMetrics),
    ]
    lib.updateMetaLearningState.restype = None


def serialize_controller(ctrl) -> dict:
    c = ctrl.contents
    n = c.num_regions
    return {
        "meta_learning_rate": float(c.meta_learning_rate),
        "exploration_factor": float(c.exploration_factor),
        "num_regions":        n,
        "region_importance_scores": [
            float(c.region_importance_scores[i]) for i in range(n)
        ],
        "learning_efficiency_history": [
            float(c.learning_efficiency_history[i]) for i in range(n)
        ],
    }


def serialize_metacog(mc) -> dict:
    m = mc.contents
    return {
        "confidence_level":  float(m.confidence_level),
        "adaptation_rate":   float(m.adaptation_rate),
        "cognitive_load":    float(m.cognitive_load),
        "error_awareness":   float(m.error_awareness),
        "context_relevance": float(m.context_relevance),
        "performance_history": [
            float(m.performance_history[i])
            for i in range(HISTORY_LENGTH)
        ],
    }


def serialize_meta_state(ms) -> dict:
    s = ms.contents
    return {
        "learning_efficiency": float(s.learning_efficiency),
        "exploration_rate":    float(s.exploration_rate),
        "stability_index":     float(s.stability_index),
        "current_phase":       int(s.current_phase),
    }


def make_performance_metrics(
    scores: list[float],
    lib_alloc=None,
) -> NetworkPerformanceMetrics:
    n = len(scores)
    arr = (c_float * n)(*scores)
    perf = NetworkPerformanceMetrics()
    perf.region_performance_scores = arr
    perf.region_error_rates = None
    perf.region_output_variance = None
    perf.num_regions = n
    return perf
