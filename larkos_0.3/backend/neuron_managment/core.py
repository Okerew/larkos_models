from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_int,
    POINTER,
)


class NeuronPerformanceMetric(Structure):
    _fields_ = [
        ("output_stability",  c_float),
        ("prediction_error",  c_float),
        ("connection_quality", c_float),
        ("adaptive_response", c_float),
        ("importance_score",  c_float),
    ]


def bind(lib: CDLL, max_neurons: int):
    lib.computeNeuronPerformanceMetrics.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        POINTER(NeuronPerformanceMetric),
        c_uint,
        POINTER(c_float),
        c_int,
    ]
    lib.computeNeuronPerformanceMetrics.restype = None

    lib.removeUnderperformingNeuron.argtypes = [
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        POINTER(c_uint),
        c_uint,
        c_int,
    ]
    lib.removeUnderperformingNeuron.restype = None

    lib.addNewNeuron.argtypes = [
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        POINTER(c_uint),
        c_uint,
        POINTER(c_float),
    ]
    lib.addNewNeuron.restype = None

    lib.advancedNeuronManagement.argtypes = [
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        POINTER(c_uint),
        c_uint,
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_int,
    ]
    lib.advancedNeuronManagement.restype = None

    lib.initPredictiveCodingParams.argtypes = [c_int]
    lib.initPredictiveCodingParams.restype  = None

    lib.updateNeuronsWithPredictiveCoding.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int,
        c_float,
    ]
    lib.updateNeuronsWithPredictiveCoding.restype = None


def serialize_metrics(
    metrics: NeuronPerformanceMetric,
    num_neurons: int,
) -> list[dict]:
    return [
        {
            "output_stability":  float(metrics[i].output_stability),
            "prediction_error":  float(metrics[i].prediction_error),
            "connection_quality": float(metrics[i].connection_quality),
            "adaptive_response": float(metrics[i].adaptive_response),
            "importance_score":  float(metrics[i].importance_score),
        }
        for i in range(num_neurons)
    ]
