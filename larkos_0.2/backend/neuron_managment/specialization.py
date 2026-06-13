from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_int,
    POINTER,
)

from modules.config import (
    MAX_NEURONS, INPUT_SIZE,
    MAX_SPECIALIZATIONS,
    MAX_SPECIALIZED_NEURONS,
    ACTIVATION_HISTORY_SIZE,
    SPEC_NONE,
    SPEC_PATTERN_DETECTOR,
    SPEC_FEATURE_EXTRACTOR,
    SPEC_TEMPORAL_PROCESSOR,
    SPEC_CONTEXT_INTEGRATOR,
    SPEC_DECISION_MAKER,
    SPEC_MEMORY_ENCODER,
    SPEC_EMOTIONAL_PROCESSOR,
    SPEC_PREDICTION_GENERATOR,
)

SPEC_TYPE_NAMES = [
    "None",
    "Pattern Detector",
    "Feature Extractor",
    "Temporal Processor",
    "Context Integrator",
    "Decision Maker",
    "Memory Encoder",
    "Emotional Processor",
    "Prediction Generator",
]


class SpecializedNeuron(Structure):
    _fields_ = [
        ("neuron_id",            c_uint),
        ("type",                 c_int),
        ("specialization_score", c_float),
        ("activation_history",
         c_float * ACTIVATION_HISTORY_SIZE),
        ("history_index",        c_uint),
        ("avg_activation",       c_float),
        ("importance_factor",    c_float),
    ]


class NeuronSpecializationSystem(Structure):
    _fields_ = [
        ("neurons",
         SpecializedNeuron * MAX_SPECIALIZED_NEURONS),
        ("count",                    c_uint),
        ("type_distribution",
         c_float * MAX_SPECIALIZATIONS),
        ("specialization_threshold", c_float),
    ]


def bind(lib: CDLL, max_neurons: int, max_connections: int) -> None:
    lib.initializeSpecializationSystem.argtypes = [c_float]
    lib.initializeSpecializationSystem.restype  = (
        POINTER(NeuronSpecializationSystem)
    )

    lib.detectSpecializations.argtypes = [
        POINTER(NeuronSpecializationSystem),
        POINTER(c_float),
        c_int,
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
    ]
    lib.detectSpecializations.restype = None

    lib.applySpecializations.argtypes = [
        POINTER(NeuronSpecializationSystem),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_int),
        c_int,
        c_int,
    ]
    lib.applySpecializations.restype = None

    lib.updateSpecializationImportance.argtypes = [
        POINTER(NeuronSpecializationSystem),
        c_float,
        c_float,
        POINTER(c_float),
    ]
    lib.updateSpecializationImportance.restype = None

    lib.evaluateSpecializationEffectiveness.argtypes = [
        POINTER(NeuronSpecializationSystem),
        c_float,
    ]
    lib.evaluateSpecializationEffectiveness.restype = c_float

    lib.printSpecializationStats.argtypes = [
        POINTER(NeuronSpecializationSystem),
    ]
    lib.printSpecializationStats.restype = None


def serialize_specialized_neuron(sn) -> dict:
    return {
        "neuron_id":            int(sn.neuron_id),
        "type":                 int(sn.type),
        "type_name":            SPEC_TYPE_NAMES[int(sn.type)],
        "specialization_score": float(sn.specialization_score),
        "importance_factor":    float(sn.importance_factor),
        "avg_activation":       float(sn.avg_activation),
        "activation_history": [
            float(sn.activation_history[i])
            for i in range(ACTIVATION_HISTORY_SIZE)
        ],
    }


def serialize_system(spec_sys) -> dict:
    s = spec_sys.contents
    return {
        "count":                    int(s.count),
        "specialization_threshold": float(s.specialization_threshold),
        "type_distribution": {
            SPEC_TYPE_NAMES[i]: float(s.type_distribution[i])
            for i in range(MAX_SPECIALIZATIONS)
        },
        "neurons": [
            serialize_specialized_neuron(s.neurons[i])
            for i in range(int(s.count))
        ],
    }
