from ctypes import (
    CDLL, Structure,
    c_float, c_uint32,
    POINTER, c_void_p, c_bool, c_char_p, c_int,
)

from modules.config import PATTERN_SIZE, EXPERIENCE_VECTOR_SIZE


class VerificationSystem(Structure):
    _fields_ = [
        ("threshold",        c_float),
        ("reference_state",  POINTER(c_float)),
        ("state_size",       c_uint32),
    ]


class SelfIdentitySystem(Structure):
    _fields_ = [
        ("core_values",          POINTER(c_float)),
        ("belief_system",        POINTER(c_float)),
        ("identity_markers",     POINTER(c_float)),
        ("experience_history",   POINTER(c_float)),
        ("behavioral_patterns",  POINTER(c_float)),
        ("num_core_values",      c_uint32),
        ("num_beliefs",          c_uint32),
        ("num_markers",          c_uint32),
        ("history_size",         c_uint32),
        ("pattern_size",         c_uint32),
        ("consistency_score",    c_float),
        ("adaptation_rate",      c_float),
        ("confidence_level",     c_float),
        ("temporal_coherence",   POINTER(c_float)),
        ("coherence_window",     c_uint32),
        ("verification",         VerificationSystem),
    ]


class IdentityAnalysis(Structure):
    _fields_ = [
        ("core_value_conflicts",   c_int),
        ("belief_contradictions",  c_int),
        ("marker_deviations",      c_int),
        ("coherence_score",        c_float),
        ("stability_index",        c_float),
        ("adaptation_needed",      c_bool),
    ]


def bind(lib: CDLL):
    lib.initializeSelfIdentity.argtypes = [
        c_uint32, c_uint32, c_uint32, c_uint32, c_uint32,
    ]
    lib.initializeSelfIdentity.restype = POINTER(SelfIdentitySystem)

    lib.initializeIdentityComponents.argtypes = [
        POINTER(SelfIdentitySystem),
    ]
    lib.initializeIdentityComponents.restype = None

    lib.updateIdentity.argtypes = [
        POINTER(SelfIdentitySystem),
        c_void_p,
        c_uint32,
        c_void_p,
        POINTER(c_float),
    ]
    lib.updateIdentity.restype = None

    lib.verifyIdentity.argtypes = [POINTER(SelfIdentitySystem)]
    lib.verifyIdentity.restype  = c_bool

    lib.generateIdentityReflection.argtypes = [POINTER(SelfIdentitySystem)]
    lib.generateIdentityReflection.restype  = c_char_p

    lib.analyzeIdentitySystem.argtypes = [POINTER(SelfIdentitySystem)]
    lib.analyzeIdentitySystem.restype  = IdentityAnalysis

    lib.createIdentityBackup.argtypes = [POINTER(SelfIdentitySystem)]
    lib.createIdentityBackup.restype  = c_void_p

    lib.restoreIdentityFromBackup.argtypes = [
        POINTER(SelfIdentitySystem), c_void_p,
    ]
    lib.restoreIdentityFromBackup.restype = None

    lib.freeIdentityBackup.argtypes = [c_void_p]
    lib.freeIdentityBackup.restype  = None


def serialize_analysis(analysis: IdentityAnalysis) -> dict:
    return {
        "core_value_conflicts":  int(analysis.core_value_conflicts),
        "belief_contradictions": int(analysis.belief_contradictions),
        "marker_deviations":     int(analysis.marker_deviations),
        "coherence_score":       float(analysis.coherence_score),
        "stability_index":       float(analysis.stability_index),
        "adaptation_needed":     bool(analysis.adaptation_needed),
    }


def serialize_identity(sys_ptr) -> dict:
    s  = sys_ptr.contents
    nv = int(s.num_core_values)
    nb = int(s.num_beliefs)
    nm = int(s.num_markers)
    nh = int(s.history_size)
    np_ = int(s.pattern_size)
    cw = int(s.coherence_window)
    return {
        "num_core_values":    nv,
        "num_beliefs":        nb,
        "num_markers":        nm,
        "history_size":       nh,
        "pattern_size":       np_,
        "consistency_score":  float(s.consistency_score),
        "adaptation_rate":    float(s.adaptation_rate),
        "confidence_level":   float(s.confidence_level),
        "coherence_window":   cw,
        "core_values": [
            float(s.core_values[i]) for i in range(nv)
        ],
        "belief_system": [
            float(s.belief_system[i]) for i in range(nb)
        ],
        "identity_markers": [
            float(s.identity_markers[i]) for i in range(nm)
        ],
        "experience_history": [
            float(s.experience_history[i]) for i in range(nh)
        ],
        "behavioral_patterns": [
            float(s.behavioral_patterns[i]) for i in range(np_)
        ],
        "temporal_coherence": [
            float(s.temporal_coherence[i]) for i in range(cw)
        ],
        "verification": {
            "threshold":  float(s.verification.threshold),
            "state_size": int(s.verification.state_size),
        },
    }
