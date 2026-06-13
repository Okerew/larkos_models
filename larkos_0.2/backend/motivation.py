from ctypes import (
    CDLL, Structure,
    c_float,
    POINTER,
)


class IntrinsicMotivation(Structure):
    _fields_ = [
        ("novelty_score",     c_float),
        ("competence_score",  c_float),
        ("autonomy_score",    c_float),
        ("mastery_level",     c_float),
        ("curiosity_drive",   c_float),
        ("achievement_drive", c_float),
        ("exploration_rate",  c_float),
    ]


def bind(lib: CDLL):
    lib.initializeMotivationSystem.argtypes = []
    lib.initializeMotivationSystem.restype  = POINTER(IntrinsicMotivation)

    lib.updateMotivationSystem.argtypes = [
        POINTER(IntrinsicMotivation),
        c_float,
        c_float,
        c_float,
    ]
    lib.updateMotivationSystem.restype = None


def serialize_motivation(mot_ptr) -> dict:
    m = mot_ptr.contents
    return {
        "novelty_score":     float(m.novelty_score),
        "competence_score":  float(m.competence_score),
        "autonomy_score":    float(m.autonomy_score),
        "mastery_level":     float(m.mastery_level),
        "curiosity_drive":   float(m.curiosity_drive),
        "achievement_drive": float(m.achievement_drive),
        "exploration_rate":  float(m.exploration_rate),
    }
