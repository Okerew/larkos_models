from ctypes import (
    CDLL, Structure,
    c_float, c_int,
    POINTER,
)


class DynamicParameters(Structure):
    _fields_ = [
        ("input_noise_scale",       c_float),
        ("weight_noise_scale",      c_float),
        ("base_adaptation_rate",    c_float),
        ("current_adaptation_rate", c_float),
        ("learning_momentum",       c_float),
        ("stability_threshold",     c_float),
        ("noise_tolerance",         c_float),
        ("recovery_rate",           c_float),
        ("plasticity",              c_float),
        ("homeostatic_factor",      c_float),
    ]


def bind(lib: CDLL):
    lib.initDynamicParameters.argtypes = []
    lib.initDynamicParameters.restype  = DynamicParameters


def make_default() -> DynamicParameters:
    p = DynamicParameters()
    p.input_noise_scale       = 0.1
    p.weight_noise_scale      = 0.05
    p.base_adaptation_rate    = 0.01
    p.current_adaptation_rate = 0.01
    p.learning_momentum       = 0.9
    p.stability_threshold     = 0.8
    p.noise_tolerance         = 0.2
    p.recovery_rate           = 0.05
    p.plasticity              = 1.0
    p.homeostatic_factor      = 0.1
    return p


def serialize_params(p: DynamicParameters) -> dict:
    return {
        "input_noise_scale":       float(p.input_noise_scale),
        "weight_noise_scale":      float(p.weight_noise_scale),
        "base_adaptation_rate":    float(p.base_adaptation_rate),
        "current_adaptation_rate": float(p.current_adaptation_rate),
        "learning_momentum":       float(p.learning_momentum),
        "stability_threshold":     float(p.stability_threshold),
        "noise_tolerance":         float(p.noise_tolerance),
        "recovery_rate":           float(p.recovery_rate),
        "plasticity":              float(p.plasticity),
        "homeostatic_factor":      float(p.homeostatic_factor),
    }
