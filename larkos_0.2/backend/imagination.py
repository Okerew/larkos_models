from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_int, c_bool, c_char,
    POINTER,
)

from modules.backend.memory import MemorySystem
from modules.config import (
    MEMORY_VECTOR_SIZE,
    MAX_SCENARIOS,
    MAX_OUTCOMES_PER_SCENARIO,
    SCENARIO_NAME_SIZE,
    OUTCOME_DESC_SIZE,
    DIVERGENCE_HISTORY_SIZE,
)


class ImaginedOutcome(Structure):
    _fields_ = [
        ("probability",  c_float),
        ("confidence",   c_float),
        ("impact_score", c_float),
        ("plausibility", c_float),
        ("vector",       c_float * MEMORY_VECTOR_SIZE),
        ("description",  c_char * OUTCOME_DESC_SIZE),
    ]


class ImaginationScenario(Structure):
    _fields_ = [
        ("num_outcomes",      c_int),
        ("outcomes",          ImaginedOutcome * MAX_OUTCOMES_PER_SCENARIO),
        ("divergence_factor", c_float),
        ("creativity_level",  c_float),
    ]


class ImaginationSystem(Structure):
    _fields_ = [
        ("scenarios",
            ImaginationScenario * MAX_SCENARIOS),
        ("num_scenarios",              c_int),
        ("current_scenario",           c_int),
        ("creativity_factor",          c_float),
        ("coherence_threshold",        c_float),
        ("novelty_weight",             c_float),
        ("memory_influence",           c_float),
        ("identity_influence",         c_float),
        ("active",                     c_bool),
        ("steps_simulated",            c_uint),
        ("divergence_history",
            c_float * DIVERGENCE_HISTORY_SIZE),
        ("current_scenario_name",      c_char * SCENARIO_NAME_SIZE),
        ("total_scenarios_generated",  c_uint),
    ]


def bind(lib: CDLL):
    lib.initializeImaginationSystem.argtypes = [c_float, c_float]
    lib.initializeImaginationSystem.restype  = POINTER(ImaginationSystem)

    lib.freeImaginationSystem.argtypes = [POINTER(ImaginationSystem)]
    lib.freeImaginationSystem.restype  = None

    lib.createScenario.argtypes = [
        POINTER(c_float),
        POINTER(MemorySystem),
        c_int,
        c_float,
    ]
    lib.createScenario.restype = ImaginationScenario

    lib.simulateScenario.argtypes = [
        POINTER(ImaginationScenario),
        POINTER(c_float),
        POINTER(c_float),
        c_int,
        c_int,
    ]
    lib.simulateScenario.restype = None

    lib.evaluateScenarioPlausibility.argtypes = [
        POINTER(ImaginationScenario),
        POINTER(MemorySystem),
    ]
    lib.evaluateScenarioPlausibility.restype = None

    lib.applyImaginationToDecision.argtypes = [
        POINTER(ImaginationSystem),
        POINTER(c_float),
        POINTER(c_float),
        c_int,
    ]
    lib.applyImaginationToDecision.restype = c_float

    lib.updateImaginationCreativity.argtypes = [
        POINTER(ImaginationSystem),
        c_float,
        c_float,
    ]
    lib.updateImaginationCreativity.restype = None

    lib.blendImaginedOutcomes.argtypes = [
        POINTER(ImaginedOutcome),
        c_int,
        POINTER(c_float),
    ]
    lib.blendImaginedOutcomes.restype = None

    lib.isScenarioCoherent.argtypes = [
        POINTER(ImaginationScenario),
        c_float,
    ]
    lib.isScenarioCoherent.restype = c_bool

    lib.adjustNeuronsWithImagination.argtypes = [
        POINTER(c_float),
        POINTER(ImaginedOutcome),
        c_int,
        c_float,
    ]
    lib.adjustNeuronsWithImagination.restype = None


def serialize_outcome(outcome: ImaginedOutcome) -> dict:
    return {
        "vector": [
            float(outcome.vector[i])
            for i in range(MEMORY_VECTOR_SIZE)
        ],
        "probability":  float(outcome.probability),
        "confidence":   float(outcome.confidence),
        "impact_score": float(outcome.impact_score),
        "plausibility": float(outcome.plausibility),
        "description":  outcome.description.decode(errors="replace"),
    }


def serialize_scenario(scenario: ImaginationScenario) -> dict:
    n = max(0, min(scenario.num_outcomes, MAX_OUTCOMES_PER_SCENARIO))
    return {
        "num_outcomes":      int(scenario.num_outcomes),
        "divergence_factor": float(scenario.divergence_factor),
        "creativity_level":  float(scenario.creativity_level),
        "outcomes": [
            serialize_outcome(scenario.outcomes[i])
            for i in range(n)
        ],
    }


def serialize_imagination(imag_sys) -> dict:
    s = imag_sys.contents
    n = max(0, min(s.num_scenarios, MAX_SCENARIOS))
    return {
        "num_scenarios":             int(s.num_scenarios),
        "current_scenario":          int(s.current_scenario),
        "creativity_factor":         float(s.creativity_factor),
        "coherence_threshold":       float(s.coherence_threshold),
        "novelty_weight":            float(s.novelty_weight),
        "memory_influence":          float(s.memory_influence),
        "identity_influence":        float(s.identity_influence),
        "active":                    bool(s.active),
        "steps_simulated":           int(s.steps_simulated),
        "total_scenarios_generated": int(s.total_scenarios_generated),
        "current_scenario_name":
            s.current_scenario_name.decode(errors="replace"),
        "divergence_history": [
            float(s.divergence_history[i])
            for i in range(DIVERGENCE_HISTORY_SIZE)
        ],
        "scenarios": [
            serialize_scenario(s.scenarios[i])
            for i in range(n)
        ],
    }
