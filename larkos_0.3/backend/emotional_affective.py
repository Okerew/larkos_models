from ctypes import (
    CDLL, Structure,
    c_float, c_uint, c_uint32, c_int, c_bool, c_char,
    c_void_p, POINTER,
)

from modules.config import (
    EMOTION_LOVE, EMOTION_HATE, EMOTION_SURPRISE,
    EMOTION_HISTORY_SIZE,
    MAX_EMOTION_ATTRACTORS,
    MAX_ATTACHMENT_BONDS,
    MAX_EMOTION_TYPES,
    MAX_LINKED_ATTRACTORS,
    MAX_BOND_SHARED_HISTORY,
)


class EmotionVector(Structure):
    _fields_ = [
        ("valence",        c_float),
        ("arousal",        c_float),
        ("dominance",      c_float),
        ("complexity",     c_float),
        ("temporal_depth", c_float),
        ("duration_steps", c_uint32),
        ("stability",      c_float),
        ("momentum",       c_float * 3),
    ]


class AttachmentBond(Structure):
    _fields_ = [
        ("entity_id",                   c_uint32),
        ("entity_name",                 c_char * 64),
        ("attachment_strength",         c_float),
        ("trust",                       c_float),
        ("familiarity",                 c_float),
        ("dependency",                  c_float),
        ("care_investment",             c_float),
        ("loss_cost",                   c_float),
        ("shared_history_index",        c_uint32),
        ("emotional_resonance",         c_float),
        ("conflict_history",            c_float),
        ("interaction_count",           c_uint32),
        ("last_interaction_valence",    c_float),
        ("predicted_behavior_alignment",c_float),
        ("emotional_debt",              c_float),
    ]


class EmotionAttractor(Structure):
    _fields_ = [
        ("attractor_id",            c_uint32),
        ("attractor_name",          c_char * 64),
        ("center_point",            EmotionVector),
        ("basin_strength",          c_float),
        ("entry_threshold",         c_float),
        ("exit_threshold",          c_float),
        ("stability_factor",        c_float),
        ("visit_count",             c_uint32),
        ("average_duration",        c_float),
        ("context_weights",         POINTER(c_float)),
        ("context_dim",             c_uint32),
        ("is_pathological",         c_bool),
        ("reinforcement_rate",      c_float),
        ("linked_attractors",       c_uint32 * MAX_LINKED_ATTRACTORS),
        ("transition_probabilities",c_float  * MAX_LINKED_ATTRACTORS),
    ]


class AffectiveSystem(Structure):
    _fields_ = [
        ("current_state",                EmotionVector),
        ("base_state",                   EmotionVector),
        ("attractors",                   POINTER(EmotionAttractor)),
        ("num_attractors",               c_uint32),
        ("current_attractor_id",         c_uint32),
        ("steps_in_current_attractor",   c_uint32),
        ("history",                      EmotionVector * EMOTION_HISTORY_SIZE),
        ("history_index",                c_uint32),
        ("affective_embeddings",         POINTER(c_float)),
        ("embedding_dim",                c_uint32),
        ("plasticity",                   c_float),
        ("self_complexity",              c_float),
        ("bonds",                        POINTER(AttachmentBond)),
        ("num_bonds",                    c_uint32),
        ("max_bonds",                    c_uint32),
        ("relational_bias",              c_float),
        ("predictive_commitment_weight", c_float),
        ("subconscious_influence",       c_float),
    ]


class EmotionState(Structure):
    _fields_ = [
        ("intensity",          c_float),
        ("decay_rate",         c_float),
        ("influence_factor",   c_float),
        ("threshold",          c_float),
        ("previous_intensity", c_float),
        ("momentum",           c_float),
        ("last_update",        c_uint),
    ]


class EmotionalSystem(Structure):
    _fields_ = [
        ("emotions",             EmotionState * MAX_EMOTION_TYPES),
        ("cognitive_impact",     c_float),
        ("emotional_regulation", c_float),
        ("emotional_memory",     (c_float * 10) * MAX_EMOTION_TYPES),
        ("memory_index",         c_int),
    ]


def bind(lib: CDLL, max_neurons: int):
    lib.initializeAffectiveSystem.argtypes = [c_uint32]
    lib.initializeAffectiveSystem.restype  = POINTER(AffectiveSystem)

    lib.initializeEmotionalSystem.argtypes = []
    lib.initializeEmotionalSystem.restype  = POINTER(EmotionalSystem)

    lib.triggerEmotion.argtypes = [
        POINTER(EmotionalSystem),
        c_int,
        c_float,
        c_uint,
    ]
    lib.triggerEmotion.restype = None

    lib.updateEmotionalMemory.argtypes = [POINTER(EmotionalSystem)]
    lib.updateEmotionalMemory.restype  = None

    lib.calculateEmotionalBias.argtypes = [
        POINTER(EmotionalSystem),
        POINTER(c_float),
        c_int,
    ]
    lib.calculateEmotionalBias.restype = c_float

    lib.detectEmotionalTriggers.argtypes = [
        POINTER(EmotionalSystem),
        c_void_p,
        POINTER(c_float),
        c_int,
        c_uint,
        c_float,
        POINTER(AffectiveSystem),
        c_void_p,
    ]
    lib.detectEmotionalTriggers.restype = None

    lib.applyEmotionalProcessing.argtypes = [
        POINTER(EmotionalSystem),
        c_void_p,
        c_int,
        POINTER(c_float),
        c_float,
        c_float,
        POINTER(AffectiveSystem),
    ]
    lib.applyEmotionalProcessing.restype = None

    lib.updateAttractorDynamics.argtypes = [
        POINTER(AffectiveSystem),
        POINTER(c_float),
        c_uint32,
    ]
    lib.updateAttractorDynamics.restype = None

    lib.updateAffectiveComplexity.argtypes = [
        POINTER(AffectiveSystem),
        c_uint32,
    ]
    lib.updateAffectiveComplexity.restype = None

    lib.updateEmotionMomentum.argtypes = [
        POINTER(EmotionVector),
        POINTER(EmotionVector),
        c_float,
    ]
    lib.updateEmotionMomentum.restype = None

    lib.reshapeEmbeddingsWithEmotion.argtypes = [
        POINTER(AffectiveSystem),
        POINTER(c_float),
        c_uint32,
    ]
    lib.reshapeEmbeddingsWithEmotion.restype = None

    lib.updateAttachmentBond.argtypes = [
        POINTER(AffectiveSystem),
        POINTER(AttachmentBond),
        c_float,
        c_float,
        c_float,
    ]
    lib.updateAttachmentBond.restype = None

    lib.integrateAttachmentsIntoIdentity.argtypes = [
        POINTER(AffectiveSystem),
        POINTER(c_float),
        c_uint32,
    ]
    lib.integrateAttachmentsIntoIdentity.restype = None

    lib.h_iga.argtypes = [
        c_void_p,
        POINTER(AffectiveSystem),
        POINTER(EmotionalSystem),
        c_int,
    ]
    lib.h_iga.restype = c_float

    lib.freeEmotionalSystem.argtypes = [POINTER(EmotionalSystem)]
    lib.freeEmotionalSystem.restype  = None


def _serialize_emotion_vector(ev: EmotionVector) -> dict:
    return {
        "valence":        float(ev.valence),
        "arousal":        float(ev.arousal),
        "dominance":      float(ev.dominance),
        "complexity":     float(ev.complexity),
        "temporal_depth": float(ev.temporal_depth),
        "duration_steps": int(ev.duration_steps),
        "stability":      float(ev.stability),
        "momentum":       [float(ev.momentum[i]) for i in range(3)],
    }


def _serialize_bond(bond: AttachmentBond) -> dict:
    return {
        "entity_id":   int(bond.entity_id),
        "entity_name": bond.entity_name.decode(errors="replace"),
        "attachment_strength":          float(bond.attachment_strength),
        "trust":                        float(bond.trust),
        "familiarity":                  float(bond.familiarity),
        "dependency":                   float(bond.dependency),
        "care_investment":              float(bond.care_investment),
        "loss_cost":                    float(bond.loss_cost),
        "shared_history_index":         int(bond.shared_history_index),
        "emotional_resonance":          float(bond.emotional_resonance),
        "conflict_history":             float(bond.conflict_history),
        "interaction_count":            int(bond.interaction_count),
        "last_interaction_valence":     float(bond.last_interaction_valence),
        "predicted_behavior_alignment": float(
            bond.predicted_behavior_alignment
        ),
        "emotional_debt": float(bond.emotional_debt),
    }


def _serialize_attractor(attr: EmotionAttractor) -> dict:
    return {
        "attractor_id":   int(attr.attractor_id),
        "attractor_name": attr.attractor_name.decode(errors="replace"),
        "center_point":   _serialize_emotion_vector(attr.center_point),
        "basin_strength": float(attr.basin_strength),
        "entry_threshold":    float(attr.entry_threshold),
        "exit_threshold":     float(attr.exit_threshold),
        "stability_factor":   float(attr.stability_factor),
        "visit_count":        int(attr.visit_count),
        "average_duration":   float(attr.average_duration),
        "is_pathological":    bool(attr.is_pathological),
        "reinforcement_rate": float(attr.reinforcement_rate),
        "linked_attractors": [
            int(attr.linked_attractors[i])
            for i in range(MAX_LINKED_ATTRACTORS)
        ],
        "transition_probabilities": [
            float(attr.transition_probabilities[i])
            for i in range(MAX_LINKED_ATTRACTORS)
        ],
    }


def serialize_affective(aff_sys) -> dict:
    s = aff_sys.contents
    bonds = [
        _serialize_bond(s.bonds[i])
        for i in range(int(s.num_bonds))
    ]
    attractors = [
        _serialize_attractor(s.attractors[i])
        for i in range(int(s.num_attractors))
    ]
    return {
        "current_state":  _serialize_emotion_vector(s.current_state),
        "base_state":     _serialize_emotion_vector(s.base_state),
        "plasticity":     float(s.plasticity),
        "self_complexity":float(s.self_complexity),
        "relational_bias":float(s.relational_bias),
        "predictive_commitment_weight": float(
            s.predictive_commitment_weight
        ),
        "subconscious_influence": float(s.subconscious_influence),
        "current_attractor_id":   int(s.current_attractor_id),
        "steps_in_current_attractor": int(s.steps_in_current_attractor),
        "num_bonds":      int(s.num_bonds),
        "bonds":          bonds,
        "num_attractors": int(s.num_attractors),
        "attractors":     attractors,
    }


def serialize_emotional(emo_sys) -> dict:
    s = emo_sys.contents
    emotions = []
    for i in range(MAX_EMOTION_TYPES):
        e = s.emotions[i]
        emotions.append({
            "intensity":          float(e.intensity),
            "decay_rate":         float(e.decay_rate),
            "influence_factor":   float(e.influence_factor),
            "threshold":          float(e.threshold),
            "previous_intensity": float(e.previous_intensity),
            "momentum":           float(e.momentum),
            "last_update":        int(e.last_update),
            "memory_trace": [
                float(s.emotional_memory[i][j]) for j in range(10)
            ],
        })
    return {
        "cognitive_impact":     float(s.cognitive_impact),
        "emotional_regulation": float(s.emotional_regulation),
        "memory_index":         int(s.memory_index),
        "emotions":             emotions,
    }
