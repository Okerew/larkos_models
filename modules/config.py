import numpy as np
import torch

INPUT_SIZE  = 32
MAX_NEURONS = 128
NUM_REGIONS = 2

N_PREFIX = 8

DECISION_CANDIDATES = 4
# 1B scale-up: BASE_LR /= D_MODEL_scale * sqrt(N_LAYERS_scale)
# = 0.0003 / (2 * sqrt(2)) ~= 1e-4 (Documents/scaling.md rule)
BASE_LR             = 0.0001
HIDDEN_DIM          = 512

EMBED_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_DIM   = 384
PROJ_DIM    = 256
D_NODE = 8

FOURIER_ENCODINGS = 4
# 2 frames of attended history. TEMPORAL_WINDOW dropped to 2 so
# MODEL_INPUT_DIM stays within VRAM budget at 128N.
TEMPORAL_WINDOW   = 2
FOURIER_OUT_DIM   = 2 * INPUT_SIZE * FOURIER_ENCODINGS  # 256
MODEL_INPUT_DIM   = TEMPORAL_WINDOW * FOURIER_OUT_DIM     # 512

# Temporal-axis attention encoder. d_model = FOURIER_OUT_DIM = 256;
# nhead must divide d_model.
TEMPORAL_NHEAD  = 8
TEMPORAL_LAYERS = 2
TEMPORAL_DIM_FF = 512

EMA_DECAY   = 0.995

# scaled by the same 2*sqrt(2) factor as BASE_LR
MAML_INNER_LR    = 0.00001
MAML_INNER_STEPS = 3

COSINE_T_MAX = 120

MEM_WEIGHT_RATIO_BASE  = 0.2
MEM_WEIGHT_RATIO_RANGE = 0.7

EMOTION_LOVE     = 0
EMOTION_HATE     = 1
EMOTION_SURPRISE = 2

# Model architecture. ~805M trainable params at 128N; MAX_NEURONS (and
# everything derived from it - FUSION_DIM, MODEL_INPUT_DIM, the C-side
# constants) deliberately does NOT move with this scale-up. The encoder
# sequence length stays 512: FOURIER_ENCODINGS=4 is not negotiable,
# activations scale quadratically with it (Documents/scaling.md).
VOCAB_SIZE = 256
D_MODEL    = 2048
NHEAD      = 64
N_LAYERS   = 16
DIM_FF     = 8192
DROPOUT    = 0.1

# Training. FUSION_DIM = BAND_Q + BAND_M after BAND_N was retired from
# cognitive_fuse — see Documents/scaling.md.
FUSION_DIM          = 1024
MC_DROPOUT_T       = 4
EXPLORE_THRESHOLD  = 0.15
INTERNAL_DIM       = MAX_NEURONS

# Graph-reasoning fusion head: attends over per-neuron tokens from the
# GAT plus a handful of context tokens (band_q, band_m, driver).
# At 128N the sequence is MAX_NEURONS+3 tokens long, so the head needs
# enough width to mix per-neuron content with the Q/M/driver context.
FUSE_GRAPH_DMODEL = 256
FUSE_GRAPH_NHEAD  = 8
FUSE_GRAPH_LAYERS = 3
FUSE_GRAPH_DIM_FF = 512

# Neuron-graph attention layer. Hand-rolled GAT over the MAX_NEURONS
# graph exposed by backend_state.get_neurons().
GAT_HEADS  = 4
GAT_LAYERS = 2

# Text codec
GPT2_HIDDEN  = 768
TEXT_MAX_NEW = 20

# Epoch intervals for side-effects / verification
EMOTION_LOG_INTERVAL   = 3
VERIFY_INTERVAL        = 5
TARGET_FREEZE_INTERVAL = 5

# Verification health thresholds
GRAD_FLOOR       = 1e-12
ACT_UPPER        = 1e4
OVERFIT_STEPS    = 40
OVERFIT_THRESH   = 0.01
LOSS_SPIKE_RATIO     = 3.0
LOSS_PLATEAU_DELTA   = 1e-4
LOSS_PLATEAU_WINDOW  = 10
LOSS_DIVERGE_FLOOR   = 50.0
LR_SENSITIVITY_DELTA = 0.05
GRAD_SMOOTH_ALPHA    = 0.1
DEAD_WINDOW          = 8

# Data pipeline
FALLBACK = (
    "The network explores a latent space shaped by experience."
)
SHUFFLE_BUFFER = 1000
TEXT_COLUMN_HINTS = (
    "text", "content", "body", "sentence",
    "document", "passage", "input", "output",
)

# Backend shared constants — auto-derives from INPUT_SIZE and MAX_NEURONS
MAX_CONNECTIONS    = 32
MEMORY_VECTOR_SIZE = 2 * MAX_NEURONS + INPUT_SIZE  # 288

# Input tensor builder, EDIT THIS when you change INPUT_SIZE / MAX_NEURONS
def build_input_tensor(
    states:        np.ndarray,
    outputs:       np.ndarray,
    weights_flat:  np.ndarray,
    step_counter:  int,
    mem_state:     dict,
    mem_capacity:  int,
    input_size:    int,
) -> np.ndarray:
    """Build the observation vector fed into the model each step.

    Must return an array of exactly ``input_size`` floats.

    Layout (32 channels — channels past ``input_size`` are dropped,
    channels past the feature count are zero-padded):

      0-5   original 6 channels (mean state, mean output, std state,
            mean |weight|, phase, short-term memory churn) — kept so a
            shrink back to INPUT_SIZE=6 still works
      6-11  extra activity stats (std outputs, min/max state/output,
            output-state drift)
      12-15 weight stats (signed mean, std, sparsity, positive ratio)
      16-21 multi-scale temporal phases (sin/cos at periods 8, 256, 1024)
      22-25 memory hierarchy (medium-term churn, long-term churn,
            mean fill, short-vs-long drift)
      26-31 distribution shape (state/output skewness, fraction positive,
            output saturation fraction, normalized L2 of state/output)
    """
    s_mean = float(np.mean(states))
    s_std  = float(np.std(states))
    s_max  = float(np.max(states))
    s_min  = float(np.min(states))
    o_mean = float(np.mean(outputs))
    o_std  = float(np.std(outputs))
    o_max  = float(np.max(outputs))
    o_min  = float(np.min(outputs))

    def _skew(x: np.ndarray, mu: float, sigma: float) -> float:
        if sigma < 1e-6:
            return 0.0
        return float(np.mean(((x - mu) / sigma) ** 3))

    s_skew = _skew(states, s_mean, s_std)
    o_skew = _skew(outputs, o_mean, o_std)

    n_state = max(states.size, 1)
    frac_states_pos  = float(np.count_nonzero(states > 0)) / n_state
    frac_outputs_sat = (
        float(np.count_nonzero(np.abs(outputs) > 0.5)) / n_state
    )
    s_l2 = float(np.linalg.norm(states))  / np.sqrt(n_state)
    o_l2 = float(np.linalg.norm(outputs)) / np.sqrt(n_state)

    if weights_flat.size:
        w_mean_abs = float(np.mean(np.abs(weights_flat)))
        w_mean     = float(np.mean(weights_flat))
        w_std      = float(np.std(weights_flat))
        w_sparsity = (
            float(np.count_nonzero(np.abs(weights_flat) < 0.01))
            / weights_flat.size
        )
        w_pos_ratio = (
            float(np.count_nonzero(weights_flat > 0))
            / weights_flat.size
        )
    else:
        w_mean_abs = w_mean = w_std = 0.0
        w_sparsity = w_pos_ratio = 0.5

    def _churn(level: str) -> float:
        # mem_state may be the full serialization or the sizes-only
        # stats dict (backend_state.get_memory_stats) - prefer the
        # reported size, fall back to counting entries
        lvl = mem_state.get(level, {})
        if "size" in lvl:
            entries = int(lvl["size"])
        else:
            entries = len(lvl.get("entries", []))
        return (entries / max(mem_capacity, 1)) * 2.0 - 1.0

    st_churn = _churn("short_term")
    mt_churn = _churn("medium_term")
    lt_churn = _churn("long_term")
    mean_churn = (st_churn + mt_churn + lt_churn) / 3.0

    two_pi = 2.0 * np.pi
    def _sin(period: float) -> float:
        return float(np.sin(step_counter * (two_pi / period)))
    def _cos(period: float) -> float:
        return float(np.cos(step_counter * (two_pi / period)))

    features = [
        # 0-5 : original layout
        np.tanh(s_mean),
        np.tanh(o_mean),
        np.tanh(s_std),
        np.tanh(w_mean_abs),
        _sin(64.0),
        st_churn,
        # 6-11 : extra activity stats
        np.tanh(o_std),
        np.tanh(s_max),
        np.tanh(s_min),
        np.tanh(o_max),
        np.tanh(o_min),
        np.tanh(o_mean - s_mean),
        # 12-15 : weight stats
        np.tanh(w_mean),
        np.tanh(w_std),
        2.0 * w_sparsity  - 1.0,
        2.0 * w_pos_ratio - 1.0,
        # 16-21 : multi-scale temporal phases
        _sin(8.0),     _cos(8.0),
        _sin(256.0),   _cos(256.0),
        _sin(1024.0),  _cos(1024.0),
        # 22-25 : memory hierarchy
        mt_churn,
        lt_churn,
        float(np.clip(mean_churn, -1.0, 1.0)),
        np.tanh(st_churn - lt_churn),
        # 26-31 : distribution shape
        np.tanh(s_skew),
        np.tanh(o_skew),
        2.0 * frac_states_pos  - 1.0,
        2.0 * frac_outputs_sat - 1.0,
        np.tanh(s_l2),
        np.tanh(o_l2),
    ]

    channels = np.zeros(input_size, dtype=np.float32)
    n_copy = min(len(features), input_size)
    channels[:n_copy] = np.array(features[:n_copy], dtype=np.float32)
    return channels

# Imagination
MAX_SCENARIOS             = 10
MAX_OUTCOMES_PER_SCENARIO = 10
SCENARIO_NAME_SIZE        = 100
OUTCOME_DESC_SIZE         = 256
DIVERGENCE_HISTORY_SIZE   = 100

# Emotional / affective
EMOTION_HISTORY_SIZE      = 100
MAX_EMOTION_ATTRACTORS    = 20
MAX_ATTACHMENT_BONDS      = 50
MAX_EMOTION_TYPES         = 8
MAX_LINKED_ATTRACTORS     = 5
MAX_BOND_SHARED_HISTORY   = 32

# Specialization
MAX_SPECIALIZATIONS       = 8
MAX_SPECIALIZED_NEURONS   = 64
ACTIVATION_HISTORY_SIZE   = 200
SPEC_NONE                 = 0
SPEC_PATTERN_DETECTOR     = 1
SPEC_FEATURE_EXTRACTOR    = 2
SPEC_TEMPORAL_PROCESSOR   = 3
SPEC_CONTEXT_INTEGRATOR   = 4
SPEC_DECISION_MAKER       = 5
SPEC_MEMORY_ENCODER       = 6
SPEC_EMOTIONAL_PROCESSOR  = 7
SPEC_PREDICTION_GENERATOR = 8

# Memory
FEATURE_VECTOR_SIZE       = 512
CONTEXT_VECTOR_SIZE       = 1024
# 1M entries: runtime arg to createMemorySystem, no C recompile.
# Costs ~2.3 GB of C-side RAM (hierarchy tiers + flat ring) and a
# ~2.3 GB memory.bin on save; Python-side hot paths read sizes only
# (backend_state.get_memory_stats) and serialize_state truncates
# entry lists at MAX_SERIALIZED_ENTRIES per level.
MEMORY_CAPACITY           = 1_000_000

# Self identity
PATTERN_SIZE              = 3
EXPERIENCE_VECTOR_SIZE    = 1024

# Reflections
REASONING_SIZE            = 4096
HISTORY_SIZE              = 500

# Decision path
NUM_PATHS                 = 5
MAX_DECISION_STEPS        = 20

# Meta
HISTORY_LENGTH            = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modules whose grad norm is structurally orders of magnitude below
# the in-graph fusion_transformer and should NOT trigger the grad
# imbalance warning. Two flavours qualify:
#   - cut-then-aux: gradient is cut from the main loss by the C-side
#     detach and only re-enters via aux_loss (embed_weight_net,
#     cross_attn, text_proj).
#   - far-upstream: gradient flows through many layers of the model
#     before reaching the module, so chain-rule attenuation alone
#     puts it 1e4-1e5 below the head (temporal_encoder).
# In both cases the module is still tracked for dead_module /
# oscillation checks — only the head-vs-this-module ratio is skipped.
DETACHED_LABELS = frozenset({
    "embed_weight_net",
    "cross_attn",
    "text_proj",
    "temporal_encoder",
})

CHECKPOINT_VERSION = 1

# Band layout must match the BAND_Q / BAND_M split in
# fusion_mechanism.c. If those change in C these must change in lockstep.
# BAND_N was removed when the GAT replaced the C-side neuron projection;
# FUSION_DIM = BAND_Q + BAND_M.
# 128N: BAND_Q + BAND_M = 1024 = FUSION_DIM
BAND_Q = 512
BAND_M = 512

# Test-data domains
TEST_DATA = "test_data"
CKPT_DIR = "test_checkpoints"
SAMPLE_POOL_SIZE = 8

# Dynamic epoch controller (modules/epoch_controller.py). Active when
# training_loop gets epochs=None instead of a fixed count. Statistics
# are aligned to "cycles" of SAMPLE_POOL_SIZE epochs so one full pass
# over the prompt pool is compared against the previous one and
# per-prompt loss differences cancel out.
DYN_MIN_EPOCHS        = 30    # soft minimum: no stop before this
DYN_SOFT_MAX_EPOCHS   = 120   # soft limit: plateau rules relax past this
DYN_MAX_EPOCHS        = 250   # hard cap: unconditional stop
DYN_PLATEAU_REL_DELTA = 0.02  # cycle-mean improvement below this = flat
DYN_PLATEAU_PATIENCE  = 2     # consecutive flat cycles needed to stop
DYN_TARGET_LOSS       = 0.12  # "good enough" cycle-mean total loss
DYN_TARGET_REFRESH    = 0.20  # "good enough" refresh-epoch base loss
DYN_GOOD_PATIENCE     = 2     # consecutive good cycles needed to stop
# A cycle must beat the best cycle mean seen so far by this relative
# margin, otherwise it counts toward the stagnation stop.
DYN_STAGN_TOLERANCE   = 0.01
DYN_STAGN_PATIENCE    = 4     # non-improving cycles needed to stop
# Trust-gate calibration: healthy training sits at stability ~0.25
# (the C network mutates every epoch by design) and drift ~0.66, so
# the gate only fires on genuinely pathological regimes - the old
# 0.5 floor held every single check and disabled all stop paths.
DYN_STABILITY_FLOOR   = 0.15  # below this the backend is churning wild
DYN_DRIFT_CEILING     = 0.75  # above this the reflection is diverging
DYN_LR_EXHAUSTED      = 2e-5  # scheduler lr at/below this counts as spent

JOURNAL_FILE = "memory_journal.json"

# check_consistency verdict bands for MiniLM cosine: >= SUPPORT means
# the memory holds essentially this claim (same or paraphrase),
# >= RELATED means topically related evidence exists (the caller
# judges agreement from the returned texts), below means nothing
# like it was ever stored. First calibration, tune with use.
SUPPORT_SIM = 0.75
RELATED_SIM = 0.45

# Memory entries handed to the C-side cognitive_fuse, sampled by
# importance. With MEMORY_CAPACITY at 1M the serialized tiers can
# carry tens of thousands of entries; marshalling all of them through
# ctypes every step is pure overhead when the fusion attention only
# ever weighs the salient tail anyway. The C side already takes
# m_count as a runtime arg, so no recompile is needed.
FUSION_MEM_TOP_K = 1024
