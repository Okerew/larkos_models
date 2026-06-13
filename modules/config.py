import numpy as np
import torch

INPUT_SIZE  = 6
MAX_NEURONS = 8
NUM_REGIONS = 2

N_PREFIX = 8

DECISION_CANDIDATES = 4
BASE_LR             = 0.01
HIDDEN_DIM          = 32

EMBED_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_DIM   = 384
PROJ_DIM    = 32
D_NODE = 8

FOURIER_ENCODINGS = 4
# 4 frames of attended history. TEMPORAL_WINDOW enters MODEL_INPUT_DIM
# linearly and attention memory quadratically — at 8N the seq_len is
# TEMPORAL_WINDOW * 2 * INPUT_SIZE * FOURIER_ENCODINGS = 192, well
# within budget.
TEMPORAL_WINDOW   = 4
FOURIER_OUT_DIM   = 2 * INPUT_SIZE * FOURIER_ENCODINGS  # 48
MODEL_INPUT_DIM   = TEMPORAL_WINDOW * FOURIER_OUT_DIM     # 192

# Temporal-axis attention encoder. d_model = FOURIER_OUT_DIM = 48; nhead
# must divide d_model so 4 is the sane pick at this scale.
TEMPORAL_NHEAD  = 4
TEMPORAL_LAYERS = 2
TEMPORAL_DIM_FF = 64

EMA_DECAY   = 0.995

MAML_INNER_LR    = 0.001
MAML_INNER_STEPS = 3

COSINE_T_MAX = 50

MEM_WEIGHT_RATIO_BASE  = 0.2
MEM_WEIGHT_RATIO_RANGE = 0.7

EMOTION_LOVE     = 0
EMOTION_HATE     = 1
EMOTION_SURPRISE = 2

# Model architecture
VOCAB_SIZE = 256
D_MODEL    = 64
NHEAD      = 2
N_LAYERS   = 2
DIM_FF     = 64
DROPOUT    = 0.1

# Training. FUSION_DIM = BAND_Q + BAND_M after BAND_N was retired from
# cognitive_fuse — see Documents/scaling.md.
FUSION_DIM          = 64
MC_DROPOUT_T       = 10
EXPLORE_THRESHOLD  = 0.15
INTERNAL_DIM       = MAX_NEURONS

# Graph-reasoning fusion head: attends over per-neuron tokens from the
# GAT plus a handful of context tokens (band_q, band_m, driver).
# Head capacity bumped (32→64 d_model, +1 layer, 64→128 dim_ff) after
# the 0.3 refactor dropped the absolute loss floor on harder domains;
# the sequence is MAX_NEURONS+3 tokens, so the head needs enough width
# to mix per-neuron content with the Q/M/driver context.
FUSE_GRAPH_DMODEL = 64
FUSE_GRAPH_NHEAD  = 4
FUSE_GRAPH_LAYERS = 3
FUSE_GRAPH_DIM_FF = 128

# Neuron-graph attention layer. Hand-rolled GAT over the MAX_NEURONS
# graph exposed by backend_state.get_neurons().
GAT_HEADS  = 2
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

# Backend shared constants
MAX_CONNECTIONS    = 6
MEMORY_VECTOR_SIZE = 2 * MAX_NEURONS + INPUT_SIZE  # 288 (2*128+32)

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
    The first few channels are filled from live backend state;
    remaining slots are zero-padded (the online normalizer adapts).
    """
    mean_weight = (
        float(np.mean(np.abs(weights_flat)))
        if weights_flat.size else 0.0
    )
    st_entries = len(
        mem_state.get("short_term", {}).get("entries", [])
    )
    st_churn = (st_entries / max(mem_capacity, 1)) * 2.0 - 1.0

    phase = np.sin(step_counter * (2.0 * np.pi / 64.0))

    channels = np.zeros(input_size, dtype=np.float32)
    channels[0] = np.tanh(np.mean(states))
    channels[1] = np.tanh(np.mean(outputs))
    channels[2] = np.tanh(np.std(states))
    channels[3] = np.tanh(mean_weight)
    channels[4] = float(phase)
    channels[5] = st_churn
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
ACTIVATION_HISTORY_SIZE   = 50
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
FEATURE_VECTOR_SIZE       = 128
CONTEXT_VECTOR_SIZE       = 256
MEMORY_CAPACITY           = 100

# Self identity
PATTERN_SIZE              = 3
EXPERIENCE_VECTOR_SIZE    = 256

# Reflections
REASONING_SIZE            = 1024
HISTORY_SIZE              = 100

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
BAND_Q = 22
BAND_N = 21

# Test-data domains
TEST_DATA = "test_data"
CKPT_DIR = "test_checkpoints"
