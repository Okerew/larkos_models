# Scaling Guide

When increasing `MAX_NEURONS`, `INPUT_SIZE`, or `MAX_CONNECTIONS`,
these constants must be scaled in lockstep to keep the architecture balanced.

---

## What auto-derives (safe: no extra edits)

| Constant | Formula | File |
|---|---|---|
| `MEMORY_VECTOR_SIZE` | `2 * MAX_NEURONS + INPUT_SIZE` | `config.py`, `definitions.h`, `fusion_mechanism.c` |
| `FOURIER_OUT_DIM` | `2 * INPUT_SIZE * FOURIER_ENCODINGS` | `config.py` |
| `MODEL_INPUT_DIM` | `TEMPORAL_WINDOW * FOURIER_OUT_DIM` | `config.py` |
| `INTERNAL_DIM` | `= MAX_NEURONS` | `config.py` |
| `NEURON_STRIDE` | `NEURON_FIELDS + MAX_CONNECTIONS * 2` | `fusion_mechanism.c` |
| `MAX_NEURON_FLAT` | `MAX_NEURONS * NEURON_STRIDE` | `fusion_mechanism.c` |

---

## Rule of thumb ratios

Keep these roughly constant when scaling:

```
D_MODEL           ≈ MAX_NEURONS × 8–16
FUSION_DIM        ≈ MAX_NEURONS × 8–16
FOURIER_ENCODINGS ≈ MAX_NEURONS × 0.5–1
NHEAD             ≈ D_MODEL / 32     (must divide D_MODEL)
FUSE_NHEAD        ≈ FUSION_DIM / 16  (must divide FUSION_DIM)
EMBEDDING_SIZE    ≈ D_MODEL / 4
NUM_HEADS         ≈ EMBEDDING_SIZE / 2
HIDDEN_DIM        ≈ D_MODEL × 0.5
```

---

## What must be manually edited

### 1. `include/definitions.h`
- `MAX_NEURONS` — desired neuron count
- `MAX_CONNECTIONS` — connections per neuron (≈ ratio × MAX_NEURONS)
- `INPUT_SIZE` — input feature count (match `build_input_tensor()`)
- `FUSION_DIM` — fusion bottleneck width (must be large enough)
- `EMBEDDING_SIZE` — word/token embedding dimension
- `NUM_HEADS` — C-side attention heads (must divide `EMBEDDING_SIZE`)

### 2. `modules/fusion_mechanism/fusion_mechanism.c`
- `FUSION_DIM` — must match `config.py`
- `BAND_Q`, `BAND_N`, `BAND_M` — must sum to `FUSION_DIM`, update in lockstep with `_FusionTransformerHead._BAND_*` in `modules/training.py`

### 3. `modules/config.py`
- Same constants as in the include/definitions.h section
- `FOURIER_ENCODINGS` — frequency bands per input dim
- `TEMPORAL_WINDOW` — context steps stacked in input
- `D_MODEL`, `NHEAD`, `N_LAYERS`, `DIM_FF` — transformer size
- `FUSION_DIM`, `FUSE_NHEAD`, `FUSE_DIM_FF` — fusion head size
- `HIDDEN_DIM` — NeuralBlock hidden size
- `BAND_Q/N/M`- must match C-side `BAND_*` values and sum to `FUSION_DIM`
- `build_input_tensor()` — **EDIT THIS FUNCTION** when you change `INPUT_SIZE`/`MAX_NEURONS`. Must return an array of exactly `input_size` floats. The default emits 6 channels (mean state, mean output, activity spread, mean weight, temporal phase, memory churn) and zero-pads the rest.

---

## Optimizer re-tuning

Scaling up the architecture increases gradient variance, deeper layers get much larger updates than shallow ones. **Always reduce `BASE_LR` when scaling.**

### Rule of thumb

```
BASE_LR ≈ previous_BASE_LR / (D_MODEL_scale_factor × √N_LAYERS_scale_factor)
```

Example: D_MODEL 64→256 (4×), N_LAYERS 2→4 (2×):
- `BASE_LR` 0.01 → 0.002
- `MAML_INNER_LR` 0.001 → 0.0003

### Files to edit

| File | Key |
|---|---|
| `modules/config.py` | `BASE_LR`, `MAML_INNER_LR` |

---

## Example: scaling for 32 neurons

| Constant | Current (8) | Scaled (32) |
|---|---|---|
| `MAX_NEURONS` | 8 | 32 |
| `MAX_CONNECTIONS` | 6 | 16 |
| `INPUT_SIZE` | 6 | 16 |
| `FUSION_DIM` | 64 | 256 |
| `D_MODEL` | 64 | 256 |
| `NHEAD` | 2 | 8 |
| `DIM_FF` | 64 | 512 |
| `N_LAYERS` | 2 | 4 |
| `FUSE_NHEAD` | 4 | 16 |
| `FUSE_DIM_FF` | 64 | 512 |
| `EMBEDDING_SIZE` | 16 | 64 |
| `NUM_HEADS` | 8 | 8 |
| `FOURIER_ENCODINGS` | 4 | 16 |
| `HIDDEN_DIM` | 32 | 128 |
| `BAND_Q` | 22 | 85 |
| `BAND_N` | 21 | 86 |
| `BAND_M` | 21 | 85 |

(Sum of BAND_* must equal FUSION_DIM; adjust ±1 as needed.)
