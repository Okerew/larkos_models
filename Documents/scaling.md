# Scaling Guide

When increasing `MAX_NEURONS`, `INPUT_SIZE`, or `MAX_CONNECTIONS`,
these constants must be scaled in lockstep to keep the architecture balanced.

---

## Read this first — the sequence-length pitfall

`MODEL_INPUT_DIM` is **not just an embedding width**. In
`modules/model.py:NumericTokenizer`, each scalar in the
`MODEL_INPUT_DIM`-long input vector becomes one transformer token, so

```
encoder sequence length ≈ MODEL_INPUT_DIM + 1   (+1 = CLS)
```

Self-attention is `O(seq_len²)` per layer per head, and the encoder is
re-traversed several times per training step (MAML inner loop ×
`MAML_INNER_STEPS`, plus the MC-dropout sweep, plus the post-MAML adapted
forward). So `MODEL_INPUT_DIM` enters activation memory **quadratically**
and gets multiplied by `N_LAYERS × NHEAD × (a small constant)`.

Because

```
MODEL_INPUT_DIM = TEMPORAL_WINDOW * 2 * INPUT_SIZE * FOURIER_ENCODINGS
```

scaling any of these by 2× scales attention memory by ~4×. Scaling
`FOURIER_ENCODINGS` 16× (4→64) scales attention memory by ~256× — that
alone is enough to OOM most GPUs regardless of `D_MODEL`.

**Therefore: `FOURIER_ENCODINGS` and `TEMPORAL_WINDOW` do NOT scale with
`MAX_NEURONS`. Treat them as fixed hyperparameters of the feature
extractor, sized to the input signal — not to the model.** `INPUT_SIZE`
similarly only grows when you add real backend channels in
`build_input_tensor()`, not as a stand-in for "more capacity".

### Rough VRAM budget for the encoder

Per training step, peak attention activations are dominated by

```
~ N_LAYERS × NHEAD × MODEL_INPUT_DIM² × 4 bytes × ~5
```

(the ×5 covers attn weights + softmax + dropout mask + Q/K/V intermediates
saved for backward). At `MODEL_INPUT_DIM ≈ 512`, `NHEAD=32`, `N_LAYERS=8`
this is ~1 GB. At `MODEL_INPUT_DIM ≈ 16 000` the same formula gives ~1 TB,
which is why anything that inflates `MODEL_INPUT_DIM` — especially
`FOURIER_ENCODINGS` — blows past available VRAM almost regardless of how
small you make the other constants.

To size `MODEL_INPUT_DIM` for your hardware: plug your `NHEAD` and
`N_LAYERS` into the formula above, compare against the VRAM left after
parameters, optimizer state, and the fusion head, and pick the largest
`MODEL_INPUT_DIM` that still leaves headroom. If you need a larger cap
than fits, enable gradient checkpointing before shrinking the other
constants — it trades a 1.3–1.5× slowdown for a major drop in activation
memory.

---

## What auto-derives (safe: no extra edits)

| Constant | Formula | File |
|---|---|---|
| `MEMORY_VECTOR_SIZE` | `2 * MAX_NEURONS + INPUT_SIZE` | `config.py`, `definitions.h`, `fusion_mechanism.c` |
| `FOURIER_OUT_DIM` | `2 * INPUT_SIZE * FOURIER_ENCODINGS` | `config.py` |
| `MODEL_INPUT_DIM` | `TEMPORAL_WINDOW * FOURIER_OUT_DIM` | `config.py` (≡ encoder seq_len, see above) |
| `INTERNAL_DIM` | `= MAX_NEURONS` | `config.py` |
| `NEURON_STRIDE` | `NEURON_FIELDS + MAX_CONNECTIONS * 2` | `fusion_mechanism.c` |
| `MAX_NEURON_FLAT` | `MAX_NEURONS * NEURON_STRIDE` | `fusion_mechanism.c` |

---

## Rule of thumb ratios — things that *do* scale with `MAX_NEURONS`

Keep these roughly constant when scaling:

```
D_MODEL           ≈ MAX_NEURONS × 8–16
FUSION_DIM        ≈ MAX_NEURONS × 8–16
NHEAD             ≈ D_MODEL / 32       (must divide D_MODEL)
FUSE_NHEAD        ≈ FUSION_DIM / 16    (must divide FUSION_DIM)
EMBEDDING_SIZE    ≈ D_MODEL / 4
NUM_HEADS         ≈ EMBEDDING_SIZE / 2 (must divide EMBEDDING_SIZE)
HIDDEN_DIM        ≈ D_MODEL × 0.5
DIM_FF            ≈ D_MODEL × 4
N_LAYERS          ≈ log2(MAX_NEURONS)  (2 → 4 → 6 → 8 across 8 → 32 → 64 → 128)
```

These widen the model and add parameters without touching the encoder
sequence length, so they grow VRAM only linearly.

## Constants that do NOT scale with `MAX_NEURONS`

Set these based on the input signal and your VRAM budget, not on neuron
count:

```
FOURIER_ENCODINGS   4–8        (frequency bands per input channel)
TEMPORAL_WINDOW     2–4        (context steps stacked in input)
INPUT_SIZE          6–32       (real backend feature channels)
MC_DROPOUT_T        4–10       (MC dropout samples per probe)
MAML_INNER_STEPS    2–3        (each step is a full encoder forward+backward)
```

Their product `TEMPORAL_WINDOW × 2 × INPUT_SIZE × FOURIER_ENCODINGS` must
stay within the `MODEL_INPUT_DIM` cap above.

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
- `BAND_Q`, `BAND_M` — must sum to `FUSION_DIM`, update in lockstep with `BAND_Q`/`BAND_M` in `modules/config.py`

### 3. `modules/config.py`
- Same constants as in the include/definitions.h section
- `FOURIER_ENCODINGS` — frequency bands per input dim.
  **Do not scale this with `MAX_NEURONS`.** Keep at 4–8. See the
  sequence-length pitfall section: this enters
  `MODEL_INPUT_DIM = encoder seq_len` linearly and therefore attention
  memory quadratically.
- `TEMPORAL_WINDOW` — context steps stacked in input. Same warning: keep
  at 2–4 unless you've measured the VRAM headroom.
- `D_MODEL`, `NHEAD`, `N_LAYERS`, `DIM_FF` — transformer size
- `FUSION_DIM`, `FUSE_NHEAD`, `FUSE_DIM_FF` — fusion head size
- `HIDDEN_DIM` — NeuralBlock hidden size
- `BAND_Q`, `BAND_M` — must match C-side values and sum to `FUSION_DIM` (`BAND_N` was retired when the GAT replaced the C-side neuron projection)
- `MC_DROPOUT_T` — number of MC dropout forward passes per probe. Each
  one re-runs the encoder; drop to 4 on small GPUs.
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

## Worked examples

| Constant | 8N (baseline) | 32N | 128N (~100M params) |
|---|---|---|---|
| `MAX_NEURONS` | 8 | 32 | 128 |
| `MAX_CONNECTIONS` | 6 | 16 | 32 |
| `INPUT_SIZE` | 6 | 16 | 32 |
| `FUSION_DIM` | 64 | 256 | 1024 |
| `D_MODEL` | 64 | 256 | 1024 |
| `NHEAD` | 2 | 8 | 32 |
| `DIM_FF` | 64 | 1024 | 4096 |
| `N_LAYERS` | 2 | 4 | 8 |
| `FUSE_NHEAD` | 4 | 16 | 64 |
| `FUSE_N_LAYER` | 2 | 2 | 4 |
| `FUSE_DIM_FF` | 64 | 512 | 4096 |
| `EMBEDDING_SIZE` | 16 | 64 | 256 |
| `NUM_HEADS` | 8 | 8 | 16 |
| `HIDDEN_DIM` | 32 | 128 | 512 |
| `PROJ_DIM` | 32 | 64 | 256 |
| `BAND_Q` | 32 | 128 | 512 |
| `BAND_M` | 32 | 128 | 512 |
| `BASE_LR` | 0.01 | 0.002 | 0.0003 |
| `MAML_INNER_LR` | 0.001 | 0.0003 | 0.00003 |
| `FOURIER_ENCODINGS` | 4 | 4 | 4 |
| `TEMPORAL_WINDOW` | 4 | 4 | 2 |
| `MC_DROPOUT_T` | 10 | 10 | 4 |
| `MODEL_INPUT_DIM` (= seq_len) | 192 | 512 | 512 |

Notes:

- `BAND_Q + BAND_M` must equal `FUSION_DIM`; split evenly (each = `FUSION_DIM / 2`).
- `FOURIER_ENCODINGS` and `TEMPORAL_WINDOW` are **deliberately constant or
  even decreasing** across the columns — they belong to the feature
  extractor, not the model, and inflating them is what causes OOM on
  consumer GPUs (see the sequence-length pitfall section).
- The 128N column drops `TEMPORAL_WINDOW` to 2 and `MC_DROPOUT_T` to 4 so
  peak training VRAM stays around ~6 GB, which is the band most consumer
  GPUs sit in. If you have headroom, restore `TEMPORAL_WINDOW=4` and
  `MC_DROPOUT_T=10` for better uncertainty estimates and generalisation.
- Scale thins like memory capacity and history sizes for the C backend stuff also.
