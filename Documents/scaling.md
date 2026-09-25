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

## VRAM diet (applied to the 128N ~100M config)

Five changes cut peak training VRAM from ~6 GB to ~2.5–3 GB at
`MODEL_INPUT_DIM=512`. All are permanent code behaviour now, not
toggles:

1. **bf16 autocast** (`strategies.py:bf16_autocast`, used by
   `training.py` `_forward`/`_backward` and the MAML inner loop).
   Master weights, grads and the optimizer stay fp32; no GradScaler is
   needed because bf16 keeps fp32's exponent range. Tensors crossing
   back to numpy (`.numpy()` on bf16 fails) and the freeze-window
   driver cache are `.float()`-ed explicitly.
2. **Gradient checkpointing** of every `TransformerEncoderLayer` in
   `LarkosModel.forward` (`use_reentrant=False`). This is the big one:
   the `O(seq_len²)` attention activations above are recomputed in
   backward instead of stored. Under `no_grad` (runner, MC probes) the
   checkpoint degrades to a plain call.
3. **EMA shadow lives on CPU** (`model.py:EMAWrapper`). The shadow and
   the apply/restore backup cost zero VRAM; the checkpoint format is
   unchanged (it already saved the shadow detached-to-CPU), but
   `load_checkpoint` now restores it to CPU too.
4. **distilgpt2 runs bf16** (`training.py:_TextCodec`). It is a frozen
   readout, so half precision is free quality-wise; `encode()` casts
   its output back to fp32 so `text_encoding` consumers are untouched.
5. **8-bit AdamW** (`bitsandbytes.optim.AdamW8bit` via
   `training.py:_make_adamw`, falls back to `torch.optim.AdamW` on CPU
   boxes). Optimizer moments are block-quantised to 8 bits — ~1/4 of
   the fp32 state — with identical hyperparameters.

Verify with `python test_vram_diet.py --smoke` (2 epochs, prints
`torch.cuda.max_memory_allocated()`; expect ≤ ~3 GB). Note the smoke
run overwrites the live state files like `main.py` does.

When you scale up (see the 1B section below), items 1, 2 and 5 are
what keep the budget linear in parameters; budget roughly
`params × (4 B weights + 2 B moments + grads/activations)`.

---

## MEMORY_CAPACITY = 1,000,000

`MEMORY_CAPACITY` is a runtime argument to `createMemorySystem` — no C
recompile when it changes. At 1M the C side holds ~2.3 GB (hierarchy
tiers 0.5/0.3/0.2 × capacity + the flat ring, ~1160 B per entry) and
`memory.bin` grows to ~2.3 GB on save. Three Python-side guards keep
the per-step cost flat:

- `backend_state.get_memory_stats()` → `memory.serialize_stats()` is
  the **per-step** read (input-tensor churn channels, tier counts):
  sizes and capacities only, zero entry marshalling.
- `memory.serialize_state()` still powers the API / journal /
  output-prompt paths, but truncates each level's entry list at
  `MAX_SERIALIZED_ENTRIES = 10 000` while reporting the true
  size/capacity. Anything counting entries via `len(entries)`
  under-reports past the cap — use `size` (memory_service does).
- `fusion.py:_mem_entries_to_c` samples the **top 1024 entries by
  importance** before ctypes marshalling, so `cognitive_fuse` never
  sees more than `FUSION_MEM_TOP_K` entries regardless of tier fill.

Verify with `python test_memory_capacity.py --backend`, then re-run
`test_journal_recall.py` / `test_memory_service.py` after a training
run (the journal timestamp→tier join only sees the capped entry
window, so very recent entries may show as `(not in memory)` in recall
output once a tier exceeds 10 000).

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
FUSE_GRAPH_DMODEL ≈ MAX_NEURONS × 2    (refactored head's d_model)
FUSE_GRAPH_NHEAD  ≈ FUSE_GRAPH_DMODEL / 32 (must divide FUSE_GRAPH_DMODEL)
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

### Aside on the temporal encoder

`TEMPORAL_WINDOW` now also controls a small `_TemporalAttentionEncoder`
(in `modules/training.py`) that runs over the `[TEMPORAL_WINDOW,
FOURIER_OUT_DIM]` input history before it is flattened for
`LarkosModel`. Its `d_model = FOURIER_OUT_DIM` (256 today), `nhead =
TEMPORAL_NHEAD = 8`, and only `TEMPORAL_LAYERS = 2` encoder layers,
so its attention activations are tiny compared to `LarkosModel`'s
input encoder. **It is not the OOM driver — the OOM driver remains
the `LarkosModel` encoder, whose sequence length is still
`MODEL_INPUT_DIM + 1`.** The same `2–4` band for `TEMPORAL_WINDOW`
applies.

### Aside on the GAT and BAND_N

The `_NeuronGraphReasoner` (GAT) runs over `MAX_NEURONS` tokens at
`FUSE_GRAPH_DMODEL = 256`. Its attention cost is `O(MAX_NEURONS²)`
per layer, well under `LarkosModel`'s encoder. When `MAX_NEURONS`
scales beyond ~256 the GAT will eventually need its own audit, but
the `LarkosModel` encoder hits the VRAM wall first in every realistic
config.

`BAND_N` is no longer written by `cognitive_fuse` — the GAT replaced
it. `FUSION_DIM` is now `BAND_Q + BAND_M`; the `BAND_N` row in the
worked examples below is kept for historical scaling reference but
the value contributes zero to the live `FUSION_DIM`. When scaling up,
distribute the former `BAND_N` share across `BAND_Q` and `BAND_M` or
adjust `FUSION_DIM` directly.

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
- `FOURIER_ENCODINGS` — frequency bands per input dim.
  **Do not scale this with `MAX_NEURONS`.** Keep at 4–8. See the
  sequence-length pitfall section: this enters
  `MODEL_INPUT_DIM = encoder seq_len` linearly and therefore attention
  memory quadratically.
- `TEMPORAL_WINDOW` — context steps stacked in input. Same warning: keep
  at 2–4 unless you've measured the VRAM headroom.
- `D_MODEL`, `NHEAD`, `N_LAYERS`, `DIM_FF` — transformer size
- `FUSION_DIM`, `FUSE_GRAPH_DMODEL`, `FUSE_GRAPH_NHEAD`,
  `FUSE_GRAPH_LAYERS`, `FUSE_GRAPH_DIM_FF` — graph-reasoning fusion
  head size (replaces the old `FUSE_NHEAD` / `FUSE_N_LAYER` /
  `FUSE_DIM_FF` that were retired with the 3-stream head)
- `GAT_HEADS`, `GAT_LAYERS` — GAT over the neuron graph
- `TEMPORAL_NHEAD`, `TEMPORAL_LAYERS`, `TEMPORAL_DIM_FF` — temporal
  encoder over the input history
- `HIDDEN_DIM` — NeuralBlock hidden size
- `BAND_Q`, `BAND_M` — must match C-side `BAND_*` and sum to
  `FUSION_DIM`. `BAND_N` is retired (the GAT replaced it)
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

| Constant | 8N (baseline) | 32N | 128N (~100M params) | 128N-wide (~805M params) |
|---|---|---|---|---|
| `MAX_NEURONS` | 8 | 32 | 128 | 128 |
| `MAX_CONNECTIONS` | 6 | 16 | 32 | 32 |
| `INPUT_SIZE` | 6 | 16 | 32 | 32 |
| `FUSION_DIM` | 64 | 256 | 1024 | 1024 |
| `D_MODEL` | 64 | 256 | 1024 | 2048 |
| `NHEAD` | 2 | 8 | 32 | 64 |
| `DIM_FF` | 64 | 1024 | 4096 | 8192 |
| `N_LAYERS` | 2 | 4 | 8 | 16 |
| `FUSE_GRAPH_DMODEL` | 32 | 128 | 256 | 256 |
| `FUSE_GRAPH_NHEAD` | 4 | 8 | 8 | 8 |
| `FUSE_GRAPH_LAYERS` | 2 | 2 | 3 | 3 |
| `FUSE_GRAPH_DIM_FF` | 64 | 256 | 512 | 512 |
| `GAT_HEADS` | 2 | 4 | 4 | 4 |
| `GAT_LAYERS` | 2 | 2 | 2 | 2 |
| `EMBEDDING_SIZE` | 16 | 64 | 256 | 256 |
| `NUM_HEADS` | 8 | 8 | 16 | 16 |
| `HIDDEN_DIM` | 32 | 128 | 512 | 512 |
| `PROJ_DIM` | 32 | 64 | 256 | 256 |
| `BAND_Q` | 32 | 128 | 512 | 512 |
| `BAND_M` | 32 | 128 | 512 | 512 |
| `BASE_LR` | 0.01 | 0.002 | 0.0003 | 0.0001 |
| `MAML_INNER_LR` | 0.001 | 0.0003 | 0.00003 | 0.00001 |
| `FOURIER_ENCODINGS` | 4 | 4 | 4 | 4 |
| `TEMPORAL_WINDOW` | 4 | 4 | 2 | 2 |
| `MC_DROPOUT_T` | 10 | 10 | 4 | 4 |
| `MODEL_INPUT_DIM` (= seq_len) | 192 | 512 | 512 | 512 |

Notes:

- Sum of `BAND_*` must equal `FUSION_DIM`; adjust ±1 as needed. (The
  old 342/341 row predates the `FUSION_DIM=1024` restore — the live
  128N split is 512/512 on both the Python and C sides.)
- `FOURIER_ENCODINGS` and `TEMPORAL_WINDOW` are **deliberately constant or
  even decreasing** across the columns — they belong to the feature
  extractor, not the model, and inflating them is what causes OOM on
  consumer GPUs (see the sequence-length pitfall section).
- The 128N columns drop `TEMPORAL_WINDOW` to 2 and `MC_DROPOUT_T` to 4;
  with the VRAM diet applied the ~100M config peaks at ~2.5–3 GB. If
  you have headroom, restore `TEMPORAL_WINDOW=4` and `MC_DROPOUT_T=10`
  for better uncertainty estimates and generalisation.

### The 128N-wide (~805M) column — what made it fit a 12 GB card

This column widens ONLY the transformer (`D_MODEL` 1024→2048,
`N_LAYERS` 8→16, `DIM_FF` 4096→8192, `NHEAD` 32→64 → ≈805M params).
Everything that feeds the encoder sequence length or the C boundary is
frozen: `MAX_NEURONS`, `INPUT_SIZE`, `FUSION_DIM`, `BAND_*`,
`MODEL_INPUT_DIM=512`, `FOURIER_ENCODINGS=4` — so `definitions.h` and
`fusion_mechanism.c` need **no edits and no recompile** for this step.

LR retune follows the rule above: D_MODEL ×2, N_LAYERS ×2 →
`BASE_LR = 0.0003 / (2 × √2) ≈ 1e-4`, and `MAML_INNER_LR` scales by
the same factor (3e-5 → 1e-5).

MAML changes shape at this scale: the persistent `fast_model` clone
(a second full weight copy) is gone — `maml_inner_update` now saves
the trainable params to CPU, runs the inner SGD steps on the model
itself and returns a `restore()` closure the caller invokes right
after computing `maml_pred`. Side effect: the outer loss through
`maml_pred` now lands on the real optimizer params (the old clone's
grads were never stepped), evaluated at the adapted point —
first-order MAML / Reptile semantics.

VRAM budget with the diet (bf16 autocast + checkpointing + 8-bit
AdamW + CPU EMA): ~4 GB weights + ~2 GB moments + ~1.5 GB activations
+ ~0.3 GB readouts ≈ 8–10 GB including fp32 grads — fits 12 GB.
Two scale-up-specific landmines, both fixed:

- `bitsandbytes` is a HARD dependency on CUDA boxes — `_make_adamw`
  raises instead of silently downgrading to fp32 AdamW, whose ~6.4 GB
  of moments OOM at the first `optimizer.step()`.
- `verify.check_overfit` no longer deep-copies the model onto the GPU
  (~13 GB extra at this scale): it snapshots the trainable params to
  CPU, probes the live model in place with an 8-bit Adam and restores
  afterwards. Worst-case verifier epoch ≈ weights + training moments
  + probe moments + one grad block ≈ 10.4 GB.

If the allocator still complains about fragmentation, export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (run.sh sets it).
Verify with `python test_scale_1b.py` (unit checks) and
`python test_scale_1b.py --smoke` (1 real epoch + peak-VRAM report;
overwrites the live state files like main.py).

