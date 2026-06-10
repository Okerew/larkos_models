/*
 * fusion_mechanism.c
 *
 * Projects neurons, memory entries and the LLM embedding into a
 * shared FUSION_DIM space and writes a balanced fused vector back
 * to the caller.
 *
 * The C side is deliberately a deterministic, information-preserving
 * feature extractor not a learner. The ctypes boundary kills
 * gradients, so any learnable mixing has to live in the Python
 * fusion_transformer which has autograd. Our job here is to hand
 * that transformer a rich, stable, non-degenerate 64-dim vector in
 * which all three input streams (llm query, neurons, memory) are
 * present in balanced proportion.
 *
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NEURONS 8
#define MAX_CONNECTIONS 6
#define INPUT_SIZE 6 // Size of the input tensor
#define MEMORY_VECTOR_SIZE (2 * MAX_NEURONS + INPUT_SIZE)
#define FUSION_DIM 64
#define NEURON_FIELDS 4 // state, output, num_conn, layer_id
#define NEURON_STRIDE (NEURON_FIELDS + MAX_CONNECTIONS * 2)
#define MAX_NEURON_FLAT (MAX_NEURONS * NEURON_STRIDE)
#define MAX_MEM_ENTRIES 300 // short + medium + long combined

/* Top-K memory entries used in attention: softmax over all slots
 * with similar importances collapses to uniform weights and mv
 * becomes a noise average. We score first then only attend over the
 * K highest-scoring entries so mv carries signal. */
#define MEM_TOP_K 8

/* The fused output is carved into three contiguous, balanced bands
 * so no stream can drown the others the way the old sigmoid gating
 * let neuron/memory state bury the llm query. Each band is filled by
 * its own stream then the whole vector is layer-normed and lightly
 * cross-mixed so the transformer still sees interactions. */
#define BAND_Q 22 // llm query band
#define BAND_N 21 // neuron band
#define BAND_M 21 // memory band  (BAND_Q + BAND_N + BAND_M == FUSION_DIM)

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void layer_norm(float *v, int n) {
  float mean = 0.0f, var = 0.0f;
  for (int i = 0; i < n; ++i)
    mean += v[i];
  mean /= (float)n;
  for (int i = 0; i < n; ++i) {
    float d = v[i] - mean;
    var += d * d;
  }
  var /= (float)n;
  float inv = 1.0f / sqrtf(var + 1e-6f);
  for (int i = 0; i < n; ++i)
    v[i] = (v[i] - mean) * inv;
}

static void softmax(float *v, int n) {
  float mx = v[0];
  for (int i = 1; i < n; ++i)
    if (v[i] > mx)
      mx = v[i];
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    v[i] = expf(v[i] - mx);
    sum += v[i];
  }
  for (int i = 0; i < n; ++i)
    v[i] /= sum;
}

static float dot(const float *a, const float *b) {
  float s = 0.0f;
  for (int i = 0; i < FUSION_DIM; ++i)
    s += a[i] * b[i];
  return s;
}

/* Deterministic hash -> [-1, 1]. Seeded by (row, col) so the
 * projection matrix is fixed across runs without storing it —
 * splitmix64-style mixing gives well-distributed pseudo-random
 * signs/magnitudes so every source dim feeds every output dim. */
static float proj_coef(uint32_t row, uint32_t col) {
  uint64_t z = ((uint64_t)row << 32) ^ (uint64_t)(col * 2654435761u + 1u);
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z = z ^ (z >> 31);
  /* map low 24 bits to [-1, 1] */
  uint32_t bits = (uint32_t)(z & 0xffffffu);
  return ((float)bits / (float)0x800000u) - 1.0f;
}

/* Dense pseudo-random projection of an n-dim source into FUSION_DIM.
 * Unlike the old strided average this mixes every input dimension
 * into every output dimension, so no information is dropped and the
 * output is not blocky. 1/sqrt(n) scaling keeps the variance stable
 * regardless of source length (Johnson-Lindenstrauss style). */
static void project(const float *src, int n, float *dst) {
  memset(dst, 0, FUSION_DIM * sizeof(float));
  if (n <= 0)
    return;
  float scale = 1.0f / sqrtf((float)n);
  for (int i = 0; i < FUSION_DIM; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j)
      acc += src[j] * proj_coef((uint32_t)i, (uint32_t)j);
    dst[i] = acc * scale;
  }
}

/* Project an n-dim source into a contiguous band [off, off+len) of a
 * FUSION_DIM destination, leaving the rest untouched. Uses the same
 * deterministic coefficients (row offset by band so bands don't
 * share a projection) so each stream owns a distinct sub-space. */
static void project_band(const float *src, int n, float *dst, int off, int len,
                         uint32_t band_seed) {
  if (n <= 0) {
    for (int i = 0; i < len; ++i)
      dst[off + i] = 0.0f;
    return;
  }
  float scale = 1.0f / sqrtf((float)n);
  for (int i = 0; i < len; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j)
      acc += src[j] * proj_coef((uint32_t)i + band_seed, (uint32_t)j);
    dst[off + i] = acc * scale;
  }
}

typedef struct {
  float state;
  float output;
  uint32_t num_connections;
  uint32_t layer_id;
  uint32_t connections[MAX_CONNECTIONS];
  float weights[MAX_CONNECTIONS];
} Neuron;

typedef struct {
  float vector[MEMORY_VECTOR_SIZE];
  float importance;
  uint32_t timestamp;
} MemoryEntry;

/*
 * cognitive_fuse
 *
 * llm_embed        : float[llm_dim] — the model-derived query
 * llm_dim          : length of llm_embed
 * neurons          : Neuron[n_neurons]
 * n_neurons        : how many neurons are active this step
 * mem_entries      : MemoryEntry[n_mem] — flattened short+med+long
 * n_mem            : total number of memory entries passed in
 * default_weights  : float[MEMORY_VECTOR_SIZE] — current system state
 *                    used as a prior when blending with memory vectors
 * mem_weight_ratio : float in [0,1] — blend between default weights
 *                    and stored memory vectors. 0 = pure default,
 *                    1 = pure memory.
 * context_factor   : scalar in [0,1] from derive_alpha_from_context,
 *                    modulates the cross-band mixing strength so the
 *                    fusion is context-dependent as asked
 * out              : float[FUSION_DIM] — caller-allocated output
 */
void cognitive_fuse(const float *llm_embed, int llm_dim, const Neuron *neurons,
                    int n_neurons, const MemoryEntry *mem_entries, int n_mem,
                    const float *default_weights, float mem_weight_ratio,
                    float context_factor, const float *text_embed, int text_dim,
                    float *out) {

  // 1. llm query, full-rank projection then layer-norm
  float q[FUSION_DIM];
  project(llm_embed, llm_dim, q);
  layer_norm(q, FUSION_DIM);

  // 1b. text stream projected into its own FUSION_DIM vector
  /* The input sentence reaches fused HERE, in the query band, not via
   * the memory prior. We add it into q before the query band is laid
   * down so the band carries both the model's own query and the
   * sentence content. text_dim 0 (no text) leaves q untouched. The
   * mix is light so the model query isn't swamped, but it is ungated
   * so mem_weight_ratio can never delete the sentence the way it would
   * if the text rode in through default_weights. */
  if (text_dim > 0 && text_embed != NULL) {
    float tv[FUSION_DIM];
    project(text_embed, text_dim, tv);
    layer_norm(tv, FUSION_DIM);
    for (int i = 0; i < FUSION_DIM; ++i)
      q[i] = q[i] + 0.5f * tv[i];
    layer_norm(q, FUSION_DIM);
  }

  // 2. neuron flat feature vector
  /* [state, output, num_conn, layer_id, conn_0..5, w_0..5] per neuron;
   * graph topology and edge weights give a richer structural picture
   * than scalars alone */
  float neuron_flat[MAX_NEURON_FLAT];
  int nf = 0;
  int neuron_count = n_neurons > 0 ? n_neurons : 0;
  for (int i = 0; i < neuron_count && i < MAX_NEURONS; ++i) {
    neuron_flat[nf++] = neurons[i].state;
    neuron_flat[nf++] = neurons[i].output;
    neuron_flat[nf++] = (float)neurons[i].num_connections;
    neuron_flat[nf++] = (float)neurons[i].layer_id;
    for (int c = 0; c < MAX_CONNECTIONS; ++c)
      neuron_flat[nf++] = (float)neurons[i].connections[c];
    for (int c = 0; c < MAX_CONNECTIONS; ++c)
      neuron_flat[nf++] = neurons[i].weights[c];
  }
  float nv[FUSION_DIM];
  project(neuron_flat, nf, nv);
  layer_norm(nv, FUSION_DIM);

  // 3. top-K memory attention, llm query attends over memory
  /* Attending over all slots with similar importances collapses
   * softmax to uniform weights so mv becomes a noise average. We
   * score all entries, keep the top-K by similarity to q, then
   * softmax over those K so the distribution is actually peaked. */
  int capped =
      n_mem > 0 ? (n_mem < MAX_MEM_ENTRIES ? n_mem : MAX_MEM_ENTRIES) : 0;
  float scale = 1.0f / sqrtf((float)FUSION_DIM);

  float blend_ratio = mem_weight_ratio;
  if (blend_ratio < 0.0f)
    blend_ratio = 0.0f;
  if (blend_ratio > 1.0f)
    blend_ratio = 1.0f;

  float mv[FUSION_DIM];
  float keys[MAX_MEM_ENTRIES][FUSION_DIM];
  float raw_scores[MAX_MEM_ENTRIES];

  for (int m = 0; m < capped; ++m) {
    /* key = importance-weighted blend of default prior and stored
     * memory vector. blend_ratio 0 -> pure default, 1 -> pure memory */
    float weighted[MEMORY_VECTOR_SIZE];
    for (int j = 0; j < MEMORY_VECTOR_SIZE; ++j) {
      float blended = (1.0f - blend_ratio) * default_weights[j] +
                      blend_ratio * mem_entries[m].vector[j];
      weighted[j] = blended * mem_entries[m].importance;
    }
    project(weighted, MEMORY_VECTOR_SIZE, keys[m]);
    layer_norm(keys[m], FUSION_DIM);
    raw_scores[m] = dot(q, keys[m]) * scale;
  }

  if (capped > 0) {
    int k = capped < MEM_TOP_K ? capped : MEM_TOP_K;
    int top_idx[MEM_TOP_K];
    float top_scr[MEM_TOP_K];
    for (int t = 0; t < k; ++t) {
      top_idx[t] = -1;
      top_scr[t] = -1e30f;
    }
    for (int m = 0; m < capped; ++m) {
      if (raw_scores[m] > top_scr[k - 1]) {
        top_idx[k - 1] = m;
        top_scr[k - 1] = raw_scores[m];
        for (int t = k - 1; t > 0 && top_scr[t] > top_scr[t - 1]; --t) {
          float fs = top_scr[t];
          top_scr[t] = top_scr[t - 1];
          top_scr[t - 1] = fs;
          int fi = top_idx[t];
          top_idx[t] = top_idx[t - 1];
          top_idx[t - 1] = fi;
        }
      }
    }
    softmax(top_scr, k);
    memset(mv, 0, FUSION_DIM * sizeof(float));
    for (int t = 0; t < k; ++t)
      if (top_idx[t] >= 0)
        for (int i = 0; i < FUSION_DIM; ++i)
          mv[i] += top_scr[t] * keys[top_idx[t]][i];
    layer_norm(mv, FUSION_DIM);
  } else {
    memset(mv, 0, FUSION_DIM * sizeof(float));
  }

  // 4. banded assembly
  /* Each stream is re-projected into its OWN contiguous band so the
   * three never compete for the same coordinates. This is the core
   * fix for the old behaviour where sigmoid gating let neuron/memory
   * state bury the llm query at a fixed ~0.5 coefficient. Now the
   * query owns BAND_Q dims outright; the transformer downstream can
   * weight bands however it learns to. Band seeds keep each band's
   * projection distinct. */
  float fused[FUSION_DIM];
  project_band(q, FUSION_DIM, fused, 0, BAND_Q, 1009u);
  project_band(nv, FUSION_DIM, fused, BAND_Q, BAND_N, 2003u);
  project_band(mv, FUSION_DIM, fused, BAND_Q + BAND_N, BAND_M, 3001u);

  // 5. context-modulated cross-band mixing
  /* The bands are independent after step 4, so we add a single light
   * cross-band interaction term controlled by context_factor. This
   * gives the transformer some pre-mixed structure to latch onto and
   * makes the fusion genuinely context-dependent, without letting any
   * stream dominate. mix_strength stays small so bands remain mostly
   * separable. */
  float mix_strength = sigmoid(context_factor * 2.0f - 1.0f) * 0.5f;
  float cross[FUSION_DIM];
  for (int i = 0; i < FUSION_DIM; ++i) {
    int partner = (i + FUSION_DIM / 2) % FUSION_DIM;
    cross[i] = fused[i] + mix_strength * fused[partner];
  }
  memcpy(fused, cross, FUSION_DIM * sizeof(float));

  layer_norm(fused, FUSION_DIM);
  memcpy(out, fused, FUSION_DIM * sizeof(float));
}

int fusion_dim(void) { return FUSION_DIM; }
