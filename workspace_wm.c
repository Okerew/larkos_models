/* Workspace working memory (WM runtime) - split out of neural_web.c.
 * Slot/workspace types, WM_* constants and the public prototypes live
 * in include/workspace_wm.h, shared neural_web structs come via nw.h. */
#include "include/workspace_wm.h"

/* ===================== Workspace pool (WM runtime) =====================
 * Server-side allocated working memory: short-lived manipulable state
 * for tasks that need exact intermediate state across turns (board
 * games, multi-step refactors, multi-hop research). Sequential
 * single-focus design - one workspace owns WorkingMemorySystem.focus at
 * a time via wmSyncFocus, zero parallel buffers.
 * Slots are plain value types (no inner pointers) so the whole pool is
 * one fwrite/fread block saved alongside the memory snapshot.
 */

static Workspace *wmFindWorkspace(WorkspacePool *pool, unsigned int id) {
  if (!pool)
    return NULL;
  for (int i = 0; i < WM_POOL_CAPACITY; i++) {
    Workspace *ws = &pool->workspaces[i];
    if (ws->active && ws->workspace_id == id)
      return ws;
  }
  return NULL;
}

static Workspace *wmFindByKind(WorkspacePool *pool, unsigned int kind) {
  if (!pool)
    return NULL;
  for (int i = 0; i < WM_POOL_CAPACITY; i++) {
    Workspace *ws = &pool->workspaces[i];
    if (ws->active && ws->kind == kind)
      return ws;
  }
  return NULL;
}

static int wmFindSlot(Workspace *ws, const char *label) {
  if (!ws || !label)
    return -1;
  for (int i = 0; i < WM_MAX_SLOTS; i++) {
    if (ws->slots[i].used && strncmp(ws->slots[i].label, label,
                                     WM_LABEL_SIZE) == 0)
      return i;
  }
  return -1;
}

static void wmSanitizeVector(float *v, unsigned int n) {
  for (unsigned int i = 0; i < n; i++) {
    if (!isfinite(v[i]))
      v[i] = 0.0f;
  }
}

static void wmCountSlots(Workspace *ws) {
  unsigned int n = 0;
  for (int i = 0; i < WM_MAX_SLOTS; i++)
    if (ws->slots[i].used)
      n++;
  ws->num_slots = n;
}

static void wmCountLinks(Workspace *ws) {
  unsigned int n = 0;
  for (int i = 0; i < WM_MAX_LINKS; i++)
    if (ws->links[i].used)
      n++;
  ws->num_links = n;
}

static void wmRecordHistory(Workspace *ws, const WMSlot *slot) {
  WMSlotSnapshot *snap = &ws->history[ws->history_head];
  snap->step = slot->step_tag;
  strncpy(snap->label, slot->label, WM_LABEL_SIZE - 1);
  snap->label[WM_LABEL_SIZE - 1] = '\0';
  memcpy(snap->vector, slot->vector, MEMORY_VECTOR_SIZE * sizeof(float));
  snap->importance = slot->importance;
  ws->history_head = (ws->history_head + 1) % WM_HISTORY_SIZE;
  ws->history_count++;
}

static void wmDropSlot(Workspace *ws, int idx) {
  if (idx < 0 || idx >= WM_MAX_SLOTS)
    return;
  ws->slots[idx].used = false;
  for (int i = 0; i < WM_MAX_LINKS; i++) {
    WMLink *l = &ws->links[i];
    if (l->used && (l->a == (unsigned int)idx || l->b == (unsigned int)idx))
      l->used = false;
  }
  wmCountSlots(ws);
  wmCountLinks(ws);
}

/* Seed context_vector by top-k importance-weighted retrieval from the
 * persistent tiers. With a query the score is cosine x (0.5 + imp),
 * without one it degrades to plain importance. The query itself always
 * rides along so an empty memory still leaves a usable context.
 */
static void wmSeedContext(Workspace *ws, const float *query,
                          MemorySystem *mem) {
  memset(ws->context_vector, 0, CONTEXT_VECTOR_SIZE * sizeof(float));
  if (query) {
    for (int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
      float v = query[j];
      ws->context_vector[j] = isfinite(v) ? v : 0.0f;
    }
  }
  if (mem) {
    const float *top_vec[WM_SEED_K];
    float top_score[WM_SEED_K];
    int ntop = 0;
    MemoryCluster *tiers[3] = {&mem->hierarchy.short_term,
                               &mem->hierarchy.medium_term,
                               &mem->hierarchy.long_term};
    for (int t = 0; t < 3; t++) {
      MemoryCluster *tier = tiers[t];
      unsigned int n =
          tier->size < tier->capacity ? tier->size : tier->capacity;
      for (unsigned int i = 0; i < n; i++) {
        MemoryEntry *e = &tier->entries[i];
        float score;
        if (query) {
          score = computeCosineSimilarity(e->vector, query,
                                          MEMORY_VECTOR_SIZE) *
                  (0.5f + e->importance);
        } else {
          score = e->importance;
        }
        if (!(score > 0.0f))
          continue;
        if (ntop < WM_SEED_K) {
          top_vec[ntop] = e->vector;
          top_score[ntop] = score;
          ntop++;
        } else {
          int worst = 0;
          for (int k = 1; k < WM_SEED_K; k++)
            if (top_score[k] < top_score[worst])
              worst = k;
          if (score > top_score[worst]) {
            top_vec[worst] = e->vector;
            top_score[worst] = score;
          }
        }
      }
    }
    float total = 0.0f;
    for (int k = 0; k < ntop; k++)
      total += top_score[k];
    if (total > 1e-6f) {
      for (int k = 0; k < ntop; k++)
        addWeightedVector(ws->context_vector, top_vec[k],
                          top_score[k] / total, MEMORY_VECTOR_SIZE);
    }
  }
  normalizeVector(ws->context_vector, CONTEXT_VECTOR_SIZE);
}

/* The workspace focus set IS work_mem->focus - one source of truth.
 * Selected slots are materialised as focus entries (slot vector in the
 * feature head, workspace context as context_vector) and updateContext
 * folds them into global_context, the existing maintenance path.
 */
static void wmSyncFocus(Workspace *ws, WorkingMemorySystem *wm,
                        const int *sel, int n) {
  if (!wm)
    return;
  for (unsigned int i = 0; i < wm->focus.size; i++) {
    free(wm->focus.entries[i].features);
    free(wm->focus.entries[i].context_vector);
    wm->focus.entries[i].features = NULL;
    wm->focus.entries[i].context_vector = NULL;
  }
  wm->focus.size = 0;
  if (n > (int)wm->focus.capacity)
    n = (int)wm->focus.capacity;
  for (int s = 0; s < n; s++) {
    WMSlot *slot = &ws->slots[sel[s]];
    WorkingMemoryEntry *e = &wm->focus.entries[wm->focus.size];
    e->features = (float *)calloc(FEATURE_VECTOR_SIZE, sizeof(float));
    e->context_vector =
        (float *)calloc(CONTEXT_VECTOR_SIZE, sizeof(float));
    if (!e->features || !e->context_vector) {
      free(e->features);
      free(e->context_vector);
      e->features = NULL;
      e->context_vector = NULL;
      continue;
    }
    // MEMORY_VECTOR_SIZE < FEATURE_VECTOR_SIZE, tail stays zeroed
    memcpy(e->features, slot->vector,
           MEMORY_VECTOR_SIZE * sizeof(float));
    memcpy(e->context_vector, ws->context_vector,
           CONTEXT_VECTOR_SIZE * sizeof(float));
    e->abstraction_level = slot->importance;
    e->depth = 0;
    wm->focus.size++;
  }
  updateContext(wm);
}

/* Tier routing for a committed slot, mirrors addMemory's storage half
 * (adaptive thresholds + least-important replacement + flat ring) so
 * commits land wherever the live distribution says they belong.
 */
static void wmStoreEntry(MemorySystem *mem, const MemoryEntry *entry) {
  float mt_threshold, lt_threshold;
  promotionThresholds(mem, &mt_threshold, &lt_threshold);
  MemoryEntry e = *entry;

  if (e.importance >= lt_threshold) {
    if (mem->hierarchy.long_term.size <
        mem->hierarchy.long_term.capacity) {
      mem->hierarchy.long_term
          .entries[mem->hierarchy.long_term.size++] = e;
    } else {
      unsigned int rc = 0;
      int *li = findLeastImportantMemory(
          mem->hierarchy.long_term.entries,
          mem->hierarchy.long_term.size, 5, &rc);
      if (li && rc > 0)
        mem->hierarchy.long_term.entries[li[0]] = e;
      free(li);
    }
  } else if (e.importance >= mt_threshold) {
    if (mem->hierarchy.medium_term.size <
        mem->hierarchy.medium_term.capacity) {
      mem->hierarchy.medium_term
          .entries[mem->hierarchy.medium_term.size++] = e;
    } else {
      unsigned int rc = 0;
      int *li = findLeastImportantMemory(
          mem->hierarchy.medium_term.entries,
          mem->hierarchy.medium_term.size, 3, &rc);
      if (li && rc > 0 &&
          e.importance >
              mem->hierarchy.medium_term.entries[li[0]].importance)
        mem->hierarchy.medium_term.entries[li[0]] = e;
      free(li);
    }
  } else {
    if (mem->hierarchy.short_term.size <
        mem->hierarchy.short_term.capacity) {
      mem->hierarchy.short_term
          .entries[mem->hierarchy.short_term.size++] = e;
    } else {
      unsigned int rc = 0;
      int *li = findLeastImportantMemory(
          mem->hierarchy.short_term.entries,
          mem->hierarchy.short_term.size, 5, &rc);
      if (li && rc > 0)
        mem->hierarchy.short_term.entries[li[0]] = e;
      free(li);
    }
  }

  if (mem->capacity == 0 || mem->entries == NULL)
    return;
  if (mem->head >= mem->capacity)
    mem->head = 0;
  mem->entries[mem->head] = e;
  mem->head = (mem->head + 1) % mem->capacity;
  if (mem->size < mem->capacity)
    mem->size++;
}

WorkspacePool *createWorkspacePool(void) {
  WorkspacePool *pool =
      (WorkspacePool *)calloc(1, sizeof(WorkspacePool));
  if (!pool) {
    fprintf(stderr, "Failed to allocate workspace pool\n");
    return NULL;
  }
  pool->next_id = 1;
  pool->epoch = 0;
  return pool;
}

void freeWorkspacePool(WorkspacePool *pool) { free(pool); }

static int wmAllocRaw(WorkspacePool *pool, unsigned int kind,
                      const float *query, MemorySystem *mem,
                      unsigned int step) {
  int idx = -1;
  for (int i = 0; i < WM_POOL_CAPACITY; i++) {
    if (!pool->workspaces[i].active) {
      idx = i;
      break;
    }
  }
  if (idx < 0)
    return WM_ERR_NO_SPACE;
  Workspace *ws = &pool->workspaces[idx];
  memset(ws, 0, sizeof(Workspace));
  ws->workspace_id = pool->next_id++;
  ws->kind = kind;
  ws->active = true;
  ws->epoch = pool->epoch++;
  ws->created_step = step;
  ws->last_used_step = step;
  wmSeedContext(ws, query, mem);
  return (int)ws->workspace_id;
}

/* Session and global are singletons and auto-created on any allocate;
 * task workspaces cap at WM_TASK_CAP - further allocate is refused so
 * the caller commits or discards instead of juggling parallel buffers.
 */
int createWorkspace(WorkspacePool *pool, unsigned int kind,
                    const float *query, MemorySystem *mem,
                    unsigned int step) {
  if (!pool || kind > WM_KIND_TASK)
    return WM_ERR_ARGS;
  if (kind != WM_KIND_SESSION && !wmFindByKind(pool, WM_KIND_SESSION))
    wmAllocRaw(pool, WM_KIND_SESSION, NULL, mem, step);
  if (kind != WM_KIND_GLOBAL && !wmFindByKind(pool, WM_KIND_GLOBAL))
    wmAllocRaw(pool, WM_KIND_GLOBAL, NULL, mem, step);

  if (kind == WM_KIND_SESSION || kind == WM_KIND_GLOBAL) {
    Workspace *ex = wmFindByKind(pool, kind);
    if (ex) {
      ex->last_used_step = step;
      return (int)ex->workspace_id;
    }
  }
  if (kind == WM_KIND_TASK) {
    int tasks = 0;
    for (int i = 0; i < WM_POOL_CAPACITY; i++)
      if (pool->workspaces[i].active &&
          pool->workspaces[i].kind == WM_KIND_TASK)
        tasks++;
    if (tasks >= WM_TASK_CAP)
      return WM_ERR_NO_SPACE;
  }
  return wmAllocRaw(pool, kind, query, mem, step);
}

int destroyWorkspace(WorkspacePool *pool, unsigned int id) {
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  memset(ws, 0, sizeof(Workspace));
  pool->epoch++;
  return WM_OK;
}

int writeSlot(WorkspacePool *pool, unsigned int id, const char *label,
              const char *text, const float *vector, float importance,
              unsigned int step) {
  if (!pool || !label || !label[0] || !vector)
    return WM_ERR_ARGS;
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  int idx = wmFindSlot(ws, label);
  if (idx < 0) {
    for (int i = 0; i < WM_MAX_SLOTS; i++) {
      if (!ws->slots[i].used) {
        idx = i;
        break;
      }
    }
  }
  if (idx < 0) {
    // fixed capacity: lowest importance gives way, no learned policy
    int worst = 0;
    for (int i = 1; i < WM_MAX_SLOTS; i++)
      if (ws->slots[i].importance < ws->slots[worst].importance)
        worst = i;
    idx = worst;
  }
  WMSlot *s = &ws->slots[idx];
  strncpy(s->label, label, WM_LABEL_SIZE - 1);
  s->label[WM_LABEL_SIZE - 1] = '\0';
  if (text) {
    strncpy(s->text, text, WM_TEXT_SIZE - 1);
    s->text[WM_TEXT_SIZE - 1] = '\0';
  } else {
    s->text[0] = '\0';
  }
  memcpy(s->vector, vector, MEMORY_VECTOR_SIZE * sizeof(float));
  wmSanitizeVector(s->vector, MEMORY_VECTOR_SIZE);
  s->importance = isfinite(importance)
                      ? fmaxf(0.0f, fminf(1.0f, importance))
                      : 0.5f;
  s->step_tag = step;
  s->used = true;
  wmCountSlots(ws);
  wmRecordHistory(ws, s);
  ws->epoch++;
  pool->epoch++;
  ws->last_used_step = step;
  return idx;
}

int transformSlot(WorkspacePool *pool, unsigned int id,
                  const char *label, const float *target, float alpha,
                  unsigned int step) {
  if (!pool || !label || !target)
    return WM_ERR_ARGS;
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  int idx = wmFindSlot(ws, label);
  if (idx < 0)
    return WM_ERR_NOT_FOUND;
  if (!isfinite(alpha) || alpha < 0.0f || alpha > 1.0f)
    alpha = WM_DEFAULT_ALPHA;
  WMSlot *s = &ws->slots[idx];
  for (int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
    float tv = isfinite(target[j]) ? target[j] : 0.0f;
    s->vector[j] = (1.0f - alpha) * s->vector[j] + alpha * tv;
  }
  wmSanitizeVector(s->vector, MEMORY_VECTOR_SIZE);
  s->step_tag = step;
  wmRecordHistory(ws, s);
  ws->epoch++;
  pool->epoch++;
  ws->last_used_step = step;
  return WM_OK;
}

int linkSlots(WorkspacePool *pool, unsigned int id, const char *a,
              const char *b, const char *type, float weight) {
  if (!pool || !a || !b)
    return WM_ERR_ARGS;
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  int ia = wmFindSlot(ws, a);
  int ib = wmFindSlot(ws, b);
  if (ia < 0 || ib < 0)
    return WM_ERR_NOT_FOUND;
  if (!isfinite(weight))
    weight = 0.5f;
  for (int i = 0; i < WM_MAX_LINKS; i++) {
    WMLink *l = &ws->links[i];
    if (l->used &&
        ((l->a == (unsigned int)ia && l->b == (unsigned int)ib) ||
         (l->a == (unsigned int)ib && l->b == (unsigned int)ia))) {
      if (type) {
        strncpy(l->type, type, WM_LABEL_SIZE - 1);
        l->type[WM_LABEL_SIZE - 1] = '\0';
      }
      l->weight = weight;
      return WM_OK;
    }
  }
  int idx = -1;
  for (int i = 0; i < WM_MAX_LINKS; i++) {
    if (!ws->links[i].used) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    int worst = 0;
    for (int i = 1; i < WM_MAX_LINKS; i++)
      if (ws->links[i].weight < ws->links[worst].weight)
        worst = i;
    idx = worst;
  }
  WMLink *l = &ws->links[idx];
  l->a = (unsigned int)ia;
  l->b = (unsigned int)ib;
  if (type) {
    strncpy(l->type, type, WM_LABEL_SIZE - 1);
    l->type[WM_LABEL_SIZE - 1] = '\0';
  } else {
    l->type[0] = '\0';
  }
  l->weight = weight;
  l->used = true;
  wmCountLinks(ws);
  return WM_OK;
}

/* Workspace-local refinement: importance-weighted cosine over the
 * workspace's own slots, top-k promoted into work_mem focus. Not a
 * world search - persistent recall stays larkos_recall's job.
 */
int wmRetrieve(WorkspacePool *pool, unsigned int id, const float *query,
               int k, WorkingMemorySystem *wm, unsigned int step,
               int *out_idx, float *out_score, int max_out) {
  if (!pool || !query)
    return WM_ERR_ARGS;
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  if (k < 1)
    k = 1;
  if (k > WM_MAX_SLOTS)
    k = WM_MAX_SLOTS;
  if (max_out < k)
    k = max_out < 0 ? 0 : max_out;

  int sel[WM_MAX_SLOTS];
  float score[WM_MAX_SLOTS];
  int n = 0;
  for (int i = 0; i < WM_MAX_SLOTS; i++) {
    if (!ws->slots[i].used)
      continue;
    float s = computeCosineSimilarity(ws->slots[i].vector, query,
                                      MEMORY_VECTOR_SIZE) *
              (0.5f + ws->slots[i].importance);
    int pos = n < k ? n : -1;
    if (pos < 0) {
      int worst = 0;
      for (int j = 1; j < n; j++)
        if (score[j] < score[worst])
          worst = j;
      if (s > score[worst])
        pos = worst;
    }
    if (pos >= 0) {
      sel[pos] = i;
      score[pos] = s;
      if (n < k)
        n++;
    }
  }
  // sort the tiny selection so hits come back best-first
  for (int i = 1; i < n; i++) {
    for (int j = i; j > 0 && score[j] > score[j - 1]; j--) {
      float ts = score[j];
      score[j] = score[j - 1];
      score[j - 1] = ts;
      int ti = sel[j];
      sel[j] = sel[j - 1];
      sel[j - 1] = ti;
    }
  }
  if (out_idx && out_score) {
    for (int i = 0; i < n; i++) {
      out_idx[i] = sel[i];
      out_score[i] = score[i];
    }
  }
  wmSyncFocus(ws, wm, sel, n);
  ws->last_used_step = step;
  ws->epoch++;
  pool->epoch++;
  return n;
}

int wmCommitSlot(WorkspacePool *pool, unsigned int id,
                 const char *label, MemorySystem *mem,
                 unsigned int step, float blend) {
  if (!pool || !mem || !label)
    return WM_ERR_ARGS;
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  int idx = wmFindSlot(ws, label);
  if (idx < 0)
    return WM_ERR_NOT_FOUND;
  WMSlot *s = &ws->slots[idx];
  if (s->importance < WM_COMMIT_IMPORTANCE_FLOOR)
    return WM_ERR_LOW_IMP;
  if (!isfinite(blend) || blend < 0.0f || blend > 1.0f)
    blend = 0.3f;

  MemoryEntry e;
  memset(&e, 0, sizeof(MemoryEntry));
  e.timestamp = step;
  float sum = 0.0f;
  for (int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
    float cv = (j < CONTEXT_VECTOR_SIZE) ? ws->context_vector[j] : 0.0f;
    if (!isfinite(cv))
      cv = 0.0f;
    e.vector[j] = (1.0f - blend) * s->vector[j] + blend * cv;
    if (!isfinite(e.vector[j]))
      e.vector[j] = 0.0f;
    sum += e.vector[j] * e.vector[j];
  }
  float norm = sqrtf(sum);
  if (norm > 1e-6f) {
    for (int j = 0; j < MEMORY_VECTOR_SIZE; j++)
      e.vector[j] /= norm;
  }
  e.importance = s->importance;
  wmStoreEntry(mem, &e);
  ws->last_used_step = step;
  ws->epoch++;
  pool->epoch++;
  return WM_OK;
}

/* Per-slot self-reflection: coherence against the workspace context,
 * consistency against the consolidated long-term tier. confab =
 * 1 - max(coh, cons) past 0.65 dereferences the slot outright, the
 * softer bands only flag. out_* arrays are indexed by the PRE-validation
 * slot positions, parallel to the workspace struct the caller holds.
 */
int wmValidateWorkspace(WorkspacePool *pool, unsigned int id,
                        MemorySystem *mem, int *out_idx, int *out_flag,
                        float *out_coh, float *out_cons, int max_out) {
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws || !out_idx || !out_flag || !out_coh || !out_cons)
    return WM_ERR_ARGS;
  int n = 0;
  for (int i = 0; i < WM_MAX_SLOTS && n < max_out; i++) {
    if (!ws->slots[i].used)
      continue;
    WMSlot *s = &ws->slots[i];
    float coh = computeCosineSimilarity(s->vector, ws->context_vector,
                                        MEMORY_VECTOR_SIZE);
    coh = fmaxf(0.0f, fminf(1.0f, coh));
    float cons = 0.0f;
    if (mem) {
      MemoryCluster *lt = &mem->hierarchy.long_term;
      unsigned int ln =
          lt->size < lt->capacity ? lt->size : lt->capacity;
      float best = -1.0f;
      for (unsigned int j = 0; j < ln; j++) {
        float c = computeCosineSimilarity(s->vector,
                                          lt->entries[j].vector,
                                          MEMORY_VECTOR_SIZE) *
                  (0.5f + lt->entries[j].importance) / 1.5f;
        if (c > best)
          best = c;
      }
      cons = fmaxf(0.0f, fminf(1.0f, best));
    }
    float confab = 1.0f - fmaxf(coh, cons);
    int flag = 0;
    bool drop = false;
    if (confab > WM_CONFAB_THRESHOLD) {
      /*
       * Text-backed slots carry journal provenance, and symbolic
       * content (FEN strings, code, exact state) legitimately embeds
       * far from the semantic context - embedding distance alone
       * must not delete them. Only ungrounded raw-vector slots, which
       * nothing can re-derive, are auto-dereferenced on confab.
       */
      if (s->text[0] == '\0') {
        flag = 3;
        drop = true;
      } else {
        flag = 4;
      }
    } else if (coh < WM_COHERENCE_THRESHOLD) {
      flag = 1;
    } else if (cons < WM_CONSISTENCY_THRESHOLD) {
      flag = 2;
    }
    out_idx[n] = i;
    out_flag[n] = flag;
    out_coh[n] = coh;
    out_cons[n] = cons;
    n++;
    if (drop)
      wmDropSlot(ws, i);
  }
  return n;
}

int wmDiscardSlot(WorkspacePool *pool, unsigned int id,
                  const char *label, float *out_importance,
                  unsigned int step) {
  if (!pool || !label)
    return WM_ERR_ARGS;
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  int idx = wmFindSlot(ws, label);
  if (idx < 0)
    return WM_ERR_NOT_FOUND;
  if (out_importance)
    *out_importance = ws->slots[idx].importance;
  wmDropSlot(ws, idx);
  ws->last_used_step = step;
  ws->epoch++;
  pool->epoch++;
  return WM_OK;
}

/* Never trust the current buffer: re-derive slot states from the
 * step-tagged history ring, oldest first.
 */
int replayWorkspace(WorkspacePool *pool, unsigned int id,
                    unsigned int from_step, WMSlotSnapshot *out,
                    int max_out) {
  Workspace *ws = wmFindWorkspace(pool, id);
  if (!ws)
    return WM_ERR_NOT_FOUND;
  if (max_out <= 0)
    return 0;
  unsigned int stored = ws->history_count < WM_HISTORY_SIZE
                            ? ws->history_count
                            : WM_HISTORY_SIZE;
  unsigned int start =
      (ws->history_head + WM_HISTORY_SIZE - stored) % WM_HISTORY_SIZE;
  int n = 0;
  for (unsigned int i = 0; i < stored && n < max_out; i++) {
    WMSlotSnapshot *snap =
        &ws->history[(start + i) % WM_HISTORY_SIZE];
    if (snap->step < from_step)
      continue;
    if (out)
      out[n] = *snap;
    n++;
  }
  return n;
}

bool validateWorkspacePool(WorkspacePool *pool) {
  if (!validateMemoryBlock(pool, sizeof(WorkspacePool),
                           "WorkspacePool")) {
    return false;
  }
  if (pool->next_id > 100000000u)
    pool->next_id = 1;
  for (int w = 0; w < WM_POOL_CAPACITY; w++) {
    Workspace *ws = &pool->workspaces[w];
    if (ws->num_slots > WM_MAX_SLOTS)
      ws->num_slots = WM_MAX_SLOTS;
    if (ws->num_links > WM_MAX_LINKS)
      ws->num_links = WM_MAX_LINKS;
    if (ws->history_head >= WM_HISTORY_SIZE)
      ws->history_head = 0;
    if (!ws->active)
      continue;
    if (ws->kind > WM_KIND_TASK) {
      fprintf(stderr,
              "WARNING: Workspace %u has invalid kind, dropped\n",
              ws->workspace_id);
      memset(ws, 0, sizeof(Workspace));
      continue;
    }
    for (unsigned int i = 0; i < CONTEXT_VECTOR_SIZE; i++)
      if (!isfinite(ws->context_vector[i]))
        ws->context_vector[i] = 0.0f;
    for (int i = 0; i < WM_MAX_SLOTS; i++) {
      WMSlot *s = &ws->slots[i];
      if (!isfinite(s->importance))
        s->importance = 0.0f;
      for (int j = 0; j < MEMORY_VECTOR_SIZE; j++)
        if (!isfinite(s->vector[j]))
          s->vector[j] = 0.0f;
    }
  }
  return true;
}

void saveWorkspacePool(WorkspacePool *pool, const char *filename) {
  if (!pool || !filename)
    return;
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    fprintf(stderr,
            "Error opening workspace pool file for writing\n");
    return;
  }
  uint32_t magic = WM_POOL_MAGIC;
  uint32_t version = WM_POOL_VERSION;
  fwrite(&magic, sizeof(magic), 1, fp);
  fwrite(&version, sizeof(version), 1, fp);
  fwrite(pool, sizeof(WorkspacePool), 1, fp);
  fclose(fp);
}

WorkspacePool *loadWorkspacePool(const char *filename) {
  if (!filename)
    return NULL;
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL)
    return NULL;
  uint32_t magic = 0;
  uint32_t version = 0;
  if (fread(&magic, sizeof(magic), 1, fp) != 1 ||
      magic != WM_POOL_MAGIC ||
      fread(&version, sizeof(version), 1, fp) != 1 ||
      version != WM_POOL_VERSION) {
    fprintf(stderr,
            "Workspace pool file bad magic/version, ignoring\n");
    fclose(fp);
    return NULL;
  }
  WorkspacePool *pool = createWorkspacePool();
  if (!pool) {
    fclose(fp);
    return NULL;
  }
  if (fread(pool, sizeof(WorkspacePool), 1, fp) != 1) {
    fprintf(stderr, "Workspace pool file truncated, ignoring\n");
    freeWorkspacePool(pool);
    fclose(fp);
    return NULL;
  }
  fclose(fp);
  if (!validateWorkspacePool(pool)) {
    freeWorkspacePool(pool);
    return NULL;
  }
  return pool;
}
