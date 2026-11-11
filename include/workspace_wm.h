#ifndef WORKSPACE_WM_H
#define WORKSPACE_WM_H

/* Workspace pool (WM runtime) - standalone module implemented in
 * workspace_wm.c. nw.h only mirrors neural_web.c/immitrin_functions.c,
 * so the WM types and API live here; shared structs come via nw.h. */
#include "nw.h"

#define WM_POOL_CAPACITY 8
#define WM_TASK_CAP 2
#define WM_MAX_SLOTS 64
#define WM_MAX_LINKS 128
#define WM_LABEL_SIZE 64
#define WM_TEXT_SIZE 256
#define WM_HISTORY_SIZE 256
#define WM_SEED_K 8

#define WM_KIND_SESSION 0
#define WM_KIND_GLOBAL 1
#define WM_KIND_TASK 2

#define WM_CONFAB_THRESHOLD 0.65f
#define WM_COHERENCE_THRESHOLD 0.70f
#define WM_CONSISTENCY_THRESHOLD 0.75f
#define WM_COMMIT_IMPORTANCE_FLOOR 0.5f
#define WM_DEFAULT_ALPHA 0.5f

#define WM_POOL_MAGIC 0x574d504cu /* 'WMPL' */
#define WM_POOL_VERSION 1u

#define WM_OK 0
#define WM_ERR_ARGS -1
#define WM_ERR_NO_SPACE -2
#define WM_ERR_NOT_FOUND -3
#define WM_ERR_LOW_IMP -5

typedef struct {
  char label[WM_LABEL_SIZE];
  char text[WM_TEXT_SIZE];
  float vector[MEMORY_VECTOR_SIZE];
  float importance;
  unsigned int step_tag;
  bool used;
} WMSlot;

typedef struct {
  unsigned int a;
  unsigned int b;
  char type[WM_LABEL_SIZE];
  float weight;
  bool used;
} WMLink;

typedef struct {
  unsigned int step;
  char label[WM_LABEL_SIZE];
  float vector[MEMORY_VECTOR_SIZE];
  float importance;
} WMSlotSnapshot;

typedef struct {
  unsigned int workspace_id;
  unsigned int kind;
  bool active;
  unsigned int epoch;
  unsigned int created_step;
  unsigned int last_used_step;
  float context_vector[CONTEXT_VECTOR_SIZE];
  WMSlot slots[WM_MAX_SLOTS];
  unsigned int num_slots;
  WMLink links[WM_MAX_LINKS];
  unsigned int num_links;
  WMSlotSnapshot history[WM_HISTORY_SIZE];
  unsigned int history_count;
  unsigned int history_head;
} Workspace;

typedef struct WorkspacePool {
  Workspace workspaces[WM_POOL_CAPACITY];
  unsigned int next_id;
  unsigned int epoch;
} WorkspacePool;
WorkspacePool *createWorkspacePool(void);
void freeWorkspacePool(WorkspacePool *pool);
int createWorkspace(WorkspacePool *pool, unsigned int kind,
                    const float *query, MemorySystem *mem,
                    unsigned int step);
int destroyWorkspace(WorkspacePool *pool, unsigned int id);
int writeSlot(WorkspacePool *pool, unsigned int id, const char *label,
              const char *text, const float *vector, float importance,
              unsigned int step);
int transformSlot(WorkspacePool *pool, unsigned int id,
                  const char *label, const float *target, float alpha,
                  unsigned int step);
int linkSlots(WorkspacePool *pool, unsigned int id, const char *a,
              const char *b, const char *type, float weight);
int wmRetrieve(WorkspacePool *pool, unsigned int id, const float *query,
               int k, WorkingMemorySystem *wm, unsigned int step,
               int *out_idx, float *out_score, int max_out);
int wmCommitSlot(WorkspacePool *pool, unsigned int id,
                 const char *label, MemorySystem *mem,
                 unsigned int step, float blend);
int wmValidateWorkspace(WorkspacePool *pool, unsigned int id,
                        MemorySystem *mem, int *out_idx, int *out_flag,
                        float *out_coh, float *out_cons, int max_out);
int wmDiscardSlot(WorkspacePool *pool, unsigned int id,
                  const char *label, float *out_importance,
                  unsigned int step);
int replayWorkspace(WorkspacePool *pool, unsigned int id,
                    unsigned int from_step, WMSlotSnapshot *out,
                    int max_out);
bool validateWorkspacePool(WorkspacePool *pool);
void saveWorkspacePool(WorkspacePool *pool, const char *filename);
WorkspacePool *loadWorkspacePool(const char *filename);

#endif
