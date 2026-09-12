"""Text journal for memory entries.

Every C-side MemoryEntry carries the training step counter as its
timestamp, and at that step the training loop knows the driver text.
We persist {step: {text, embedding}} as JSON next to memory.bin, so
inference can answer "what was the system reading when this entry
formed?".

"""

import json
from pathlib import Path
from modules.config import JOURNAL_FILE, SUPPORT_SIM, RELATED_SIM

import numpy as np

_entries: dict[int, dict] = {}
_emb_cache: dict[str, list] = {}


def reset() -> None:
    _entries.clear()
    _emb_cache.clear()


def encode_cached(text: str, encode_fn) -> list:
    # The training sample pool repeats the same handful of driver
    # texts every cycle, so encoding per epoch would be waste
    if text not in _emb_cache:
        emb = encode_fn(text)
        _emb_cache[text] = [
            round(float(x), 6) for x in np.asarray(emb).ravel()
        ]
    return _emb_cache[text]


def record(step: int, text: str, embedding=None) -> None:
    _entries[int(step)] = {"text": text, "emb": embedding}


def get(step: int) -> str | None:
    e = _entries.get(int(step))
    if e is None:
        return None
    return e["text"] if isinstance(e, dict) else e


def entries_with_embeddings() -> list[tuple[int, dict]]:
    # public so tests / tools don't have to reach into _entries
    return [
        (step, e) for step, e in _entries.items()
        if isinstance(e, dict) and e.get("emb")
    ]


def next_timestamp(backend) -> int:
    """Step counter for runner-side writes. The runner's backend
    starts its counter at 0 while a loaded memory.bin can carry
    training-time timestamps, so writing without bumping would
    collide with existing journal keys and tier joins."""
    ts = [int(k) for k in _entries.keys()]
    mem = backend.get_memory_state()
    for level in ("short_term", "medium_term", "long_term"):
        for e in mem.get(level, {}).get("entries", []):
            ts.append(int(e.get("timestamp", -1)))
    return max(ts, default=-1) + 1


def save(path: str = JOURNAL_FILE) -> dict:
    with open(path, "w") as fh:
        json.dump({str(k): v for k, v in _entries.items()}, fh)
    return {"status": "saved", "file": path, "entries": len(_entries)}


def load(path: str = JOURNAL_FILE) -> dict:
    _entries.clear()
    if not Path(path).is_file():
        return {"status": "missing", "file": path, "entries": 0}
    with open(path) as fh:
        raw = json.load(fh)
    # values may be plain strings (pre-embedding journals) or dicts
    _entries.update({int(k): v for k, v in raw.items()})
    return {"status": "loaded", "file": path, "entries": len(_entries)}


class Reminiscence:
    """The LLM-facing memory tool: memorize / recall / check.

    memorize(text) writes a real C-side memory entry (the same call
    training uses) plus the journal text + embedding, and persists
    both files so the memory survives the process.

    recall(query) finds related memories, embedding mode by default
    (state mode kept for experiments - it lost the relevance test).

    check_consistency(claim) packages recall into a verdict:
    supported / related / unsupported. Embedding distance plays the
    novelty role here - unsupported means nothing like it was ever
    stored. The tool does NOT judge truth: a contradiction is
    semantically CLOSE to what it contradicts, so 'related' hits
    come back with their texts for the caller to judge agreement.
    """

    def __init__(self, runner) -> None:
        self.runner = runner

    def memorize(self, text: str) -> dict:
        backend = self.runner.backend
        backend._step_counter = next_timestamp(backend)
        # step first so the substrate enters the post-experience
        # state, then write the entry exactly like training does
        self.runner.step(text_input=text)
        mem_step = backend.add_memory_step()
        emb = encode_cached(
            text,
            lambda t: self.runner.model.embedder._st.encode(
                t, convert_to_numpy=True
            ),
        )
        record(mem_step["step"], text, emb)
        journal_result = save()
        mem_result = backend.save_memory(
            getattr(self.runner, "_mem_path", "memory.bin")
        )
        tier, imp = self._ts_map().get(
            mem_step["step"], ("(unknown)", 0.0)
        )
        return {
            "status":     "memorized",
            "step":       mem_step["step"],
            "tier":       tier,
            "importance": imp,
            "journal":    journal_result,
            "memory":     mem_result,
        }

    def check_consistency(self, claim: str, top_k: int = 3) -> dict:
        hits = self.recall(claim, top_k=top_k, mode="embedding")
        if not hits:
            return {"verdict": "no_memory", "support": 0.0, "hits": []}
        support = hits[0]["similarity"]
        if support >= SUPPORT_SIM:
            verdict = "supported"
        elif support >= RELATED_SIM:
            verdict = "related"
        else:
            verdict = "unsupported"
        return {"verdict": verdict, "support": support, "hits": hits}

    def _state_vector(self) -> np.ndarray:
        neurons = self.runner.backend.get_neurons()
        input_tensor = self.runner.backend.get_input_tensor()
        return (
            self.runner._build_default_weights(neurons, input_tensor)
            .detach().cpu().numpy().astype(np.float32)
        )

    def _ts_map(self) -> dict:
        mem = self.runner.backend.get_memory_state()
        ts_map = {}
        for level in ("short_term", "medium_term", "long_term"):
            for e in mem.get(level, {}).get("entries", []):
                ts_map[int(e.get("timestamp", -1))] = (
                    level, float(e.get("importance", 0.0))
                )
        return ts_map

    def _recall_state(self, query: str, top_k: int) -> list[dict]:
        self.runner.step(text_input=query)
        state = self._state_vector()
        mem = self.runner.backend.get_memory_state()

        cands = []
        for level in ("short_term", "medium_term", "long_term"):
            for e in mem.get(level, {}).get("entries", []):
                v = np.asarray(e.get("vector", []), dtype=np.float32)
                if v.shape != state.shape:
                    continue
                norms = np.linalg.norm(v) * np.linalg.norm(state)
                if norms < 1e-8:
                    continue
                cands.append({
                    "similarity": float(v @ state / norms),
                    "tier":       level,
                    "importance": float(e.get("importance", 0.0)),
                    "timestamp":  int(e.get("timestamp", -1)),
                })
        cands.sort(key=lambda c: -c["similarity"])

        out = []
        for c in cands[:top_k]:
            text = get(c["timestamp"])
            c["text"] = (
                text if text is not None
                else "(no text recorded for this entry)"
            )
            out.append(c)
        return out

    def _recall_embedding(
        self, query: str, top_k: int, tier: str | None
    ) -> list[dict]:
        encode = self.runner.model.embedder._st.encode
        q = np.asarray(
            encode(query, convert_to_numpy=True), dtype=np.float32
        ).ravel()
        qn = np.linalg.norm(q)
        ts_map = self._ts_map()

        cands = []
        for step, entry in _entries.items():
            if not isinstance(entry, dict) or not entry.get("emb"):
                continue
            v = np.asarray(entry["emb"], dtype=np.float32)
            denom = np.linalg.norm(v) * qn
            if denom < 1e-8:
                continue
            lvl, imp = ts_map.get(step, ("(not in memory)", 0.0))
            if tier is not None and lvl != tier:
                continue
            cands.append({
                "similarity": float(v @ q / denom),
                "tier":       lvl,
                "importance": imp,
                "timestamp":  step,
                "text":       entry["text"],
            })
        cands.sort(key=lambda c: -c["similarity"])

        # the sample pool repeats driver texts across epochs, so many
        # entries share one text with an identical embedding; keep only
        # the best-scoring copy or top_k returns the same text k times
        out = []
        seen_texts = set()
        for c in cands:
            if c["text"] in seen_texts:
                continue
            seen_texts.add(c["text"])
            out.append(c)
            if len(out) >= top_k:
                break
        return out

    def recall(
        self,
        query: str,
        top_k: int = 3,
        mode:  str = "embedding",
        tier:  str | None = None,
    ) -> list[dict]:
        if mode == "state":
            return self._recall_state(query, top_k)
        return self._recall_embedding(query, top_k, tier)
