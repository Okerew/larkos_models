"""Text journal for memory entries.

Every C-side MemoryEntry carries the training step counter as its
timestamp, and at that step the training loop knows the driver text.
We persist {step: {text, embedding}} as JSON next to memory.bin, so
inference can answer "what was the system reading when this entry
formed?".

"""

import json
import time
from pathlib import Path
from modules.config import (
    JOURNAL_FILE, SUPPORT_SIM, RELATED_SIM, DUPLICATE_SIM,
)

import numpy as np

_entries: dict[int, dict] = {}
_emb_cache: dict[str, list] = {}
# steps written by record() in THIS process: their journal<->C-side
# pairing was witnessed live. Anything else is joined post-hoc by
# integer key across two files that may come from different runs
# (memory.bin restored without the journal, or vice versa), and the
# join cannot be verified from here - recall flags those hits as
# "possible_mismatch" instead of pretending they are clean.
_live_steps: set[int] = set()


def reset() -> None:
    _entries.clear()
    _emb_cache.clear()
    _live_steps.clear()


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
    # wall-clock rides along with the training step so recall can
    # tell the model "recorded today" from "recorded weeks ago" -
    # step order alone is meaningless across runs and reloads
    _entries[int(step)] = {
        "text": text, "emb": embedding, "time": time.time(),
    }
    _live_steps.add(int(step))


def get(step: int) -> str | None:
    e = _entries.get(int(step))
    if e is None:
        return None
    return e["text"] if isinstance(e, dict) else e


def get_time(step: int) -> float | None:
    e = _entries.get(int(step))
    if isinstance(e, dict):
        t = e.get("time")
        return float(t) if t is not None else None
    return None


def age_str(step: int) -> str:
    t = get_time(step)
    if t is None:
        return "age unknown"
    s = max(0.0, time.time() - t)
    if s < 60.0:
        return "just now"
    if s < 3600.0:
        return f"{int(s // 60)}m ago"
    if s < 86400.0:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


def _provenance(step: int, has_text: bool, joined: bool) -> str:
    # "journal": text stored together with the entry, or the pairing
    #   was written live in this process
    # "possible_mismatch": text/tier attached by an integer-key join
    #   against file-loaded data - honest, but unverifiable here
    # "placeholder": no text exists for the entry at all
    if not has_text:
        return "placeholder"
    if joined and int(step) not in _live_steps:
        return "possible_mismatch"
    return "journal"


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
    ts.extend(t for _, t, _ in backend.get_entry_meta())
    return max(ts, default=-1) + 1


def save(path: str = JOURNAL_FILE) -> dict:
    with open(path, "w") as fh:
        json.dump({str(k): v for k, v in _entries.items()}, fh)
    return {"status": "saved", "file": path, "entries": len(_entries)}


def load(path: str = JOURNAL_FILE) -> dict:
    _entries.clear()
    # loaded entries were not witnessed being written next to their
    # C-side counterparts, so they start out non-live for provenance
    _live_steps.clear()
    if not Path(path).is_file():
        return {"status": "missing", "file": path, "entries": 0}
    with open(path) as fh:
        raw = json.load(fh)
    # values may be plain strings (pre-embedding journals) or dicts
    _entries.update({int(k): v for k, v in raw.items()})
    return {"status": "loaded", "file": path, "entries": len(_entries)}


def snapshot_live_steps() -> set[int]:
    # Copy of the steps recorded live in this process. load() clears
    # the witness set, so a caller about to reload (the service
    # rebuilding its runner after a train) snapshots first and calls
    # mark_live() after, so entries written moments ago are not
    # downgraded to possible_mismatch on the next recall.
    return set(_live_steps)


def mark_live(steps) -> None:
    # Re-union steps into the live-witness set (additive, never clears).
    _live_steps.update(int(s) for s in steps)


class Reminiscence:
    """The LLM-facing memory tool: memorize / recall / check.

    memorize(text) writes a real C-side memory entry (the same call
    training uses) plus the journal text + embedding, and persists
    both files so the memory survives the process. Writing the same
    fact twice creates two near-identical entries (recall's dedupe
    hides the pile-up), so the response carries a duplicate_of note
    when nearly the same text is already stored.

    recall(query) finds related memories, embedding mode by default
    (state mode kept for experiments - it lost the relevance test).
    Every hit carries text_provenance (journal / possible_mismatch /
    placeholder), recorded_at and a human-readable age, so the caller
    can tell a clean hit from a suspect post-reload join and place
    memories in time. recall_with_stats() additionally reports how
    many entries were scanned and the best similarity seen, which
    separates "nothing relevant exists" from "the search missed".

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
        # encode up front: the duplicate scan needs the embedding
        # before record() adds the new entry (else it matches itself)
        emb = encode_cached(
            text,
            lambda t: self.runner.model.embedder._st.encode(
                t, convert_to_numpy=True
            ),
        )
        dup = self._find_duplicate(emb)
        backend._step_counter = next_timestamp(backend)
        # step first so the substrate enters the post-experience
        # state, then write the entry exactly like training does
        self.runner.step(text_input=text)
        mem_step = backend.add_memory_step()
        record(mem_step["step"], text, emb)
        journal_result = save()
        mem_result = backend.save_memory(
            getattr(self.runner, "_mem_path", "memory.bin")
        )
        tier, imp = self._ts_map().get(
            mem_step["step"], ("(unknown)", 0.0)
        )
        result = {
            "status":     "memorized",
            "step":       mem_step["step"],
            "tier":       tier,
            "importance": imp,
            "journal":    journal_result,
            "memory":     mem_result,
        }
        if dup is not None:
            result["duplicate_of"] = dup
            result["note"] = (
                f"nearly the same text is already stored at step "
                f"{dup['step']} (sim {dup['similarity']:.2f}); "
                f"a new entry was written anyway"
            )
        return result

    def _find_duplicate(self, emb) -> dict | None:
        v = np.asarray(emb, dtype=np.float32).ravel()
        vn = np.linalg.norm(v)
        best = None
        for step, e in _entries.items():
            if not isinstance(e, dict) or not e.get("emb"):
                continue
            w = np.asarray(e["emb"], dtype=np.float32)
            denom = np.linalg.norm(w) * vn
            if denom < 1e-8:
                continue
            sim = float(v @ w / denom)
            if sim >= DUPLICATE_SIM and (
                best is None or sim > best[1]
            ):
                best = (step, sim)
        if best is None:
            return None
        return {
            "step":       best[0],
            "similarity": round(best[1], 4),
            "text":       get(best[0]),
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
        # hits[0] alone only shows the single best match; the band
        # counts tell the caller how much of the neighborhood agrees
        return {
            "verdict": verdict,
            "support": support,
            "supporting_hits": sum(
                1 for h in hits if h["similarity"] >= SUPPORT_SIM
            ),
            "related_hits": sum(
                1 for h in hits if h["similarity"] >= RELATED_SIM
            ),
            "hits": hits,
        }

    def _state_vector(self) -> np.ndarray:
        neurons = self.runner.backend.get_neurons()
        input_tensor = self.runner.backend.get_input_tensor()
        return (
            self.runner._build_default_weights(neurons, input_tensor)
            .detach().cpu().numpy().astype(np.float32)
        )

    def _ts_map(self) -> dict:
        return {
            ts: (level, imp)
            for level, ts, imp in
            self.runner.backend.get_entry_meta()
        }

    def _recall_state(
        self, query: str, top_k: int
    ) -> tuple[list[dict], int, float | None]:
        self.runner.step(text_input=query)
        state = self._state_vector()

        cands = []
        for level, ts, imp, v in (
            self.runner.backend.get_entry_views()
        ):
            if v.shape != state.shape:
                continue
            norms = np.linalg.norm(v) * np.linalg.norm(state)
            if norms < 1e-8:
                continue
            cands.append({
                "similarity": float(v @ state / norms),
                "tier":       level,
                "importance": imp,
                "timestamp":  ts,
            })
        cands.sort(key=lambda c: -c["similarity"])
        best = cands[0]["similarity"] if cands else None

        out = []
        for c in cands[:top_k]:
            text = get(c["timestamp"])
            c["text"] = (
                text if text is not None
                else "(no text recorded for this entry)"
            )
            # the text here is attached by an integer-key join against
            # the journal - clean only if this process watched the
            # entry being written
            c["text_provenance"] = _provenance(
                c["timestamp"], text is not None, True
            )
            c["recorded_at"] = get_time(c["timestamp"])
            c["age"] = age_str(c["timestamp"])
            out.append(c)
        return out, len(cands), best

    def _recall_embedding(
        self, query: str, top_k: int, tier: str | None
    ) -> tuple[list[dict], int, float | None]:
        encode = self.runner.model.embedder._st.encode
        q = np.asarray(
            encode(query, convert_to_numpy=True), dtype=np.float32
        ).ravel()
        qn = np.linalg.norm(q)
        ts_map = self._ts_map()

        scanned = 0
        cands = []
        for step, entry in _entries.items():
            if not isinstance(entry, dict) or not entry.get("emb"):
                continue
            scanned += 1
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
                # the text travels inside the journal record, but
                # tier/importance come from an integer-key join
                # against the C side - flag that join when it was
                # not witnessed live
                "text_provenance": _provenance(
                    step, True, step in ts_map
                ),
                "recorded_at": entry.get("time"),
                "age":         age_str(step),
            })
        cands.sort(key=lambda c: -c["similarity"])
        best = cands[0]["similarity"] if cands else None

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
        return out, scanned, best

    def recall_with_stats(
        self,
        query: str,
        top_k: int = 3,
        mode:  str = "embedding",
        tier:  str | None = None,
    ) -> dict:
        """recall() plus the scan telemetry: an empty hits list alone
        cannot distinguish "nothing relevant is stored" from "the
        query never reached the corpus", so callers also get the
        number of entries scanned and the best similarity seen."""
        if mode == "state":
            hits, scanned, best = self._recall_state(query, top_k)
        else:
            hits, scanned, best = self._recall_embedding(
                query, top_k, tier
            )
        return {
            "hits":            hits,
            "scanned":         scanned,
            "best_similarity": (
                round(best, 6) if best is not None else None
            ),
        }

    def recall(
        self,
        query: str,
        top_k: int = 3,
        mode:  str = "embedding",
        tier:  str | None = None,
    ) -> list[dict]:
        return self.recall_with_stats(
            query, top_k=top_k, mode=mode, tier=tier
        )["hits"]
