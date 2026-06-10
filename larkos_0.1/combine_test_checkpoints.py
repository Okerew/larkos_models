"""
Combine all test checkpoints in CKPT_DIR into a single pre_model.pt
and pre_model_memory.bin, suitable as a general pre-trained starting point.

Combination strategy
--------------------
Model weights (model, fusion_transformer, embed_weight_net, cross_attn,
text_proj, ema_shadow):
    Softmax-weighted average keyed on epochs_done.  A checkpoint with
    more training contributes proportionally more to the final weights —
    equivalent to a confidence-weighted model ensemble.

online_norm (running min/max):
    Element-wise true min of all mins and true max of all maxes.  The
    merged normaliser has "seen" the union of every observation range, so
    it will not clip inputs that any individual checkpoint handled.
    seen counts are summed; momentum is averaged.

fused_cog_norm (running mean/variance):
    Parallel Welford formula - the exact, lossless merge for independent
    running statistics.  combined_mean = sum(n_i * mean_i) / N and
    combined_var = sum(n_i * (var_i + (mean_i - combined_mean)^2)) / N.

text_num_to_prefix:
    Fixed random projection, never optimised.  Taken from the most-
    trained checkpoint so it matches the longest learning trajectory.

Transient / domain-specific state (cached_target, cached_fused_cog,
input_history):
    Cleared.  The combined model starts domain-agnostic with no stale
    freeze-window artefacts.

Text driver state (sample_pool, sample_pool_idx, current_text_input,
text_encoding):
    Copied from the most-trained checkpoint to preserve a consistent
    text-domain context.

Memory (.bin files):
    All entries from all files and all hierarchy levels are pooled.
    Near-duplicate entries (cosine similarity ≥ 0.9) are merged into a
    single cluster whose vector is the importance-weighted centroid.
    Each cluster is scored as mean_importance x sqrt(cluster_size), so
    memories reinforced across multiple checkpoints rank above one-off
    entries.  The top MEMORY_CAPACITY clusters are written into a fresh
    BackendState and saved.

Usage
-----
    # standalone
    python modules/combine_test_checkpoints.py
    python modules/combine_test_checkpoints.py --ckpt-dir test_checkpoints --out-dir .

    # imported
    from modules.combine_test_checkpoints import combine
    pt_path, mem_path = combine()
"""

from __future__ import annotations

import argparse
import faulthandler
import struct
import sys
from pathlib import Path

import numpy as np
import torch

faulthandler.enable()
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from modules.config import (
    CKPT_DIR,
    CHECKPOINT_VERSION,
    MEMORY_VECTOR_SIZE,
    MEMORY_CAPACITY,
)


_CKPT_DIR = _ROOT / CKPT_DIR


def _softmax(values: list) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    v -= v.max()
    e = np.exp(v)
    return e / e.sum()


def _wavg_tensors(tensors: list, weights: np.ndarray) -> torch.Tensor:
    stacked = torch.stack([t.float() for t in tensors], dim=0)
    w = torch.tensor(weights, dtype=torch.float32)
    while w.dim() < stacked.dim():
        w = w.unsqueeze(-1)
    return (stacked * w).sum(dim=0)


def _avg_state_dicts(dicts: list, weights: np.ndarray) -> dict:
    combined = {}
    for key in dicts[0]:
        combined[key] = _wavg_tensors([d[key] for d in dicts], weights)
    return combined


def _merge_online_minmax(norms: list) -> dict:
    """
    Merge online min/max normalisers: element-wise min of all mins,
    element-wise max of all maxes, summed seen counts, averaged momentum.
    """
    merged_min = torch.stack([n["min"].float() for n in norms], 0).min(0).values
    merged_max = torch.stack([n["max"].float() for n in norms], 0).max(0).values

    merged_seen = norms[0]["seen"].clone()
    for n in norms[1:]:
        merged_seen = merged_seen + n["seen"]

    merged_mom = float(np.mean([n["mom"] for n in norms]))
    return {"min": merged_min, "max": merged_max, "seen": merged_seen, "mom": merged_mom}


def _merge_online_meanstd(norms: list) -> dict:
    """
    Merge online mean/variance normalisers using the parallel Welford formula.
    This is the exact lossless combination for independent running statistics.
    """
    def _to_float(v):
        return float(v.item()) if isinstance(v, torch.Tensor) else float(v)

    seens = [_to_float(n["seen"]) for n in norms]
    total = sum(seens)
    if total < 1.0:
        return norms[0]

    means = [n["mean"].float() for n in norms]
    vars_ = [n["var"].float()  for n in norms]

    combined_mean = sum((s / total) * m for s, m in zip(seens, means))
    combined_var  = sum(
        (s / total) * (v + (m - combined_mean) ** 2)
        for s, m, v in zip(seens, means, vars_)
    )

    return {
        "mean":  combined_mean,
        "var":   combined_var,
        "seen":  int(sum(seens)),
        "mom":   float(np.mean([n["mom"]   for n in norms])),
        "floor": float(np.mean([n["floor"] for n in norms])),
    }


_MEM_ENTRY_FMT = f"<{MEMORY_VECTOR_SIZE}f f I"
_MEM_ENTRY_SZ = struct.calcsize(_MEM_ENTRY_FMT)


def _read_bin_entries(path: Path) -> list:
    """Parse a memory .bin file in pure Python - no C library needed.

    Mirrors the binary layout of ``saveMemorySystem`` / ``loadMemorySystem``
    in neural_web.c so the files are byte-for-byte compatible.
    """
    raw = path.read_bytes()
    pos = 0

    def _u32() -> int:
        nonlocal pos
        v = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        return v

    capacity = _u32()
    _u32()  # total size (unused in combine)
    _u32()  # head (unused)

    st_cap = int(capacity * 0.5)
    mt_cap = int(capacity * 0.3)
    lt_cap = int(capacity * 0.2)

    entries: list = []

    for max_slots in (st_cap, mt_cap, lt_cap):
        sz = min(_u32(), max_slots)
        for _ in range(sz):
            fields = struct.unpack_from(_MEM_ENTRY_FMT, raw, pos)
            pos += _MEM_ENTRY_SZ
            vec = np.clip(
                np.array(fields[:MEMORY_VECTOR_SIZE], dtype=np.float32),
                -1e6, 1e6,
            )
            entries.append((vec, float(fields[MEMORY_VECTOR_SIZE])))
        # skip remaining (unused) slots that were serialised
        pos += (max_slots - sz) * _MEM_ENTRY_SZ

    return entries


def _write_bin_entries(
    entries: list, out_path: Path, capacity: int = MEMORY_CAPACITY,
) -> None:
    """Write memory entries as a .bin file readable by ``loadMemorySystem``."""
    st_cap = int(capacity * 0.5)
    mt_cap = int(capacity * 0.3)
    lt_cap = int(capacity * 0.2)

    buf = bytearray()
    buf += struct.pack("<III", capacity, 0, 0)  # capacity, total_size=0, head=0

    # short term - top third of entries
    n_st = min(len(entries), st_cap)
    buf += struct.pack("<I", n_st)
    for idx in range(n_st):
        vec, imp = entries[idx]
        buf += _pack_entry(vec, imp, idx)
    buf += b"\x00" * ((st_cap - n_st) * _MEM_ENTRY_SZ)

    # medium term - middle third
    n_mt = min(len(entries) - n_st, mt_cap)
    buf += struct.pack("<I", n_mt)
    for idx in range(n_st, n_st + n_mt):
        vec, imp = entries[idx]
        buf += _pack_entry(vec, imp, idx)
    buf += b"\x00" * ((mt_cap - n_mt) * _MEM_ENTRY_SZ)

    # long term - bottom third
    n_lt = min(len(entries) - n_st - n_mt, lt_cap)
    buf += struct.pack("<I", n_lt)
    for idx in range(n_st + n_mt, n_st + n_mt + n_lt):
        vec, imp = entries[idx]
        buf += _pack_entry(vec, imp, idx)
    buf += b"\x00" * ((lt_cap - n_lt) * _MEM_ENTRY_SZ)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(buf))


def _pack_entry(vec: np.ndarray, imp: float, rank: int) -> bytes:
    return struct.pack(
        _MEM_ENTRY_FMT,
        *(float(np.clip(vec[i], -1e6, 1e6)) for i in range(MEMORY_VECTOR_SIZE)),
        float(np.clip(imp, 0.0, 1.0)),
        rank,
    )


def _pool_memory_entries(bin_paths: list) -> list:
    """
    Load every memory .bin file and return a flat list of
    (vector: np.ndarray, importance: float) from all hierarchy levels.

    Uses a pure-Python ``_read_bin_entries`` so the caller never touches
    the C shared library - no dlopen, no ctypes, no SSHFS fread segfaults.
    """
    all_entries: list[tuple] = []
    for p in bin_paths:
        try:
            all_entries.extend(_read_bin_entries(p))
        except Exception as exc:
            print(f"  [warn] skipping {p.name}: {exc}")
    return all_entries


def _cosine_cluster(entries: list, threshold: float = 0.9) -> list:
    """
    Greedy cosine-similarity clustering.  Each new entry is merged into
    the first existing cluster whose seed has cosine similarity ≥ threshold,
    or starts a new cluster.  Returns a list of entry-lists.
    """
    if not entries:
        return []

    unit_vecs = []
    for vec, _ in entries:
        vec64 = vec.astype(np.float64)
        vec64 = np.clip(vec64, -1e6, 1e6)
        n = np.linalg.norm(vec64)
        unit_vecs.append(vec64 / (n + 1e-8))

    assigned = [False] * len(entries)
    clusters: list[list] = []

    for i, (vec_i, imp_i) in enumerate(entries):
        if assigned[i]:
            continue
        cluster = [(vec_i, imp_i)]
        assigned[i] = True
        for j in range(i + 1, len(entries)):
            if assigned[j]:
                continue
            if float(np.dot(unit_vecs[i], unit_vecs[j])) >= threshold:
                cluster.append(entries[j])
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def _combine_memory_files(bin_paths: list, out_path: Path) -> None:
    """
    Pool entries from all binary memory files, cluster near-duplicates,
    score by importance x sqrt(cluster_size) to reward consistent memories,
    then write the top MEMORY_CAPACITY entries into a fresh .bin file.

    Entirely pure Python - no C library, no ctypes, no dlopen.
    On SSHFS the C library path (neural_web.so load / fread / fwrite)
    frequently segfaults; this avoids it completely.
    """
    all_entries = _pool_memory_entries(bin_paths)

    if not all_entries:
        print("  [warn] no memory entries found - saving empty memory")
        _write_bin_entries([], out_path)
        return

    clusters = _cosine_cluster(all_entries)

    scored: list[tuple] = []
    for cluster in clusters:
        vecs = np.stack([v for v, _ in cluster])
        imps = np.array([i for _, i in cluster], dtype=np.float32)
        merged_vec = np.average(vecs, axis=0, weights=imps + 1e-8)
        mean_imp   = float(np.mean(imps))
        score      = mean_imp * np.sqrt(len(cluster))
        scored.append((score, merged_vec, mean_imp))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_entries = scored[:MEMORY_CAPACITY]

    _write_bin_entries(
        [(vec, imp) for _, vec, imp in top_entries],
        out_path,
    )
    print(
        f"  memory: {len(all_entries)} pooled → {len(clusters)} clusters "
        f"→ {len(top_entries)} written to {out_path.name}"
    )


def combine(
    ckpt_dir: str | Path | None = None,
    out_pt:   str | Path | None = None,
    out_mem:  str | Path | None = None,
) -> tuple[str, str]:
    """
    Combine all test checkpoints into pre_model.pt and pre_model_memory.bin.

    Parameters
    ----------
    ckpt_dir : path to the checkpoint directory (default: CKPT_DIR from config)
    out_pt   : output path for the combined .pt file
    out_mem  : output path for the combined _memory.bin file

    Returns
    -------
    (pt_path, mem_path) as strings
    """
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else _CKPT_DIR
    out_pt   = Path(out_pt)   if out_pt   else _ROOT / "pre_model.pt"
    out_mem  = Path(out_mem)  if out_mem  else _ROOT / "pre_model_memory.bin"

    pt_files = sorted([
        p for p in ckpt_dir.glob("*.pt")
        if not p.stem.startswith("pre_model")
    ])
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint files found in {ckpt_dir}")

    print(f"\nCombining {len(pt_files)} checkpoint(s) from {ckpt_dir}\n")

    ckpts: list[tuple] = []
    for p in pt_files:
        c = torch.load(str(p), map_location="cpu", weights_only=False)
        ckpts.append((p, c))

    ckpts.sort(key=lambda x: x[1].get("epochs_done", 0))
    dicts  = [c for _, c in ckpts]
    epochs = [max(1, c.get("epochs_done", 1)) for _, c in ckpts]
    weights = _softmax(epochs)

    print("  weight (softmax of epochs_done):")
    for (p, c), w in zip(ckpts, weights):
        print(f"    {p.name:45s}  epochs={c.get('epochs_done','?'):>4}  w={w:.4f}")

    combined: dict = {
        "version":     CHECKPOINT_VERSION,
        "epochs_done": sum(epochs),
    }

    for key in (
        "model", "fusion_transformer", "embed_weight_net",
        "cross_attn", "text_proj",
    ):
        combined[key] = _avg_state_dicts([d[key] for d in dicts], weights)

    combined["ema_shadow"] = _avg_state_dicts(
        [d["ema_shadow"] for d in dicts], weights
    )

    # Fixed random projection - use the most-trained checkpoint's copy.
    combined["text_num_to_prefix"] = dicts[-1]["text_num_to_prefix"]

    combined["online_norm"]    = _merge_online_minmax(
        [d["online_norm"] for d in dicts]
    )
    combined["fused_cog_norm"] = _merge_online_meanstd(
        [d["fused_cog_norm"] for d in dicts]
    )

    combined["cached_target"]    = None
    combined["cached_fused_cog"] = None
    combined["target_epoch"]     = 0
    combined["input_history"]    = []

    # Text driver state: carry forward the most-trained checkpoint's context.
    for key in (
        "sample_pool", "sample_pool_idx",
        "current_text_input", "text_encoding",
    ):
        combined[key] = dicts[-1][key]

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, str(out_pt))
    print(f"\n  saved model checkpoint → {out_pt}")

    bin_files = sorted([
        p for p in ckpt_dir.glob("*_memory.bin")
        if not p.stem.startswith("pre_model")
    ])
    print(f"\nCombining {len(bin_files)} memory file(s) ...")
    _combine_memory_files(bin_files, out_mem)
    print(f"  saved memory         → {out_mem}")

    return str(out_pt), str(out_mem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine all test checkpoints in CKPT_DIR into "
            "pre_model.pt and pre_model_memory.bin"
        )
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help=f"Source checkpoint directory (default: {CKPT_DIR}/)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: project root)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT
    pt_path, mem_path = combine(
        ckpt_dir=args.ckpt_dir,
        out_pt=out_dir / "pre_model.pt",
        out_mem=out_dir / "pre_model_memory.bin",
    )
    print(f"\nDone.\n  model:  {pt_path}\n  memory: {mem_path}\n")
