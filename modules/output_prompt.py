"""End-of-run output prompt construction.

Collects compact cognitive-state snapshots while training runs and
while the runner performs inference, then builds one final plain-text
prompt that a normal LLM (paired with INTERPRETER_PROMPT) can read.
The data block itself carries no instructions, only labelled state.

Merely a debug tool, don't use it in pair it in produciton with actual llms
memory_journal exists for a reason.
"""

import numpy as np

from modules.config import BAND_Q

OUTPUT_PROMPT_FILE = "output_prompt.txt"
INTERPRETER_FILE   = "output_prompt_interpreter.txt"

TRAJECTORY_TAIL = 25
TOP_DIMS        = 8
TEXT_CLIP       = 240
DESC_CLIP       = 120

INTERPRETER_PROMPT = """\
You are given a LARKOS OUTPUT PROMPT: a plain-text snapshot of the
Larkos cognitive architecture (a C backend coupled to a torch model)
captured at the end of training and during inference. The block is
pure labelled data - it contains no instructions. Read it as the
system's cognitive state report and reason about it.

FORMAT
- Sections are marked [like_this]. Fields are "name: value" lines.
- Vectors appear either in full as [v0, v1, ...] or summarised as
  mean/std/min/max/l2 plus dominant_dims (largest |value| indices).
- Scores are floats in [0, 1] unless stated otherwise.

SECTIONS
[run] training counters: loss (SmoothL1 training loss, lower is
  better), learning_rate, alpha (context blend fed into fusion).
[text] the driver sentence (input) and the GPT-2 readout of the
  cognitive vector (decoded). The decoded text is a noisy window into
  the fused state, not a trained objective - treat it as qualitative.
[outputs] model_prediction is the torch model's raw output vector,
  neuron_prediction is the C-side target it trains against (training
  only), fused_output is the final fused head output vector.
[fusion_state] statistics of the fused cognitive vector. band_q is
  the query/text half [0:512), band_m the memory half [512:1024).
  Comparing band means/stds shows which stream dominates the fusion.
[meta_parameters] the meta-learning controller (meta learning rate,
  exploration factor, per-region importance), metacognition
  (confidence, cognitive load, error awareness, context relevance)
  and the learning state (efficiency, exploration, stability, phase).
[imagination_parameters] the scenario simulator: creativity factor,
  novelty weighting, memory/identity influence, and the current
  scenario with its most probable imagined outcomes.
[dynamic_parameters] the C-side self-adaptation knobs (noise scales,
  adaptation rates, momentum, plasticity, homeostasis) plus stability,
  which measures how much the last step moved the neuron states.
  Healthy training sits around 0.15-0.6; near 0 means frozen
  dynamics, near 1 means wild churn.
[memory] the hierarchical memory (short/medium/long term): fill per
  level, importance distribution, and the strongest stored entry.
[reflection_parameters] the self-reflection read: confidence,
  coherence, novelty, consistency and a confabulation flag.
  derived_drift = 1 - consistency, floored at 0.5 when the system
  confabulates; high drift means the self-model is diverging.
  consistency is measured against consistency_baseline, the system's
  own running norm - steady churn reads as consistent, a regime
  shift does not. Thresholds and recent history show the trend;
  params are the reflection-driven noise/plasticity knobs.
[training_trajectory] per-epoch scalar history (only the tail is
  shown): loss, lr, alpha, stability, confidence, drift.

REASONING GUIDE
- A healthy run: loss trending down, drift moderate (< 0.7),
  stability inside 0.15-0.6, imagination active with plausible
  outcomes, memory consolidating (entries moving short -> medium ->
  long term over time).
- Compare the training snapshot against the inference snapshots:
  large shifts in band statistics or output vectors mean inference
  conditions differ from what the model was trained under.
- The decoded text often reads as loose associations. Anchor on the
  numeric fields and use the text only as qualitative flavour.
"""

_trajectory:      list[dict] = []
_training_final:  dict       = {}
_inference_steps: list[dict] = []


def reset() -> None:
    _trajectory.clear()
    _training_final.clear()
    _inference_steps.clear()


def _arr(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).ravel()


def _to_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return _arr(v)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _stats_line(name: str, v: np.ndarray) -> str:
    if v.size == 0:
        return f"{name}: empty"
    return (
        f"{name}[{v.size}]: "
        f"mean={v.mean():.4f} std={v.std():.4f} "
        f"min={v.min():.4f} max={v.max():.4f} "
        f"l2={float(np.linalg.norm(v)):.4f}"
    )


def _top_dims(v: np.ndarray, k: int = TOP_DIMS) -> str:
    if v.size == 0:
        return "dominant_dims: none"
    idx = np.argsort(-np.abs(v))[:k]
    pairs = " ".join(f"i{i}={v[i]:.4f}" for i in idx)
    return f"dominant_dims(top {k} by |value|): {pairs}"


def _vec_list(name: str, v: np.ndarray) -> str:
    body = ", ".join(f"{x:.4f}" for x in v)
    return f"{name}[{v.size}]: [{body}]"


def _reflect_drift(metrics: dict) -> float:
    # Mirrors _derive_reflection_signal in training.py, duplicated here
    # because training imports this module, not the other way around.
    drift = 1.0 - float(metrics.get("consistency_score", 1.0))
    if metrics.get("potentially_confabulated", False):
        drift = max(drift, 0.5)
    return min(max(drift, 0.0), 1.0)


def capture_training_state(
    *,
    epoch:           int,
    loss_val:        float,
    lr:              float,
    alpha:           float,
    text_input:      str,
    text_output:     str,
    fused_cog,
    fused_np,
    model_pred_np,
    neuron_pred,
    backend,
    mem_state:       dict,
    reflect_metrics: dict,
    stability:       float,
) -> None:
    """Called once per epoch from TrainingLoop._side_effects. Keeps a
    scalar trajectory line per epoch and overwrites the full snapshot,
    so only the final (most influential) training state is retained.
    """
    _trajectory.append({
        "epoch":      epoch,
        "loss":       float(loss_val),
        "lr":         float(lr),
        "alpha":      float(alpha),
        "stability":  float(stability),
        "confidence": float(reflect_metrics.get("confidence_score", 0.5)),
        "drift":      _reflect_drift(reflect_metrics),
    })
    _training_final.clear()
    _training_final.update({
        "epoch":       epoch,
        "loss":        float(loss_val),
        "lr":          float(lr),
        "alpha":       float(alpha),
        "text_input":  text_input,
        "text_output": text_output,
        "fused_cog":   _to_np(fused_cog),
        "fused":       _to_np(fused_np),
        "model_pred":  _to_np(model_pred_np),
        "neuron_pred": _to_np(neuron_pred),
        "meta":        backend.get_meta_state(),
        "imagination": backend.get_imagination_state(),
        "dynamic":     backend.get_dynamic_params(),
        "stability":   float(stability),
        "memory":      mem_state,
        "reflection": {
            "state":   backend.get_reflection_state(),
            "metrics": dict(reflect_metrics),
        },
    })


def record_inference_step(
    step_index: int, result: dict, backend
) -> None:
    """Called once per runner step from run_model. The runner never
    runs the C-side reflection, so only the stored reflection state is
    read here, metrics stay empty."""
    _inference_steps.append({
        "step":        step_index,
        "text_input":  result.get("text_input", ""),
        "text_output": result.get("text_output", ""),
        "model_pred":  _to_np(result.get("model_pred", [])),
        "fused":       _to_np(result.get("fused", [])),
        "fused_cog":   _to_np(result.get("fused_cog", [])),
        "meta":        backend.get_meta_state(),
        "imagination": backend.get_imagination_state(),
        "dynamic":     backend.get_dynamic_params(),
        "memory":      backend.get_memory_state(),
        "reflection": {
            "state":   backend.get_reflection_state(),
            "metrics": {},
        },
    })


def _lines_trajectory() -> list[str]:
    lines = ["[training_trajectory]"]
    if not _trajectory:
        lines.append("no training epochs recorded this process")
        return lines
    losses = [t["loss"] for t in _trajectory]
    tail   = losses[-10:]
    lines.append(f"epochs_recorded: {len(_trajectory)}")
    lines.append(f"final_loss: {losses[-1]:.6f}")
    lines.append(f"best_loss: {min(losses):.6f}")
    lines.append(f"mean_loss_last_{len(tail)}: {sum(tail) / len(tail):.6f}")
    lines.append(f"recent_epochs (last {TRAJECTORY_TAIL}):")
    for t in _trajectory[-TRAJECTORY_TAIL:]:
        lines.append(
            f"  ep {t['epoch']}: "
            f"loss={t['loss']:.6f} "
            f"lr={t['lr']:.6f} "
            f"alpha={t['alpha']:.4f} "
            f"stability={t['stability']:.4f} "
            f"confidence={t['confidence']:.4f} "
            f"drift={t['drift']:.4f}"
        )
    return lines


def _lines_run(snap: dict) -> list[str]:
    return [
        "[run]",
        f"epoch: {snap['epoch']}",
        f"loss: {snap['loss']:.6f}",
        f"learning_rate: {snap['lr']:.6f}",
        f"alpha: {snap['alpha']:.4f}",
    ]


def _lines_text(snap: dict) -> list[str]:
    return [
        "[text]",
        f"input: {_clip(snap.get('text_input', ''), TEXT_CLIP)}",
        f"decoded: {_clip(snap.get('text_output', ''), TEXT_CLIP)}",
    ]


def _lines_outputs(snap: dict, training: bool) -> list[str]:
    lines = ["[outputs]"]
    lines.append("  " + _vec_list("model_prediction", snap["model_pred"]))
    if training and snap.get("neuron_pred") is not None:
        lines.append(
            "  " + _vec_list("neuron_prediction", snap["neuron_pred"])
        )
    lines.append("  " + _vec_list("fused_output", snap["fused"]))
    return lines


def _lines_fusion(snap: dict) -> list[str]:
    lines = ["[fusion_state]"]
    fc = snap.get("fused_cog")
    if fc is None or fc.size == 0:
        lines.append("  fused_cog: not captured")
        return lines
    lines.append("  " + _stats_line("fused_cog", fc))
    lines.append("  " + _stats_line("band_q", fc[:BAND_Q]))
    lines.append("  " + _stats_line("band_m", fc[BAND_Q:]))
    lines.append("  " + _top_dims(fc))
    return lines


def _lines_meta(meta: dict) -> list[str]:
    lines = ["[meta_parameters]"]
    ctrl = meta.get("controller", {}) or {}
    lines.append(
        "controller: "
        f"meta_learning_rate="
        f"{ctrl.get('meta_learning_rate', 0.0):.6f} "
        f"exploration_factor={ctrl.get('exploration_factor', 0.0):.4f} "
        f"num_regions={ctrl.get('num_regions', 0)}"
    )
    ris = _arr(ctrl.get("region_importance_scores", []))
    if ris.size:
        lines.append("  " + _vec_list("region_importance_scores", ris))
    leh = _arr(ctrl.get("learning_efficiency_history", []))
    if leh.size:
        lines.append(
            "  " + _vec_list("learning_efficiency_history", leh)
        )
    mc = meta.get("metacognition", {}) or {}
    lines.append(
        "metacognition: "
        f"confidence_level={mc.get('confidence_level', 0.0):.4f} "
        f"adaptation_rate={mc.get('adaptation_rate', 0.0):.4f} "
        f"cognitive_load={mc.get('cognitive_load', 0.0):.4f} "
        f"error_awareness={mc.get('error_awareness', 0.0):.4f} "
        f"context_relevance={mc.get('context_relevance', 0.0):.4f}"
    )
    ph = _arr(mc.get("performance_history", []))
    if ph.size:
        lines.append("  " + _vec_list("performance_history", ph))
    ls = meta.get("learning_state", {}) or {}
    lines.append(
        "learning_state: "
        f"learning_efficiency={ls.get('learning_efficiency', 0.0):.4f} "
        f"exploration_rate={ls.get('exploration_rate', 0.0):.4f} "
        f"stability_index={ls.get('stability_index', 0.0):.4f} "
        f"current_phase={ls.get('current_phase', 0)}"
    )
    return lines


def _lines_imagination(imag: dict) -> list[str]:
    lines = ["[imagination_parameters]"]
    lines.append(
        f"active={imag.get('active', False)} "
        f"creativity_factor={imag.get('creativity_factor', 0.0):.4f} "
        f"coherence_threshold="
        f"{imag.get('coherence_threshold', 0.0):.4f} "
        f"novelty_weight={imag.get('novelty_weight', 0.0):.4f}"
    )
    lines.append(
        f"memory_influence={imag.get('memory_influence', 0.0):.4f} "
        f"identity_influence={imag.get('identity_influence', 0.0):.4f} "
        f"steps_simulated={imag.get('steps_simulated', 0)} "
        f"total_scenarios_generated="
        f"{imag.get('total_scenarios_generated', 0)}"
    )
    lines.append(
        f"num_scenarios={imag.get('num_scenarios', 0)} "
        f"current_scenario={imag.get('current_scenario', -1)} "
        f"name=\"{_clip(imag.get('current_scenario_name', ''), 60)}\""
    )
    dh = _arr(imag.get("divergence_history", []))
    if dh.size:
        lines.append("  " + _stats_line("divergence_history", dh))
    scenarios = imag.get("scenarios", []) or []
    cur = imag.get("current_scenario", -1)
    if 0 <= cur < len(scenarios):
        sc = scenarios[cur]
        lines.append(
            "current_scenario_detail: "
            f"num_outcomes={sc.get('num_outcomes', 0)} "
            f"divergence_factor={sc.get('divergence_factor', 0.0):.4f} "
            f"creativity_level={sc.get('creativity_level', 0.0):.4f}"
        )
        outs = sorted(
            sc.get("outcomes", []) or [],
            key=lambda o: -o.get("probability", 0.0),
        )
        for i, o in enumerate(outs[:3]):
            lines.append(
                f"  outcome[{i}]: "
                f"probability={o.get('probability', 0.0):.4f} "
                f"confidence={o.get('confidence', 0.0):.4f} "
                f"impact={o.get('impact_score', 0.0):.4f} "
                f"plausibility={o.get('plausibility', 0.0):.4f} "
                f"desc=\"{_clip(o.get('description', ''), DESC_CLIP)}\""
            )
    return lines


def _lines_dynamic(
    dyn: dict, stability: float | None
) -> list[str]:
    lines = ["[dynamic_parameters]"]
    if stability is not None:
        lines.append(f"stability: {stability:.4f}")
    for k, v in dyn.items():
        lines.append(f"{k}: {float(v):.4f}")
    return lines


def _lines_memory(mem: dict) -> list[str]:
    lines = ["[memory]"]
    lines.append(
        f"total: {mem.get('size', 0)}/{mem.get('capacity', 0)}"
    )
    best = None
    for level in ("short_term", "medium_term", "long_term"):
        lvl = mem.get(level, {}) or {}
        entries = lvl.get("entries", []) or []
        imps = [float(e.get("importance", 0.0)) for e in entries]
        line = (
            f"{level}: {lvl.get('size', 0)}/{lvl.get('capacity', 0)}"
            " entries"
        )
        if imps:
            line += (
                f" importance mean={sum(imps) / len(imps):.4f}"
                f" max={max(imps):.4f}"
            )
        lines.append(line)
        for e in entries:
            imp = float(e.get("importance", 0.0))
            if best is None or imp > best[0]:
                best = (imp, level, e)
    if best is not None:
        imp, level, e = best
        lines.append(
            f"strongest_entry: level={level} "
            f"importance={imp:.4f} "
            f"timestamp={e.get('timestamp', 0)}"
        )
        lines.append("  " + _stats_line("vector", _arr(e.get("vector", []))))
    return lines


def _lines_reflection(refl: dict) -> list[str]:
    lines = ["[reflection_parameters]"]
    metrics = refl.get("metrics") or {}
    if metrics:
        lines.append(
            "metrics: "
            f"confidence={metrics.get('confidence_score', 0.5):.4f} "
            f"coherence={metrics.get('coherence_score', 0.5):.4f} "
            f"novelty={metrics.get('novelty_score', 0.0):.4f} "
            f"consistency={metrics.get('consistency_score', 1.0):.4f}"
        )
        lines.append(
            "potentially_confabulated: "
            f"{bool(metrics.get('potentially_confabulated', False))} "
            f"derived_drift: {_reflect_drift(metrics):.4f}"
        )
    state  = refl.get("state") or {}
    params = state.get("params") or {}
    if params:
        body = " ".join(f"{k}={float(v):.4f}" for k, v in params.items())
        lines.append(f"params: {body}")
    hist = state.get("history") or {}
    if hist:
        lines.append(
            "thresholds: "
            f"confidence={hist.get('confidence_threshold', 0.0):.4f} "
            f"coherence={hist.get('coherence_threshold', 0.0):.4f} "
            f"consistency={hist.get('consistency_threshold', 0.0):.4f}"
        )
        lines.append(
            "consistency_baseline: "
            f"{hist.get('consistency_baseline', 0.0):.4f} "
            f"saturation_baseline: "
            f"{hist.get('saturation_baseline', 0.0):.4f}"
        )
        n    = max(0, int(hist.get("history_index", 0)))
        conf = _arr(hist.get("historical_confidence", []))[:n]
        coh  = _arr(hist.get("historical_coherence", []))[:n]
        con  = _arr(hist.get("historical_consistency", []))[:n]
        if n:
            k = min(10, n)
            lines.append(
                f"history({n} entries, means over last {k}): "
                f"confidence={conf[-k:].mean():.4f} "
                f"coherence={coh[-k:].mean():.4f} "
                f"consistency={con[-k:].mean():.4f}"
            )
    return lines


def _lines_snapshot(snap: dict, training: bool) -> list[str]:
    lines: list[str] = []
    if training:
        lines.extend(_lines_run(snap))
        lines.append("")
    lines.extend(_lines_text(snap))
    lines.append("")
    lines.extend(_lines_outputs(snap, training))
    lines.append("")
    lines.extend(_lines_fusion(snap))
    lines.append("")
    lines.extend(_lines_meta(snap.get("meta", {})))
    lines.append("")
    lines.extend(_lines_imagination(snap.get("imagination", {})))
    lines.append("")
    lines.extend(_lines_dynamic(
        snap.get("dynamic", {}), snap.get("stability")
    ))
    lines.append("")
    lines.extend(_lines_memory(snap.get("memory", {})))
    lines.append("")
    lines.extend(_lines_reflection(snap.get("reflection", {})))
    return lines


def build_final_prompt() -> str:
    lines = ["=== LARKOS OUTPUT PROMPT ==="]
    lines.append(f"training_epochs_recorded: {len(_trajectory)}")
    lines.append(f"inference_steps_recorded: {len(_inference_steps)}")
    lines.append("")
    lines.extend(_lines_trajectory())
    lines.append("")
    if _training_final:
        lines.append(
            "--- TRAINING SNAPSHOT "
            f"(final epoch {_training_final['epoch']}) ---"
        )
        lines.extend(_lines_snapshot(_training_final, training=True))
    else:
        lines.append("--- TRAINING SNAPSHOT: none recorded ---")
    lines.append("")
    if _inference_steps:
        for snap in _inference_steps:
            lines.append(f"--- INFERENCE STEP {snap['step']} ---")
            lines.extend(_lines_snapshot(snap, training=False))
            lines.append("")
    else:
        lines.append("--- INFERENCE: none recorded ---")
    lines.append("=== END OUTPUT PROMPT ===")
    return "\n".join(lines)


def write_outputs(
    prompt: str | None = None,
    prompt_path: str = OUTPUT_PROMPT_FILE,
    interpreter_path: str = INTERPRETER_FILE,
) -> tuple[str, str]:
    if prompt is None:
        prompt = build_final_prompt()
    with open(prompt_path, "w") as fh:
        fh.write(prompt + "\n")
    with open(interpreter_path, "w") as fh:
        fh.write(INTERPRETER_PROMPT + "\n")
    return prompt_path, interpreter_path


def finalize() -> str:
    """Builds the final prompt, prints it, and writes both files. Called
    once at the end of run_model so the prompt always pairs the final
    training state with whatever inference steps just ran."""
    prompt = build_final_prompt()
    print(prompt)
    prompt_path, interpreter_path = write_outputs(prompt)
    print(f"  output prompt written to : {prompt_path}")
    print(f"  interpreter written to   : {interpreter_path}")
    return prompt
