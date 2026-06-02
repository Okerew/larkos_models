import copy
import math
import numpy as np
import torch
import torch.nn as nn

from modules.config import (
    BASE_LR, MAX_NEURONS,
    MAML_INNER_LR, MAML_INNER_STEPS,
)


def build_neuron_prediction(neurons: dict) -> np.ndarray:
    outputs = []
    for i in range(MAX_NEURONS):
        key = f"neuron_{i}"
        n   = neurons.get(key, {})
        val = float(n.get("output", 0.0))
        if math.isnan(val) or math.isinf(val):
            val = 0.0
        outputs.append(val)
    return np.array(outputs, dtype=np.float32)


def _normalise_neuron(
    nt: torch.Tensor,
) -> torch.Tensor:
    # Bring neuron outputs to a comparable scale before mixing.
    # They collapse toward zero when weights are freshly initialised
    # so we rescale by the max abs value if it's meaningful
    nt_max = nt.abs().max()
    return nt / nt_max if nt_max > 1e-4 else nt


def fuse_predictions(
    model_pred:  torch.Tensor,
    neuron_pred: np.ndarray,
    alpha:       float = 0.5,
) -> torch.Tensor:
    nt = torch.tensor(
        neuron_pred, dtype=torch.float32
    ).to(model_pred.device)

    # Tanh squashes model outputs smoothly so gradients survive
    # everywhere — clamp() created dead zones for saturated units.
    model_tanh    = torch.tanh(model_pred)
    nt_norm       = _normalise_neuron(nt)
    out = alpha * model_tanh + (1.0 - alpha) * nt_norm

    # Final clamp + nan guard so a diverging MAML clone can't
    # smuggle explosions into the outer loss
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.clamp(out, -1.0, 1.0)


def fuse_uncertainty_weighted(
    model_pred:  torch.Tensor,
    neuron_pred: np.ndarray,
    mc_samples:  list[torch.Tensor],
) -> torch.Tensor:
    """
    Per-neuron uncertainty fusion using MC-dropout variance.
    Lower model variance -> trust model more for that neuron.
    mc_samples should be T forward passes with dropout active.

    When variance is near zero across all outputs (e.g. early
    training with no dropout layers) this gracefully falls back
    to equal weighting between model and neuron signals.

    Scale factor on sigmoid is kept low (2.0 instead of 10.0)
    so near-zero variance doesn't hard-saturate alpha_per to 0.5
    and lose per-neuron differentiation entirely.
    """
    stack = torch.stack(mc_samples, dim=0)
    var   = stack.var(dim=0).clamp(min=1e-8)

    nt = torch.tensor(
        neuron_pred, dtype=torch.float32
    ).to(model_pred.device)

    nt_norm       = _normalise_neuron(nt)
    model_tanh    = torch.tanh(model_pred)

    # alpha_per_neuron: high variance -> trust neuron more
    alpha_per = torch.sigmoid(-var * 2.0)
    out = alpha_per * model_tanh + (1.0 - alpha_per) * nt_norm

    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.clamp(out, -1.0, 1.0)


def derive_lr(meta: dict) -> float:
    mc = meta.get("metacognition", {})
    ms = meta.get("learning_state", {})

    confidence  = mc.get("confidence_level",  0.5)
    cog_load    = mc.get("cognitive_load",     0.5)
    error_aware = mc.get("error_awareness",    0.5)
    adapt_rate  = mc.get("adaptation_rate",    0.5)

    raw_stability   = ms.get("stability_index",    None)
    raw_efficiency  = ms.get("learning_efficiency", None)
    raw_exploration = ms.get("exploration_rate",    0.5)

    # nan guards, ctypes backend returns nan for these when not yet computed
    stability   = 0.5 if (
        raw_stability  is None or not np.isfinite(raw_stability)
    ) else raw_stability
    efficiency  = 1.0 if (
        raw_efficiency is None or not np.isfinite(raw_efficiency)
    ) else raw_efficiency
    exploration = 0.5 if not np.isfinite(raw_exploration) \
        else raw_exploration

    # Higher confidence + stability -> smaller, safer steps
    # Higher error awareness + exploration -> larger corrective steps
    # Cognitive load dampens the rate to avoid thrashing
    lr  = BASE_LR
    lr *= (1.0 + error_aware * exploration)
    lr *= (1.0 - 0.5 * cog_load)
    lr *= max(adapt_rate, 1e-6) * max(efficiency, 1e-6)

    # Clamp the divisor to at least 0.25 so early-training defaults
    # (confidence=0.5, stability=0.5 -> product=0.25) can't push LR
    # above 4x BASE_LR; previously the product could approach 1e-6
    # and spike LR to 0.04-0.1 on a small network
    lr /= max(confidence * stability, 0.25)

    result = float(np.clip(lr, 1e-5, 2e-3))
    if not np.isfinite(result):
        return BASE_LR
    return result


def derive_alpha_from_context(ctx: dict) -> float:
    vec = ctx.get("global_context_vector", [])
    if not vec:
        return 0.5
    arr = np.array(vec, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5
    mean_ctx = float(np.mean(np.abs(arr)))
    return float(np.clip(0.3 + 0.4 * mean_ctx, 0.3, 0.7))


def derive_alpha_from_params(
    params: dict, fallback: float
) -> float:
    lr_scale = params.get("learning_rate_scale", None)
    if lr_scale is None:
        return fallback
    return float(np.clip(lr_scale * fallback, 0.3, 0.7))


def update_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr:        float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maml_inner_update(
    model:     nn.Module,
    x:         torch.Tensor,
    target:    torch.Tensor,
    criterion: nn.Module,
    embed_ctx: str | None = None,
) -> nn.Module:
    """
    MAML-lite inner loop - runs a few fast-adaptation gradient
    steps on a cloned model without touching the outer weights.
    Returns the adapted clone so the caller can compute the
    outer loss against it; the original model is untouched.

    We skip this when embed_ctx is None - the embedding path
    is the main signal that benefits from fast adaptation and
    running it on raw numerics alone gives negligible gain.
    """
    fast = copy.deepcopy(model)
    fast.train()
    inner_opt = torch.optim.SGD(
        fast.parameters(), lr=MAML_INNER_LR
    )
    for _ in range(MAML_INNER_STEPS):
        inner_opt.zero_grad()
        pred = fast(x, embed_ctx)
        loss = criterion(pred, target)
        loss.backward()
        inner_opt.step()
    return fast
