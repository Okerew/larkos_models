import torch
import torch.nn as nn
from dataclasses import dataclass, field

from modules.config import (
    DEVICE, MAX_NEURONS, INPUT_SIZE,
    FOURIER_OUT_DIM, TEMPORAL_WINDOW,
    GRAD_FLOOR, ACT_UPPER,
    OVERFIT_STEPS, OVERFIT_THRESH,
    LOSS_SPIKE_RATIO, LOSS_PLATEAU_DELTA,
    LOSS_PLATEAU_WINDOW, LOSS_DIVERGE_FLOOR,
    LR_SENSITIVITY_DELTA, GRAD_SMOOTH_ALPHA,
    DEAD_WINDOW, DETACHED_LABELS,
)


@dataclass
class VerifyReport:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warns:  list[str] = field(default_factory=list)

    def ok(self, tag: str) -> None:
        self.passed.append(tag)

    def fail(self, tag: str, reason: str) -> None:
        self.failed.append(f"{tag}: {reason}")

    def warn(self, tag: str, reason: str) -> None:
        self.warns.append(f"{tag}: {reason}")

    def summary(self) -> str:
        lines = [
            f"[verify] {len(self.passed)} passed  "
            f"{len(self.failed)} failed  "
            f"{len(self.warns)} warnings",
        ]
        for f in self.failed:
            lines.append(f"  FAIL  {f}")
        for w in self.warns:
            lines.append(f"  WARN  {w}")
        return "\n".join(lines)

    @property
    def all_ok(self) -> bool:
        return len(self.failed) == 0


def check_shapes(
    r:          VerifyReport,
    x_temporal: torch.Tensor,
    x_norm:     torch.Tensor,
    x_fourier:  torch.Tensor,
    model_pred: torch.Tensor,
    fused:      torch.Tensor,
    target:     torch.Tensor,
) -> None:
    expected_temporal = FOURIER_OUT_DIM * TEMPORAL_WINDOW

    _shape(r, "x_temporal",
           x_temporal, (expected_temporal,))
    _shape(r, "x_norm",
           x_norm,     (INPUT_SIZE,))
    _shape(r, "x_fourier",
           x_fourier,  (FOURIER_OUT_DIM,))
    _shape(r, "model_pred",
           model_pred, (1, MAX_NEURONS))
    _shape(r, "fused",
           fused,      (1, MAX_NEURONS))
    _shape(r, "target",
           target,     (1, MAX_NEURONS))


def _shape(
    r:        VerifyReport,
    name:     str,
    tensor:   torch.Tensor,
    expected: tuple,
) -> None:
    if tuple(tensor.shape) == expected:
        r.ok(f"shape:{name}")
    else:
        r.fail(
            f"shape:{name}",
            f"got {tuple(tensor.shape)} expected {expected}",
        )


def check_gradients(
    r:     VerifyReport,
    loss:  torch.Tensor,
    named: dict[str, nn.Module],
) -> None:
    """
    named maps a human label -> nn.Module so we can walk each
    module's parameters and confirm grad norms are healthy after
    a .backward() call on loss (which must still have grad_fn).
    Detects dead paths (zero / None grad) and explosions separately.
    """
    if loss.grad_fn is None:
        r.fail("grad:loss", "loss has no grad_fn — graph is dead")
        return
    r.ok("grad:loss_has_graph")

    loss.backward(retain_graph=True)

    for label, module in named.items():
        params = list(module.parameters())
        if not params:
            r.warn(f"grad:{label}", "no parameters")
            continue

        norms     = []
        missing   = 0
        trainable = 0
        for p in params:
            if not p.requires_grad:
                continue
            trainable += 1
            if p.grad is None:
                missing += 1
            else:
                norms.append(p.grad.norm().item())

        if trainable == 0:
            continue
        if missing == trainable:
            r.fail(
                f"grad:{label}",
                "all grads are None — path is cut",
            )
        elif missing > 0:
            r.warn(
                f"grad:{label}",
                f"{missing}/{trainable} trainable params have no grad",
            )

        if norms:
            total = sum(norms)
            if total < GRAD_FLOOR:
                r.warn(
                    f"grad:{label}",
                    f"grad norm={total:.2e} — likely vanishing",
                )
            elif total > 1e3:
                r.warn(
                    f"grad:{label}",
                    f"grad norm={total:.2e} — likely exploding",
                )
            else:
                r.ok(f"grad:{label}")


def check_fourier_reach(
    r:          VerifyReport,
    x_fourier:  torch.Tensor,
    x_temporal: torch.Tensor,
) -> None:
    """
    Verifies the Fourier encoding is non-trivial and that it
    actually made it into x_temporal intact — a zero slice here
    means the padding logic buried the signal.
    """
    if x_fourier.abs().max().item() < 1e-6:
        r.fail("fourier:output", "all-zero — encoding did nothing")
        return
    r.ok("fourier:output")

    # x_temporal is TEMPORAL_WINDOW copies cat'd along last dim;
    # the most recent slice lives at the end
    last_slice = x_temporal[-FOURIER_OUT_DIM:]
    if not torch.allclose(
        last_slice.cpu(), x_fourier.detach().cpu(), atol=1e-5
    ):
        r.fail(
            "fourier:temporal_reach",
            "latest Fourier slice not found at end of x_temporal",
        )
    else:
        r.ok("fourier:temporal_reach")


def check_activations(
    r:       VerifyReport,
    tensors: dict[str, torch.Tensor],
) -> None:
    for name, t in tensors.items():
        mx = t.detach().abs().max().item()
        if not torch.isfinite(t).all():
            r.fail(f"act:{name}", "contains NaN or Inf")
        elif mx > ACT_UPPER:
            r.warn(f"act:{name}", f"max abs={mx:.2e} — very large")
        elif mx < 1e-8:
            r.warn(f"act:{name}", f"max abs={mx:.2e} — near zero")
        else:
            r.ok(f"act:{name}")


def check_attention_mask(
    r:       VerifyReport,
    mask:    torch.Tensor | None,
    seq_len: int,
) -> None:
    if mask is None:
        r.warn("attn:mask", "mask is None — not checked")
        return

    if mask.shape[-1] != seq_len:
        r.fail(
            "attn:mask",
            f"mask width {mask.shape[-1]} != seq_len {seq_len}",
        )
        return

    # Every row should have at least one unmasked position
    if (mask.sum(dim=-1) == 0).any():
        r.fail("attn:mask", "some rows are fully masked")
    else:
        r.ok("attn:mask")


def check_overfit(
    r:         VerifyReport,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x_fixed:   torch.Tensor,
    y_fixed:   torch.Tensor,
    embed_ctx: str,
    steps:     int = OVERFIT_STEPS,
) -> None:
    """
    Clones the model to avoid mutating training state, then checks
    that a single fixed (x, y) pair can be memorised in a small
    number of steps. Persistent failure here means the architecture
    cannot learn at all on this input size / shape combination.
    """
    import copy
    m_clone = copy.deepcopy(model).to(DEVICE)
    m_clone.train()
    opt_clone = torch.optim.Adam(m_clone.parameters(), lr=1e-3)

    prev = None
    for _ in range(steps):
        opt_clone.zero_grad()
        pred = m_clone(x_fixed, embed_ctx)
        loss = criterion(pred, y_fixed)
        loss.backward()
        opt_clone.step()
        prev = loss.item()

    if prev is None or prev > OVERFIT_THRESH:
        r.fail(
            "overfit",
            f"loss={prev:.4f} after {steps} steps"
            f" (threshold {OVERFIT_THRESH})",
        )
    else:
        r.ok(f"overfit:loss={prev:.4f}")



def check_fusion_transformer(
    r:                  VerifyReport,
    fusion_transformer: nn.Module,
    fused_cog:          torch.Tensor,
) -> None:
    """
    Runs a single forward + backward through the fusion transformer
    in isolation to confirm the transformer path is fully connected.
    """
    inp   = fused_cog.detach().unsqueeze(0).requires_grad_(True)
    out   = fusion_transformer(inp)
    dummy = out.sum()
    dummy.backward()

    if inp.grad is None or inp.grad.norm().item() < GRAD_FLOOR:
        r.fail(
            "fusion_transformer:grad",
            "no gradient returned to input — path broken",
        )
    else:
        r.ok("fusion_transformer:grad")

    if not torch.isfinite(out).all():
        r.fail("fusion_transformer:output", "NaN/Inf in output")
    else:
        r.ok("fusion_transformer:output")


def check_loss_health(
    r:            VerifyReport,
    loss_val:     float,
    loss_history: list[float],
) -> None:
    """
    Inspects the scalar loss in the context of its own history.
    Catches spikes, plateaus, and outright divergence — conditions
    that are invisible when you only look at the current value.
    """
    if loss_val != loss_val or loss_val == float("inf"):
        r.fail("loss:value", "NaN or Inf loss this epoch")
        return
    r.ok("loss:finite")

    if loss_val > LOSS_DIVERGE_FLOOR:
        r.fail(
            "loss:divergence",
            f"loss={loss_val:.2f} exceeds divergence floor"
            f" {LOSS_DIVERGE_FLOOR}",
        )

    if len(loss_history) < 2:
        return

    mean_h = sum(loss_history) / len(loss_history)

    if mean_h > 0 and loss_val > mean_h * LOSS_SPIKE_RATIO:
        r.warn(
            "loss:spike",
            f"loss={loss_val:.4f} is {loss_val/mean_h:.1f}x"
            f" the running mean={mean_h:.4f}",
        )
    else:
        r.ok("loss:no_spike")

    if len(loss_history) >= LOSS_PLATEAU_WINDOW:
        recent  = loss_history[-LOSS_PLATEAU_WINDOW:]
        spread  = max(recent) - min(recent)
        if spread < LOSS_PLATEAU_DELTA:
            r.warn(
                "loss:plateau",
                f"loss unchanged (spread={spread:.2e}) over"
                f" last {LOSS_PLATEAU_WINDOW} epochs"
                f" — lr or architecture may need attention",
            )
        else:
            r.ok("loss:not_plateaued")

    # Simple trend: is loss going down over the last few epochs?
    if len(loss_history) >= 5:
        tail    = loss_history[-5:]
        # Linear regression slope via least squares on [0..4]
        n       = len(tail)
        xs      = list(range(n))
        xm      = sum(xs) / n
        ym      = sum(tail) / n
        num     = sum((xs[i] - xm) * (tail[i] - ym) for i in range(n))
        den     = sum((xs[i] - xm) ** 2 for i in range(n))
        slope   = num / (den + 1e-12)
        if slope > 0.0:
            r.warn(
                "loss:trend",
                f"loss is trending UP (slope={slope:.4f})"
                f" over the last 5 epochs",
            )
        else:
            r.ok(f"loss:trend_down(slope={slope:.4f})")


class LearningPatternTracker:
    """
    Stateful tracker that lives on TrainingLoop across epochs.
    Accumulates per-epoch signals and exposes check_learning_patterns
    which fires the pattern analysis into a VerifyReport.

    Tracks:
      - smoothed per-module grad norm history (EMA)
      - loss improvement rate
      - lr sensitivity (did a lr change actually shift the loss?)
      - gradient variance across modules (homogeneity probe)
      - dead-module detection (grad norm stuck at zero N epochs)
    """

    DEAD_WINDOW = DEAD_WINDOW

    def __init__(self) -> None:
        self.grad_ema:      dict[str, float]       = {}
        self.grad_history:  dict[str, list[float]] = {}
        self.loss_deltas:   list[float]            = []
        self.lr_history:    list[float]            = []
        self.loss_at_lr:    list[tuple[float, float]] = []

    def update(
        self,
        named_modules: dict[str, nn.Module],
        loss_val:      float,
        lr:            float,
    ) -> None:
        for label, module in named_modules.items():
            params = [
                p for p in module.parameters()
                if p.grad is not None
            ]
            norm = (
                sum(p.grad.norm().item() for p in params)
                if params else 0.0
            )
            prev_ema = self.grad_ema.get(label, norm)
            self.grad_ema[label] = (
                (1.0 - GRAD_SMOOTH_ALPHA) * prev_ema
                + GRAD_SMOOTH_ALPHA * norm
            )
            self.grad_history.setdefault(label, []).append(
                self.grad_ema[label]
            )

        if self.loss_at_lr:
            prev_loss = self.loss_at_lr[-1][1]
            self.loss_deltas.append(loss_val - prev_loss)
        self.loss_at_lr.append((lr, loss_val))
        self.lr_history.append(lr)

    def check_learning_patterns(
        self, r: VerifyReport
    ) -> None:
        self._check_dead_modules(r)
        self._check_grad_homogeneity(r)
        self._check_lr_sensitivity(r)
        self._check_improvement_rate(r)
        self._check_grad_oscillation(r)

    def _check_dead_modules(self, r: VerifyReport) -> None:
        for label, hist in self.grad_history.items():
            if len(hist) < self.DEAD_WINDOW:
                continue
            tail = hist[-self.DEAD_WINDOW:]
            if all(v < GRAD_FLOOR for v in tail):
                r.fail(
                    f"pattern:dead_module:{label}",
                    f"grad norm < {GRAD_FLOOR:.0e} for"
                    f" {self.DEAD_WINDOW} consecutive"
                    f" verification points",
                )
            else:
                r.ok(f"pattern:alive:{label}")

    def _check_grad_homogeneity(self, r: VerifyReport) -> None:
        """
        If one module's grad norm is orders of magnitude larger than
        all others the optimiser is effectively only training that
        branch — surfaces imbalanced loss contributions.

        Detached-path modules (see DETACHED_LABELS) are excluded:
        they live behind the cognitive_fuse C boundary and are fed
        only by aux_loss, so a large gap to the in-graph transformer
        is expected by design, not a sign of imbalance. Comparing
        them here produced a permanent false-alarm warning.
        """
        in_graph = {
            k: v for k, v in self.grad_ema.items()
            if k not in DETACHED_LABELS
        }
        if not in_graph:
            return
        vals  = list(in_graph.values())
        mx    = max(vals)
        mn    = min(v for v in vals if v > 0) if any(
            v > 0 for v in vals
        ) else 1.0

        ratio = mx / (mn + 1e-12)
        if ratio > 1e4:
            loudest = max(in_graph, key=in_graph.get)
            quietest = min(
                in_graph, key=lambda k: in_graph[k]
            )
            r.warn(
                "pattern:grad_imbalance",
                f"grad norm ratio={ratio:.1e}"
                f" ({loudest} vs {quietest})"
                f" — one branch dominates",
            )
        else:
            r.ok(f"pattern:grad_balance(ratio={ratio:.1f})")

    def _check_lr_sensitivity(self, r: VerifyReport) -> None:
        """
        Checks whether a learning rate change actually produced a
        measurable loss response. Insensitivity usually means the
        optimiser is stuck in a flat region or lr is too small.
        """
        if len(self.lr_history) < 3:
            return

        for i in range(1, len(self.lr_history)):
            if abs(
                self.lr_history[i] - self.lr_history[i - 1]
            ) < 1e-9:
                continue
            # lr changed at step i — check loss delta that followed
            if i >= len(self.loss_at_lr):
                break
            delta = abs(
                self.loss_at_lr[i][1] - self.loss_at_lr[i - 1][1]
            )
            if delta < LR_SENSITIVITY_DELTA:
                r.warn(
                    "pattern:lr_insensitive",
                    f"lr changed but loss moved only {delta:.4f}"
                    f" — model may be stuck",
                )
            else:
                r.ok(
                    f"pattern:lr_responsive"
                    f"(delta={delta:.4f})"
                )
            break

    def _check_improvement_rate(self, r: VerifyReport) -> None:
        """
        Measures whether the overall direction of learning is
        positive. Uses the sign-ratio of recent loss deltas:
        if more than 70 % are positive (loss going up) we warn.
        """
        if len(self.loss_deltas) < 5:
            return
        recent   = self.loss_deltas[-10:]
        positive = sum(1 for d in recent if d > 0)
        ratio    = positive / len(recent)
        if ratio > 0.7:
            r.warn(
                "pattern:degrading",
                f"{positive}/{len(recent)} recent steps"
                f" increased loss — model may be regressing",
            )
        elif ratio < 0.2:
            r.ok(
                f"pattern:improving"
                f"(down_frac={1-ratio:.2f})"
            )
        else:
            r.ok("pattern:mixed_progress")

    def _check_grad_oscillation(self, r: VerifyReport) -> None:
        """
        Detects high-frequency oscillation in grad norms — a sign
        of an lr that is too large or loss landscape instability.
        Computed as the mean of absolute successive differences
        normalised by the mean norm (coefficient of variation proxy).
        """
        for label, hist in self.grad_history.items():
            if len(hist) < 6:
                continue
            tail    = hist[-6:]
            diffs   = [
                abs(tail[i] - tail[i - 1])
                for i in range(1, len(tail))
            ]
            mean_d  = sum(diffs) / len(diffs)
            mean_v  = sum(tail) / len(tail)
            cv      = mean_d / (mean_v + 1e-12)
            if cv > 1.0:
                r.warn(
                    f"pattern:grad_oscillation:{label}",
                    f"cv={cv:.2f} — grad norms"
                    f" are highly unstable",
                )


def run_verification(
    epoch:              int,
    model:              nn.Module,
    fusion_transformer: nn.Module,
    embed_weight_net:   nn.Module,
    cross_attn:         nn.Module,
    text_proj:          nn.Module,
    optimizer:          torch.optim.Optimizer,
    criterion:          nn.Module,
    x_temporal:         torch.Tensor,
    x_norm:             torch.Tensor,
    x_fourier:          torch.Tensor,
    model_pred:         torch.Tensor,
    fused:              torch.Tensor,
    fused_cog:          torch.Tensor,
    target:             torch.Tensor,
    loss:               torch.Tensor,
    loss_val:           float,
    loss_history:       list[float],
    lr:                 float,
    embed_ctx:          str,
    pattern_tracker:    LearningPatternTracker,
    attn_mask:          torch.Tensor | None = None,
) -> VerifyReport:
    r = VerifyReport()

    named_modules = {
        "model":              model,
        "fusion_transformer": fusion_transformer,
        "embed_weight_net":   embed_weight_net,
        "cross_attn":         cross_attn,
        "text_proj":          text_proj,
    }

    check_shapes(
        r, x_temporal, x_norm, x_fourier,
        model_pred, fused, target,
    )

    check_fourier_reach(r, x_fourier, x_temporal)

    check_activations(r, {
        "x_temporal":  x_temporal,
        "model_pred":  model_pred,
        "fused":       fused,
        "fused_cog":   fused_cog,
    })

    if attn_mask is not None:
        check_attention_mask(
            r, attn_mask, x_temporal.shape[-1]
        )

    check_gradients(r, loss, named_modules)

    check_fusion_transformer(r, fusion_transformer, fused_cog)

    check_loss_health(r, loss_val, loss_history)

    # Update the tracker after grads exist (check_gradients called
    # backward already) so grad norms are populated on params
    pattern_tracker.update(named_modules, loss_val, lr)
    pattern_tracker.check_learning_patterns(r)

    # Overfit probe only on early epochs — expensive to run every step
    if epoch <= 3:
        x_fixed = x_temporal.unsqueeze(0).detach()
        y_fixed = target.detach()
        check_overfit(
            r, model, optimizer, criterion,
            x_fixed, y_fixed, embed_ctx,
        )

    print(r.summary())
    return r
