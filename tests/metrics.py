"""
Metric collection and analysis for the Larkos testing framework.

Metrics are gathered per epoch inside TestTrainingLoop._side_effects
and later analysed per test.  All analysis functions are pure; they
take lists of EpochRecord and return a flat dict of scalar scores.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from typing import List


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    fused: np.ndarray               # shape (MAX_NEURONS,) - fused output
    input_vec: np.ndarray = None
    input_refreshed: bool = False
    valence: float = 0.0
    arousal: float = 0.0
    stability: float = 0.0
    emotion_intensities: list = field(default_factory=list)
    cognitive_impact: float = 0.0
    emotional_regulation: float = 0.0
    mask_intensity: float = 0.0
    specialization_effectiveness: float = 0.0
    identity_confidence: float = 0.0
    identity_consistency: float = 0.0


def learning_efficiency_score(records: List[EpochRecord]) -> dict:
    """
    Measures how quickly the model learns a new domain.

    Returns:
      initial_loss         - loss at epoch 1
      final_loss           - loss at final epoch
      total_improvement    - initial minus final
      epochs_to_half       - epochs until loss halved from initial, -1 if never
      mean_loss_slope      - linear regression slope of loss curve (negative = learning)
      loss_cv              - coefficient of variation (stability of convergence)
      fused_drift          - mean cosine distance between consecutive fused vectors
                             (high means the internal state is still reorganising)
      fused_convergence    - cosine similarity between last two fused vectors
                             (high means the internal world model has settled)
    """
    losses = [r.loss for r in records]
    fuseds = [r.fused for r in records]
    n = len(losses)

    if n == 0:
        return {}

    initial = losses[0]
    final   = losses[-1]
    improvement = initial - final

    # Epochs until loss drops to half of initial
    half_target = initial * 0.5
    epochs_to_half = -1
    for i, l in enumerate(losses):
        if l <= half_target:
            epochs_to_half = i + 1
            break

    # Linear regression slope
    xs = list(range(n))
    xm = n / 2.0
    ym = sum(losses) / n
    num = sum((xs[i] - xm) * (losses[i] - ym) for i in range(n))
    den = sum((xs[i] - xm) ** 2 for i in range(n)) + 1e-12
    slope = num / den

    # Coefficient of variation
    std_ = float(np.std(losses)) if n > 1 else 0.0
    mean_ = float(np.mean(losses)) + 1e-12
    cv = std_ / mean_

    # Fused-vector drift: mean cosine distance between consecutive pairs
    dists = []
    for i in range(1, len(fuseds)):
        a, b = fuseds[i - 1], fuseds[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            cos_sim = float(np.dot(a, b) / (na * nb))
            dists.append(1.0 - cos_sim)
    fused_drift = float(np.mean(dists)) if dists else 0.0

    # Convergence: similarity of last two fused vectors
    fused_convergence = 0.0
    if len(fuseds) >= 2:
        a, b = fuseds[-2], fuseds[-1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            fused_convergence = float(np.dot(a, b) / (na * nb))

    return {
        "initial_loss":          initial,
        "final_loss":            final,
        "total_improvement":     improvement,
        "epochs_to_half":        epochs_to_half,
        "mean_loss_slope":       slope,
        "loss_cv":               cv,
        "fused_drift":           fused_drift,
        "fused_convergence":     fused_convergence,
    }


def transfer_score(
    domain_a_records: List[EpochRecord],
    domain_b_records: List[EpochRecord],
) -> dict:
    """
    Compares convergence speed on domain B after training on domain A
    versus a baseline of training on domain B cold (approximated as
    the early loss trajectory).

    Returns:
      domain_a_final_loss
      domain_b_initial_loss
      domain_b_final_loss
      domain_b_slope              - convergence rate on B
      fused_cosine_a_to_b_start  - cosine similarity between end-of-A fused
                                   and start-of-B fused (state carry-over)
      fused_cosine_a_to_b_end    - end-of-A vs end-of-B (final distance)
      transfer_efficiency        - 1 if B converged faster than A did cold
                                   (estimated by comparing slope magnitudes)
    """
    a_losses = [r.loss for r in domain_a_records]
    b_losses = [r.loss for r in domain_b_records]

    def _slope(losses):
        n = len(losses)
        if n < 2:
            return 0.0
        xs = list(range(n))
        xm = n / 2.0
        ym = sum(losses) / n
        num = sum((xs[i] - xm) * (losses[i] - ym) for i in range(n))
        den = sum((xs[i] - xm) ** 2 for i in range(n)) + 1e-12
        return num / den

    a_slope = _slope(a_losses)
    b_slope = _slope(b_losses)

    # Fused carry-over: end of A vs start / end of B
    cos_start, cos_end = 0.0, 0.0
    if domain_a_records and domain_b_records:
        fa_end   = domain_a_records[-1].fused
        fb_start = domain_b_records[0].fused
        fb_end   = domain_b_records[-1].fused
        def _cos(x, y):
            nx, ny = np.linalg.norm(x), np.linalg.norm(y)
            return float(np.dot(x, y) / (nx * ny)) if nx > 1e-8 and ny > 1e-8 else 0.0
        cos_start = _cos(fa_end, fb_start)
        cos_end   = _cos(fa_end, fb_end)

    # Transfer efficiency: negative slope on B that is steeper than A means
    # faster convergence -> transfer helped.
    transfer_efficiency = 0.0
    if a_slope < 0 and b_slope < a_slope:
        transfer_efficiency = 1.0
    elif b_slope < 0:
        transfer_efficiency = abs(b_slope) / (abs(a_slope) + 1e-8)

    return {
        "domain_a_final_loss":       a_losses[-1] if a_losses else None,
        "domain_b_initial_loss":     b_losses[0]  if b_losses else None,
        "domain_b_final_loss":       b_losses[-1] if b_losses else None,
        "domain_a_slope":            a_slope,
        "domain_b_slope":            b_slope,
        "fused_cosine_a_to_b_start": cos_start,
        "fused_cosine_a_to_b_end":   cos_end,
        "transfer_efficiency":       transfer_efficiency,
    }


def retention_score(
    phase_a_records:      List[EpochRecord],
    phase_b_records:      List[EpochRecord],
    phase_c_records:      List[EpochRecord],
    return_to_a_records:  List[EpochRecord],
) -> dict:
    """
    Measures how much the model retains domain A after learning B then C.

    Retention is high if the loss when returning to A is close to the
    original final-A loss rather than the original initial-A loss.

    Returns:
      phase_a_initial_loss
      phase_a_final_loss
      return_to_a_initial_loss   - first loss when re-exposed to A
      return_to_a_final_loss
      forgetting_index           - (return_initial - a_final) / (a_initial - a_final + eps)
                                   0 = perfect retention, 1 = total forgetting
      recovery_slope             - how fast A is re-learned (negative = fast)
      fused_drift_a_to_return    - cosine distance end-of-C vs start-of-return
    """
    a_i = phase_a_records[0].loss  if phase_a_records else 0.0
    a_f = phase_a_records[-1].loss if phase_a_records else 0.0
    r_i = return_to_a_records[0].loss  if return_to_a_records else a_f
    r_f = return_to_a_records[-1].loss if return_to_a_records else a_f

    forgetting = (r_i - a_f) / (a_i - a_f + 1e-8)
    forgetting = float(np.clip(forgetting, 0.0, 1.0))

    def _slope(losses):
        n = len(losses)
        if n < 2:
            return 0.0
        xs = list(range(n))
        xm = n / 2.0
        ym = sum(losses) / n
        num = sum((xs[i] - xm) * (losses[i] - ym) for i in range(n))
        den = sum((xs[i] - xm) ** 2 for i in range(n)) + 1e-12
        return num / den

    recovery_slope = _slope([r.loss for r in return_to_a_records])

    # Fused drift from end of C to start of return-to-A
    fused_drift = 0.0
    if phase_c_records and return_to_a_records:
        fc = phase_c_records[-1].fused
        fr = return_to_a_records[0].fused
        nc, nr = np.linalg.norm(fc), np.linalg.norm(fr)
        if nc > 1e-8 and nr > 1e-8:
            fused_drift = 1.0 - float(np.dot(fc, fr) / (nc * nr))

    return {
        "phase_a_initial_loss":     a_i,
        "phase_a_final_loss":       a_f,
        "return_to_a_initial_loss": r_i,
        "return_to_a_final_loss":   r_f,
        "forgetting_index":         forgetting,
        "recovery_slope":           recovery_slope,
        "fused_drift_c_to_return":  fused_drift,
    }


def discovery_score(records: List[EpochRecord]) -> dict:
    """
    Measures emergence of internal structure from minimal information.

    Higher fused variance on its own is a weak signal - it rises just as
    much from instability or random drift as from genuine structure.  The
    real questions are geometric: is the late-run similarity structure
    non-trivial rather than collapsed or uniformly noisy (intra_run_spread),
    and do similar inputs map to similar fused representations
    (input_repr_alignment).

    The alignment is computed ONLY over target-refresh epochs in the second
    half of the run.  Inside a freeze window the transformer input is the
    cached fused-cog vector, not the live backend input, so fused there is
    decoupled from input_vec and would drag the correlation toward noise.
    On refresh epochs cognitive_fuse takes a fresh reading driven by the
    live input, so input and representation are genuinely paired.

    Returns:
      fused_variance_early   - mean element-wise variance of first-quarter fused
      fused_variance_late    - mean element-wise variance of last-quarter fused
      variance_ratio         - late / early (kept for reference, not gated)
      intra_run_spread       - std of pairwise cosine sims in the last quarter
                               (near 0 = collapsed/uniform, larger = clustering)
      input_repr_alignment   - Pearson r between pairwise input similarities
                               and pairwise fused similarities over refresh
                               epochs in the second half.  Drift decorrelates
                               the two; real structure keeps them aligned.
      alignment_pairs        - number of input/fused pairs the alignment used
                               (low values mean the score is underpowered)
      loss_final             - final loss on the minimal domain
    """
    n = len(records)
    if n == 0:
        return {}

    q = max(1, n // 4)
    early = np.stack([r.fused for r in records[:q]])
    late  = np.stack([r.fused for r in records[n - q:]])

    var_early = float(np.mean(np.var(early, axis=0)))
    var_late  = float(np.mean(np.var(late,  axis=0)))
    ratio     = var_late / (var_early + 1e-8)

    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            return float(np.dot(a, b) / (na * nb))
        return None

    fused_sims = []
    for i in range(len(late)):
        for j in range(i + 1, len(late)):
            s = _cos(late[i], late[j])
            if s is not None:
                fused_sims.append(s)
    spread = float(np.std(fused_sims)) if fused_sims else 0.0

    # Only refresh epochs in the second half carry a live input paired
    # with the fused vector it produced - see the docstring for why the
    # freeze window makes the other epochs unusable for this.
    half = n // 2
    probe = [
        r for r in records[half:]
        if getattr(r, "input_refreshed", False)
        and getattr(r, "input_vec", None) is not None
    ]

    paired_input_sims = []
    paired_fused_sims = []
    for i in range(len(probe)):
        for j in range(i + 1, len(probe)):
            in_sim = _cos(
                np.ravel(probe[i].input_vec),
                np.ravel(probe[j].input_vec),
            )
            fu_sim = _cos(probe[i].fused, probe[j].fused)
            if in_sim is not None and fu_sim is not None:
                paired_input_sims.append(in_sim)
                paired_fused_sims.append(fu_sim)

    alignment = 0.0
    if len(paired_input_sims) >= 3:
        ai = np.array(paired_input_sims)
        af = np.array(paired_fused_sims)
        if np.std(ai) > 1e-8 and np.std(af) > 1e-8:
            alignment = float(np.corrcoef(ai, af)[0, 1])

    return {
        "fused_variance_early": var_early,
        "fused_variance_late":  var_late,
        "variance_ratio":       ratio,
        "intra_run_spread":     spread,
        "input_repr_alignment": alignment,
        "alignment_pairs":      len(paired_input_sims),
        "loss_final":           records[-1].loss,
    }


def stability_score(records: List[EpochRecord]) -> dict:
    """
    Checks that loss, fused vectors, and emotional state remain
    well-behaved over a long run.

    Returns:
      loss_range              - max minus min loss over the run
      loss_late_std           - std of loss in the last quarter
      fused_mean_norm         - average L2 norm of fused vector
      fused_norm_std          - std of those norms (instability proxy)
      valence_range           - max minus min valence
      arousal_range
      stability_range
      nan_epochs              - epochs where loss was NaN or Inf
    """
    n = len(records)
    if n == 0:
        return {}

    losses   = [r.loss for r in records]
    valences = [r.valence for r in records]
    arousals = [r.arousal for r in records]
    stabs    = [r.stability for r in records]
    norms    = [float(np.linalg.norm(r.fused)) for r in records]

    q = max(1, n // 4)
    late_losses = losses[n - q:]

    nan_epochs = sum(1 for l in losses if not math.isfinite(l))

    finite_losses = [l for l in losses if math.isfinite(l)]
    loss_range = (max(finite_losses) - min(finite_losses)) if len(finite_losses) > 1 else 0.0

    return {
        "loss_range":      loss_range,
        "loss_late_std":   float(np.std(late_losses)) if late_losses else 0.0,
        "fused_mean_norm": float(np.mean(norms)),
        "fused_norm_std":  float(np.std(norms)),
        "valence_range":   max(valences) - min(valences) if valences else 0.0,
        "arousal_range":   max(arousals) - min(arousals) if arousals else 0.0,
        "stability_range": max(stabs) - min(stabs) if stabs else 0.0,
        "nan_epochs":      nan_epochs,
    }


def world_model_score(records: List[EpochRecord]) -> dict:
    """
    Probes whether the model builds coherent internal representations.

    We measure the clustering coefficient of fused vectors: if the model
    encodes rules rather than noise, similar inputs should map to similar
    fused vectors even without explicit supervision.

    We also check whether fused-vector magnitude correlates with loss
    (a representation that tracks confidence should be larger when
    loss is lower).

    Returns:
      mean_pairwise_similarity    - mean cosine similarity of all fused pairs
      pairwise_sim_std            - std (low = all vectors similar = collapsed)
      loss_fused_norm_correlation - Pearson r between loss and fused norm
                                    negative means low loss -> large fused (good)
      fused_dimensionality        - fraction of fused dimensions with std > 0.01
                                    (how many dims actually carry information)
    """
    n = len(records)
    if n == 0:
        return {}

    fused_mat = np.stack([r.fused for r in records])  # (N, D)
    losses    = np.array([r.loss for r in records])
    norms     = np.linalg.norm(fused_mat, axis=1)

    # Pairwise cosine similarities - sample up to 200 pairs to keep it fast
    fuseds = fused_mat
    pairs = []
    for i in range(min(n, 20)):
        for j in range(i + 1, min(n, 20)):
            pairs.append((i, j))

    sims = []
    for i, j in pairs:
        a, b = fuseds[i], fuseds[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            sims.append(float(np.dot(a, b) / (na * nb)))

    mean_sim = float(np.mean(sims)) if sims else 0.0
    std_sim  = float(np.std(sims))  if sims else 0.0

    # Correlation between loss and fused norm
    corr = 0.0
    if len(losses) > 2 and np.std(norms) > 1e-8 and np.std(losses) > 1e-8:
        corr = float(np.corrcoef(losses, norms)[0, 1])

    # Dimensionality: fraction of dims with meaningful variation
    dim_stds = np.std(fused_mat, axis=0)
    active_dims = float(np.mean(dim_stds > 0.01))

    return {
        "mean_pairwise_similarity":    mean_sim,
        "pairwise_sim_std":            std_sim,
        "loss_fused_norm_correlation": corr,
        "fused_dimensionality":        active_dims,
    }


def adaptation_score(
    pre_change_records:  List[EpochRecord],
    post_change_records: List[EpochRecord],
) -> dict:
    """
    How quickly the model adapts after a rule change.

    Returns:
      pre_final_loss          - loss after training in original env
      post_initial_loss       - loss at start of adaptation (likely high)
      post_final_loss
      adaptation_slope        - convergence rate post-change
      disruption_magnitude    - post_initial minus pre_final (size of the shock)
      recovery_epochs         - epochs until post loss returns to pre_final level
      fused_shift             - cosine distance between pre-final and post-final fused
    """
    pre_f  = pre_change_records[-1].loss  if pre_change_records else 0.0
    post_i = post_change_records[0].loss  if post_change_records else pre_f
    post_f = post_change_records[-1].loss if post_change_records else pre_f

    disruption = post_i - pre_f

    def _slope(losses):
        n = len(losses)
        if n < 2:
            return 0.0
        xs = list(range(n))
        xm = n / 2.0
        ym = sum(losses) / n
        num = sum((xs[i] - xm) * (losses[i] - ym) for i in range(n))
        den = sum((xs[i] - xm) ** 2 for i in range(n)) + 1e-12
        return num / den

    adaptation_slope = _slope([r.loss for r in post_change_records])

    # Epochs to return to pre-change loss level
    recovery_epochs = -1
    for i, r in enumerate(post_change_records):
        if r.loss <= pre_f * 1.1:  # within 10% of pre-change loss
            recovery_epochs = i + 1
            break

    fused_shift = 0.0
    if pre_change_records and post_change_records:
        fa = pre_change_records[-1].fused
        fb = post_change_records[-1].fused
        na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
        if na > 1e-8 and nb > 1e-8:
            fused_shift = 1.0 - float(np.dot(fa, fb) / (na * nb))

    return {
        "pre_final_loss":        pre_f,
        "post_initial_loss":     post_i,
        "post_final_loss":       post_f,
        "adaptation_slope":      adaptation_slope,
        "disruption_magnitude":  disruption,
        "recovery_epochs":       recovery_epochs,
        "fused_shift":           fused_shift,
    }


def meta_learning_score(
    phase_records: List[List[EpochRecord]],
) -> dict:
    """
    Tests whether learning efficiency improves across successive domains.

    phase_records is a list of per-phase record lists (one per domain).

    Returns:
      per_phase_slopes       - list of loss slopes per phase
      slope_trend            - slope of the per-phase slopes (negative = improving)
      per_phase_initial_loss - list of initial losses per phase
      per_phase_final_loss   - list of final losses per phase
    """
    def _slope(losses):
        n = len(losses)
        if n < 2:
            return 0.0
        xs = list(range(n))
        xm = n / 2.0
        ym = sum(losses) / n
        num = sum((xs[i] - xm) * (losses[i] - ym) for i in range(n))
        den = sum((xs[i] - xm) ** 2 for i in range(n)) + 1e-12
        return num / den

    slopes  = []
    initials = []
    finals   = []
    for phase in phase_records:
        losses = [r.loss for r in phase]
        slopes.append(_slope(losses))
        initials.append(losses[0]  if losses else 0.0)
        finals.append(losses[-1] if losses else 0.0)

    # Slope of slopes: is the learning rate improving across phases?
    slope_of_slopes = _slope(slopes) if len(slopes) >= 2 else 0.0

    return {
        "per_phase_slopes":       slopes,
        "slope_trend":            slope_of_slopes,
        "per_phase_initial_loss": initials,
        "per_phase_final_loss":   finals,
    }


def affective_score(records: List[EpochRecord]) -> dict:
    """
    Analyses whether the emotional state encodes more than just reward/loss.

    Checks:
      - Correlation of valence with loss delta (expected: negative)
      - Correlation of arousal with loss magnitude (expected: positive)
      - Stability of emotional regulation over time
      - Whether emotion_intensities have non-trivial variance (not all same)
      - Bond between mask_intensity and identity_confidence

    Returns:
      valence_loss_delta_corr     - Pearson r(valence, loss_delta)
      arousal_loss_corr           - Pearson r(arousal, loss)
      emotional_regulation_trend  - slope of emotional_regulation over time
      emotion_diversity           - mean std across emotion types per epoch
      mask_identity_corr          - Pearson r(mask_intensity, identity_confidence)
      cognitive_impact_range      - max minus min cognitive_impact
      affective_complexity        - mean valence-arousal distance from origin
    """
    n = len(records)
    if n == 0:
        return {}

    losses    = np.array([r.loss for r in records])
    valences  = np.array([r.valence for r in records])
    arousals  = np.array([r.arousal for r in records])
    regs      = np.array([r.emotional_regulation for r in records])
    masks     = np.array([r.mask_intensity for r in records])
    confs     = np.array([r.identity_confidence for r in records])
    cogimps   = np.array([r.cognitive_impact for r in records])

    loss_deltas = np.diff(losses, prepend=losses[0])

    def _corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _slope(arr):
        m = len(arr)
        if m < 2:
            return 0.0
        xs = list(range(m))
        xm = m / 2.0
        ym = float(np.mean(arr))
        num = sum((xs[i] - xm) * (arr[i] - ym) for i in range(m))
        den = sum((xs[i] - xm) ** 2 for i in range(m)) + 1e-12
        return num / den

    # Per-epoch emotion diversity: std across emotion types
    diversities = []
    for r in records:
        ei = r.emotion_intensities
        if len(ei) > 1:
            diversities.append(float(np.std(ei)))
    emotion_diversity = float(np.mean(diversities)) if diversities else 0.0

    # Affective complexity: distance from neutral (0, 0) in VA space
    va_dists = np.sqrt(valences ** 2 + arousals ** 2)
    affective_complexity = float(np.mean(va_dists))

    return {
        "valence_loss_delta_corr":    _corr(valences, loss_deltas),
        "arousal_loss_corr":          _corr(arousals, losses),
        "emotional_regulation_trend": _slope(regs.tolist()),
        "emotion_diversity":          emotion_diversity,
        "mask_identity_corr":         _corr(masks, confs),
        "cognitive_impact_range":     float(np.max(cogimps) - np.min(cogimps)),
        "affective_complexity":       affective_complexity,
    }


def format_metrics(title: str, m: dict) -> str:
    lines = [f"\n  [{title}]"]
    for k, v in m.items():
        if isinstance(v, float):
            lines.append(f"    {k:<40s} {v:+.4f}")
        elif isinstance(v, list):
            lines.append(f"    {k:<40s} {[round(x, 4) if isinstance(x, float) else x for x in v]}")
        else:
            lines.append(f"    {k:<40s} {v}")
    return "\n".join(lines)
