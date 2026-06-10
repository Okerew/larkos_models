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
from typing import List, Optional


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
    # Physics simulation fields (Tests 10 / 11)
    physics_state:      np.ndarray = None  # s_t   observation vector
    physics_next_state: np.ndarray = None  # s_{t+1} observation vector
    # Entity bond interaction fields (Test 12)
    entity_id:      int   = -1
    entity_name:    str   = ""
    entity_reward:  float = 0.0
    bond_snapshots: dict  = field(default_factory=dict)  # entity_id -> bond metrics


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

    # intra_run_spread is computed only over REFRESH epochs in the last
    # quarter.  Inside a freeze window the transformer input and target
    # are pinned, so consecutive fused vectors are nearly identical and
    # would artificially collapse the spread - the metric we actually
    # care about is whether the model's internal representations form
    # non-trivial structure when they ARE allowed to vary.
    late_refresh = [
        r for r in records[n - q:]
        if getattr(r, "input_refreshed", False)
    ]
    fused_sims = []
    for i in range(len(late_refresh)):
        for j in range(i + 1, len(late_refresh)):
            s = _cos(late_refresh[i].fused, late_refresh[j].fused)
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
    Probes whether the model builds a coherent internal world model.

    Three assertions are required, none of which a frozen random projection
    can satisfy:

    1. Grounding (RSA): similar inputs produce similar fused representations.
       Measured via Representational Similarity Analysis - the Pearson r
       between pairwise input cosine similarities and pairwise fused cosine
       similarities.  A frozen random projection scores ~0 here (it
       projects to random dimensions, destroying relative distances).

    2. Learning trajectory: the grounding improves from the first half to the
       second half of training.  A fixed projection's RSA is constant.

    3. Cluster grounding: when the model forms tight clusters in fused space,
       those clusters must be meaningfully input-grounded.  Measured as the
       ratio of mean input similarity inside the top-20%-fused-similar pairs
       to the mean input similarity across all pairs.  A ratio > 1.0 means
       the model's implicit groupings are not arbitrary - it has discovered
       a rule for what belongs together even without explicit supervision.

    Only target-refresh epochs are used: inside a freeze window the
    transformer is fed the cached fused-cog target, not the live input, so
    input_vec and fused are decoupled.  Refresh epochs carry genuinely paired
    (input, fused) observations.

    Returns:
      rsa_early          - RSA(input↔fused) on first-half refresh epochs
      rsa_late           - RSA(input↔fused) on second-half refresh epochs
      rsa_improvement    - rsa_late - rsa_early  (positive = model learned grounding)
      cluster_grounding  - mean input sim of top-20% fused-similar pairs /
                           mean input sim of all pairs  (>1 = clusters are meaningful;
                           random baseline ≈ 1.0)
      temporal_structure - mean consecutive-fused cosine sim /
                           mean randomly-paired-fused cosine sim
                           (>1 = temporal continuity beyond random)
      n_grounded_pairs   - number of (input, fused) pairs used for late RSA
    """
    n = len(records)
    if n == 0:
        return {}

    refresh = [
        r for r in records
        if getattr(r, "input_refreshed", False)
        and getattr(r, "input_vec",      None) is not None
    ]

    def _cos(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            return float(np.dot(a, b) / (na * nb))
        return None

    def _rsa(recs: list) -> tuple:
        """RSA between input_vec and fused over a set of records. Returns (r, n_pairs)."""
        in_sims: list[float] = []
        fu_sims: list[float] = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                in_s = _cos(np.ravel(recs[i].input_vec), np.ravel(recs[j].input_vec))
                fu_s = _cos(recs[i].fused, recs[j].fused)
                if in_s is not None and fu_s is not None:
                    in_sims.append(in_s)
                    fu_sims.append(fu_s)
        if len(in_sims) < 3:
            return 0.0, len(in_sims)
        ai, af = np.array(in_sims), np.array(fu_sims)
        if np.std(ai) < 1e-8 or np.std(af) < 1e-8:
            return 0.0, len(in_sims)
        return float(np.corrcoef(ai, af)[0, 1]), len(in_sims)

    half = len(refresh) // 2
    rsa_early, _       = _rsa(refresh[:half])
    rsa_late,  n_pairs = _rsa(refresh[half:])
    rsa_improvement    = rsa_late - rsa_early

    # Cluster grounding: do fused-space clusters correspond to input-space structure?
    # Collect all pairwise (fused_sim, input_sim) for late refresh epochs, then
    # compare the mean input sim of the top-20% fused-similar pairs against the
    # mean input sim of all pairs.  Ratio > 1 means clusters are input-grounded.
    cluster_grounding = 1.0
    late_recs = refresh[half:]
    if len(late_recs) >= 3:
        all_pairs: list[tuple] = []
        for i in range(len(late_recs)):
            for j in range(i + 1, len(late_recs)):
                in_s = _cos(np.ravel(late_recs[i].input_vec), np.ravel(late_recs[j].input_vec))
                fu_s = _cos(late_recs[i].fused, late_recs[j].fused)
                if in_s is not None and fu_s is not None:
                    all_pairs.append((fu_s, in_s))
        if len(all_pairs) >= 5:
            mean_in_all = float(np.mean([p[1] for p in all_pairs]))
            all_pairs.sort(key=lambda p: p[0], reverse=True)
            top_n = max(1, len(all_pairs) // 5)
            mean_in_top = float(np.mean([p[1] for p in all_pairs[:top_n]]))
            if abs(mean_in_all) > 1e-8:
                cluster_grounding = mean_in_top / (mean_in_all + 1e-8)

    # Temporal structure: consecutive fused vectors should be more similar
    # to each other than randomly paired vectors, detecting representation
    # continuity across time steps.
    temporal_structure = 1.0
    all_fuseds = [r.fused for r in records]
    if len(all_fuseds) >= 10:
        consec_sims = []
        for i in range(1, len(all_fuseds)):
            s = _cos(all_fuseds[i - 1], all_fuseds[i])
            if s is not None:
                consec_sims.append(s)
        rng = np.random.RandomState(0)
        idx = rng.choice(len(all_fuseds), (min(len(all_fuseds) * 2, 200), 2))
        shuffled_sims = []
        for a, b in idx:
            if a != b:
                s = _cos(all_fuseds[a], all_fuseds[b])
                if s is not None:
                    shuffled_sims.append(s)
        if consec_sims and shuffled_sims:
            mean_shuffled = float(np.mean(shuffled_sims))
            if abs(mean_shuffled) > 1e-8:
                temporal_structure = float(np.mean(consec_sims)) / (mean_shuffled + 1e-8)

    return {
        "rsa_early":          rsa_early,
        "rsa_late":           rsa_late,
        "rsa_improvement":    rsa_improvement,
        "cluster_grounding":  cluster_grounding,
        "temporal_structure": temporal_structure,
        "n_grounded_pairs":   n_pairs,
    }


def adaptation_score(
    pre_change_records:  List[EpochRecord],
    post_change_records: List[EpochRecord],
    recovery_tolerance:  float = 1.5,
) -> dict:
    """
    How quickly the model adapts after a rule change.

    recovery_tolerance multiplies pre_final_loss to define the band the
    post-change loss has to re-enter to count as "recovered". The 1.1x
    default it used to carry was unrealistic for fundamental rule changes
    (e.g. gravity reversal) where the optimal new loss is not the same as
    the old optimal loss; 1.5x still requires the model to have re-built a
    competent representation, just without demanding it match the prior
    convergence point.

    Returns:
      pre_final_loss          - loss after training in original env
      post_initial_loss       - loss at start of adaptation (likely high)
      post_final_loss
      adaptation_slope        - convergence rate post-change
      disruption_magnitude    - post_initial minus pre_final (size of the shock)
      recovery_epochs         - epochs until post loss returns to within
                                recovery_tolerance x pre_final_loss
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

    # Epochs to return to within recovery_tolerance x pre-change loss level
    recovery_epochs = -1
    recovery_target = pre_f * recovery_tolerance
    for i, r in enumerate(post_change_records):
        if r.loss <= recovery_target:
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


def physics_world_model_score(records: List[EpochRecord]) -> dict:
    """
    Tests whether the model builds an internal model of live physics dynamics.

    Uses Representational Similarity Analysis (RSA): if the model has a world
    model, states that are physically similar should also be similar in fused
    space.  We measure the Pearson correlation between pairwise cosine
    similarities in physics state space and pairwise cosine similarities in
    fused space, on the first half vs last half of target-refresh epochs.

    Only target-refresh epochs are used: on frozen epochs the transformer
    receives the cached fused-cog target rather than the live physics text,
    so fused and physics_state are decoupled.  Refresh epochs carry genuinely
    paired (physics_state, fused) observations.

    Also measures whether consecutive fused changes are directionally aligned
    with consecutive physics state changes (state_change_alignment).

    Returns:
      rsa_early            - RSA correlation on first half of refresh epochs
      rsa_late             - RSA correlation on last half of refresh epochs
      rsa_improvement      - rsa_late - rsa_early (positive = model improved)
      state_change_alignment - mean cosine similarity between normalised
                               consecutive-physics-state deltas and
                               normalised consecutive-fused deltas
                               (dimension-capped to min of both)
      state_change_magnitude - mean L2 norm of physics state changes
      fused_change_magnitude - mean L2 norm of fused changes
      n_physics_pairs        - number of refresh records used for RSA
    """
    phys_records = [
        r for r in records
        if r.physics_state is not None and getattr(r, "input_refreshed", False)
    ]
    n = len(phys_records)
    if n < 4:
        return {"rsa_early": 0.0, "rsa_late": 0.0, "rsa_improvement": 0.0,
                "state_change_alignment": 0.0, "state_change_magnitude": 0.0,
                "fused_change_magnitude": 0.0, "n_physics_pairs": 0}

    def _cos_sim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            return float(np.dot(a, b) / (na * nb))
        return None

    def _rsa(phys_vecs, fused_vecs, max_pairs: int = 100) -> float:
        """Pearson r between pairwise cosine similarities in two spaces."""
        n_ = len(phys_vecs)
        pairs = [(i, j) for i in range(n_) for j in range(i + 1, n_)]
        if len(pairs) > max_pairs:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[k] for k in idx]
        ps, fs = [], []
        for i, j in pairs:
            p = _cos_sim(phys_vecs[i], phys_vecs[j])
            f = _cos_sim(fused_vecs[i], fused_vecs[j])
            if p is not None and f is not None:
                ps.append(p)
                fs.append(f)
        if len(ps) < 3:
            return 0.0
        ap, af = np.array(ps), np.array(fs)
        if np.std(ap) < 1e-8 or np.std(af) < 1e-8:
            return 0.0
        return float(np.corrcoef(ap, af)[0, 1])

    half  = max(1, n // 2)
    early = phys_records[:half]
    late  = phys_records[half:]

    rsa_early = _rsa(
        [r.physics_state for r in early],
        [r.fused         for r in early],
    )
    rsa_late = _rsa(
        [r.physics_state for r in late],
        [r.fused         for r in late],
    )

    # Consecutive-change alignment
    alignments: list[float] = []
    state_mags: list[float] = []
    fused_mags: list[float] = []
    for i in range(1, n):
        ds = phys_records[i].physics_state - phys_records[i - 1].physics_state
        df = phys_records[i].fused         - phys_records[i - 1].fused
        dim = min(len(ds), len(df))
        ds_, df_ = ds[:dim], df[:dim]
        s = _cos_sim(ds_, df_)
        if s is not None:
            alignments.append(s)
        state_mags.append(float(np.linalg.norm(ds)))
        fused_mags.append(float(np.linalg.norm(df)))

    return {
        "rsa_early":              rsa_early,
        "rsa_late":               rsa_late,
        "rsa_improvement":        rsa_late - rsa_early,
        "state_change_alignment": float(np.mean(alignments)) if alignments else 0.0,
        "state_change_magnitude": float(np.mean(state_mags)) if state_mags else 0.0,
        "fused_change_magnitude": float(np.mean(fused_mags)) if fused_mags else 0.0,
        "n_physics_pairs":        len(phys_records),
    }


def physics_adaptation_score(
    pre_records:  List[EpochRecord],
    post_records: List[EpochRecord],
) -> dict:
    """
    Measures adaptation after a live physics rule change (e.g., gravity flip).

    Combines loss-based adaptation metrics with RSA-based world-model
    re-alignment: after the rule changes, the model's internal representations
    should re-organise to track the new dynamics.

    Returns:
      pre_final_loss
      post_initial_loss
      post_final_loss
      disruption_magnitude   - post_initial - pre_final
      adaptation_slope       - loss slope post-change
      recovery_epochs
      rsa_pre                - RSA correlation for pre-change final quarter
      rsa_post               - RSA correlation for post-change final quarter
      rsa_recovery           - rsa_post - rsa_pre (positive = re-aligned)
    """
    base = adaptation_score(pre_records, post_records)

    rsa_pre  = physics_world_model_score(pre_records).get("rsa_late",  0.0)
    rsa_post = physics_world_model_score(post_records).get("rsa_late", 0.0)

    base["rsa_pre"]      = rsa_pre
    base["rsa_post"]     = rsa_post
    base["rsa_recovery"] = rsa_post - rsa_pre
    return base


def affective_bonds_score(records: List[EpochRecord]) -> dict:
    """
    Tests whether the affective system's emotional response correctly tracks
    entity interaction quality by measuring per-entity DELTAS.

    Absolute valence/arousal are dominated and saturated by the training-loop's
    loss-driven emotion triggers.  The entity-specific signal is instead isolated
    as the change (delta) in emotional state caused by each entity's interaction
    pass, measured immediately before and after the entity-specific trigger call.

    Expected deltas (second half of training):
      consistent_positive  → love_delta > 0, hate_delta ≈ 0, valence_delta > 0
      consistent_negative  → hate_delta > 0, love_delta ≈ 0, valence_delta < 0
      erratic              → surp_delta or mixed love/hate delta, high variability
      neutral_steady       → surp_delta > 0, valence_delta ≈ 0

    Returns:
      bond_count                 - distinct entities with delta snapshots
      per_entity_love_delta      - {name: mean LOVE intensity delta per entity}
      per_entity_hate_delta      - {name: mean HATE intensity delta}
      per_entity_surp_delta      - {name: mean SURPRISE intensity delta}
      per_entity_valence_delta   - {name: mean valence delta}
      per_entity_reward          - {name: mean interaction reward}
      targeting_accuracy         - fraction of epochs where the dominant emotion
                                   delta matches what the entity's reward dictates
                                   (reward>0 → love_delta > hate_delta, etc.)
      positive_over_negative     - 1 if consistent_positive valence_delta >
                                   consistent_negative valence_delta
      valence_delta_reward_corr  - Pearson r(entity_reward, mean_valence_delta)
      valence_delta_std          - std of per-entity mean valence_delta
    """
    bond_records = [r for r in records if r.entity_id >= 0 and r.bond_snapshots]
    if not bond_records:
        return {
            "bond_count":               0,
            "targeting_accuracy":       0.0,
            "positive_over_negative":   0,
            "valence_delta_reward_corr":0.0,
            "valence_delta_std":        0.0,
        }

    # Second half only - the affective system needs time to respond
    half = len(bond_records) // 2
    late = bond_records[half:] if half > 0 else bond_records

    per_love_d:    dict[str, list[float]] = {}
    per_hate_d:    dict[str, list[float]] = {}
    per_surp_d:    dict[str, list[float]] = {}
    per_val_d:     dict[str, list[float]] = {}
    per_reward:    dict[str, list[float]] = {}
    epoch_correct: list[int]              = []

    for rec in late:
        eid  = rec.entity_id
        name = rec.entity_name or str(eid)
        snap = rec.bond_snapshots.get(eid, {})

        love_d = float(snap.get("love_delta",    0.0))
        hate_d = float(snap.get("hate_delta",    0.0))
        surp_d = float(snap.get("surp_delta",    0.0))
        val_d  = float(snap.get("valence_delta", 0.0))
        reward = float(rec.entity_reward)

        per_love_d.setdefault(name, []).append(love_d)
        per_hate_d.setdefault(name, []).append(hate_d)
        per_surp_d.setdefault(name, []).append(surp_d)
        per_val_d.setdefault(name,  []).append(val_d)
        per_reward.setdefault(name, []).append(reward)

        # Per-epoch targeting: did the dominant emotion delta match the reward?
        if reward > 0.1:
            epoch_correct.append(int(love_d >= hate_d))
        elif reward < -0.1:
            epoch_correct.append(int(hate_d >= love_d))
        else:
            epoch_correct.append(int(abs(surp_d) >= abs(love_d) and abs(surp_d) >= abs(hate_d)))

    mean_love_d = {n: float(np.mean(v)) for n, v in per_love_d.items()}
    mean_hate_d = {n: float(np.mean(v)) for n, v in per_hate_d.items()}
    mean_surp_d = {n: float(np.mean(v)) for n, v in per_surp_d.items()}
    mean_val_d  = {n: float(np.mean(v)) for n, v in per_val_d.items()}
    mean_reward = {n: float(np.mean(v)) for n, v in per_reward.items()}

    targeting_accuracy = float(np.mean(epoch_correct)) if epoch_correct else 0.0

    # positive_over_negative: consistent_positive valence_delta > consistent_negative
    pos_vd = mean_val_d.get("consistent_positive")
    neg_vd = mean_val_d.get("consistent_negative")
    pos_over_neg = int(
        pos_vd is not None and neg_vd is not None and pos_vd > neg_vd
    )

    # Pearson r between per-entity mean reward and per-entity mean valence_delta
    names       = list(mean_val_d.keys())
    vd_list     = [mean_val_d[n] for n in names]
    reward_list = [mean_reward.get(n, 0.0) for n in names]
    corr = 0.0
    if len(names) >= 3:
        va = np.array(vd_list)
        ra = np.array(reward_list)
        if np.std(va) > 1e-8 and np.std(ra) > 1e-8:
            corr = float(np.corrcoef(va, ra)[0, 1])

    vd_std = float(np.std(vd_list)) if len(vd_list) > 1 else 0.0

    return {
        "bond_count":                len(mean_val_d),
        "per_entity_love_delta":     mean_love_d,
        "per_entity_hate_delta":     mean_hate_d,
        "per_entity_surp_delta":     mean_surp_d,
        "per_entity_valence_delta":  mean_val_d,
        "per_entity_reward":         mean_reward,
        "targeting_accuracy":        targeting_accuracy,
        "positive_over_negative":    pos_over_neg,
        "valence_delta_reward_corr": corr,
        "valence_delta_std":         vd_std,
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
