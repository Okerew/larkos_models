import math

from modules.config import (
    DYN_MIN_EPOCHS, DYN_SOFT_MAX_EPOCHS, DYN_MAX_EPOCHS,
    DYN_PLATEAU_REL_DELTA, DYN_PLATEAU_PATIENCE,
    DYN_TARGET_LOSS, DYN_TARGET_REFRESH, DYN_GOOD_PATIENCE,
    DYN_STAGN_TOLERANCE, DYN_STAGN_PATIENCE,
    DYN_STABILITY_FLOOR, DYN_DRIFT_CEILING, DYN_LR_EXHAUSTED,
)


class EpochController:
    """
    Replaces the fixed epoch count: counts epochs dynamically and
    decides when a TrainingLoop run should finalize.

    The loop trains on a rotating pool of prompts, so the raw per-epoch
    loss is a mixture of per-prompt losses and cannot be compared
    epoch-to-epoch directly. All plateau / convergence statistics are
    therefore aligned to "cycles" of pool_size epochs one full pass
    over the prompt pool. Comparing cycle means cancels the per-prompt
    differences, and the target-freeze sawtooth (refresh spike, then
    in-window decay) averages out inside a cycle as well.

    Stop paths, in priority order:

      hard cap   - epoch >= max_epochs, unconditional, so a run can
                   never go on forever
      converged  - cycle-mean loss and the last refresh-epoch base
                   loss both under their targets for good_patience
                   consecutive cycle checks ("parameters are good
                   enough")
      plateau    - cycle-mean relative improvement below
                   plateau_rel_delta for plateau_patience consecutive
                   cycle checks while the scheduler lr is spent. Past
                   soft_max_epochs the patience and delta relax and
                   the lr condition drops, so a long flat run still
                   terminates instead of grinding to the cap.
      stagnation - no cycle beat the best cycle mean seen so far by
                   more than stagn_tolerance for stagn_patience
                   consecutive cycles (lr spent or past soft_max).
                   Catches the "bouncing around the loss floor" tail
                   where prev-cycle comparisons keep wiggling above
                   the flat delta but nothing actually improves.

    The soft minimum (min_epochs) blocks every stop path before it so
    training never stops immediately. Checks taken while the C backend
    reports low stability or high reflection drift are held (counters
    frozen, not reset) so a churning backend can neither trigger a
    stop nor break a genuine streak.
    """

    def __init__(
        self,
        pool_size:           int,
        min_epochs:          int   = DYN_MIN_EPOCHS,
        soft_max_epochs:     int   = DYN_SOFT_MAX_EPOCHS,
        max_epochs:          int   = DYN_MAX_EPOCHS,
        plateau_rel_delta:   float = DYN_PLATEAU_REL_DELTA,
        plateau_patience:    int   = DYN_PLATEAU_PATIENCE,
        target_loss:         float = DYN_TARGET_LOSS,
        target_refresh_loss: float = DYN_TARGET_REFRESH,
        good_patience:       int   = DYN_GOOD_PATIENCE,
        stagn_tolerance:     float = DYN_STAGN_TOLERANCE,
        stagn_patience:      int   = DYN_STAGN_PATIENCE,
        stability_floor:     float = DYN_STABILITY_FLOOR,
        drift_ceiling:       float = DYN_DRIFT_CEILING,
        lr_exhausted:        float = DYN_LR_EXHAUSTED,
    ) -> None:
        self.pool_size           = max(int(pool_size), 1)
        self.min_epochs          = min_epochs
        self.soft_max_epochs     = soft_max_epochs
        self.max_epochs          = max_epochs
        self.plateau_rel_delta   = plateau_rel_delta
        self.plateau_patience    = plateau_patience
        self.target_loss         = target_loss
        self.target_refresh_loss = target_refresh_loss
        self.good_patience       = good_patience
        self.stagn_tolerance     = stagn_tolerance
        self.stagn_patience      = stagn_patience
        self.stability_floor     = stability_floor
        self.drift_ceiling       = drift_ceiling
        self.lr_exhausted        = lr_exhausted

        self._cycle_losses:   list[float] = []
        self._cycle_means:    list[float] = []
        self._refresh_losses: list[float] = []
        self._best_cycle_mean: float | None = None
        self._flat_checks  = 0
        self._good_checks  = 0
        self._stagn_checks = 0
        self.stop_reason: str | None = None

    @property
    def cycles_completed(self) -> int:
        return len(self._cycle_means)

    def describe(self) -> str:
        return (
            f"min={self.min_epochs} soft_max={self.soft_max_epochs} "
            f"hard_cap={self.max_epochs} cycle={self.pool_size} "
            f"target_loss={self.target_loss} "
            f"plateau={self.plateau_rel_delta}x{self.plateau_patience}"
        )

    def observe(
        self,
        *,
        epoch:        int,
        loss:         float,
        lr:           float,
        is_refresh:   bool,
        refresh_loss: float | None,
        stability:    float,
        drift:        float,
    ) -> bool:
        """
        Feed one finished epoch. Returns True when training should
        finalize; stop_reason then says why. Call once per epoch after
        the gradient step and side-effects have completed.
        """
        if math.isfinite(loss):
            self._cycle_losses.append(loss)
        if (
            is_refresh
            and refresh_loss is not None
            and math.isfinite(refresh_loss)
        ):
            self._refresh_losses.append(refresh_loss)

        # Hard cap first: the one unconditional stop.
        if epoch >= self.max_epochs:
            self.stop_reason = (
                f"hard cap at {self.max_epochs} epochs"
            )
            return True
        # Soft minimum: no stop path is allowed to fire before this.
        if epoch < self.min_epochs:
            return False
        # Decisions only on cycle boundaries so the statistics compare
        # like for like across the whole prompt pool.
        if len(self._cycle_losses) < self.pool_size:
            return False

        cycle_mean = sum(self._cycle_losses) / len(self._cycle_losses)
        self._cycle_losses.clear()
        prev_mean = self._cycle_means[-1] if self._cycle_means else None
        self._cycle_means.append(cycle_mean)

        settled = (
            stability >= self.stability_floor
            and drift <= self.drift_ceiling
        )

        # Past the soft limit the plateau rules relax: patience drops
        # to a single check, the delta doubles, and the lr condition
        # falls away so a long flat run terminates instead of grinding
        # all the way to the hard cap.
        rel_delta = self.plateau_rel_delta
        patience  = self.plateau_patience
        past_soft_max = epoch >= self.soft_max_epochs
        if past_soft_max:
            rel_delta *= 2.0
            patience = 1

        last_refresh = (
            self._refresh_losses[-1] if self._refresh_losses else None
        )
        good = (
            cycle_mean <= self.target_loss
            and last_refresh is not None
            and last_refresh <= self.target_refresh_loss
        )

        lr_spent = lr <= self.lr_exhausted

        # Stagnation is measured against the best cycle ever seen, not
        # the previous one: late training bounces around the loss floor
        # and prev-cycle comparisons keep wiggling above the flat delta
        # while nothing actually improves any more.
        improved_on_best = (
            self._best_cycle_mean is None
            or cycle_mean
            < self._best_cycle_mean * (1.0 - self.stagn_tolerance)
        )
        if self._best_cycle_mean is None:
            self._best_cycle_mean = cycle_mean
        else:
            self._best_cycle_mean = min(
                self._best_cycle_mean, cycle_mean
            )
        stagnant = (
            not improved_on_best
            and (lr_spent or past_soft_max)
        )

        rel_impr: float | None = None
        flat = False
        if prev_mean is not None:
            rel_impr = (
                (prev_mean - cycle_mean)
                / max(abs(prev_mean), 1e-8)
            )
            flat = (
                rel_impr < rel_delta
                and (lr_spent or past_soft_max)
            )

        if not settled:
            # Backend churning: hold all counters. A noisy check
            # should neither start a stop nor break a genuine streak.
            pass
        else:
            if improved_on_best:
                self._stagn_checks = 0
            elif stagnant:
                self._stagn_checks += 1
            if good:
                self._good_checks += 1
                self._flat_checks = 0
            elif flat:
                self._flat_checks += 1
                self._good_checks = 0
            else:
                self._good_checks = 0
                self._flat_checks = 0

        impr_str = (
            f"{rel_impr:+.4f}" if rel_impr is not None else "n/a"
        )
        refresh_str = (
            f"{last_refresh:.4f}"
            if last_refresh is not None else "n/a"
        )
        held = "" if settled else "  (held: backend unsettled)"
        print(
            f"  [epoch {epoch}] epoch controller - "
            f"cycle_mean={cycle_mean:.4f} "
            f"rel_impr={impr_str} "
            f"refresh={refresh_str} "
            f"good={self._good_checks}/{self.good_patience} "
            f"flat={self._flat_checks}/{patience} "
            f"stagn={self._stagn_checks}/{self.stagn_patience}{held}"
        )

        if self._good_checks >= self.good_patience:
            self.stop_reason = (
                f"converged - cycle_mean {cycle_mean:.4f} <= "
                f"{self.target_loss} and refresh loss "
                f"{last_refresh:.4f} <= {self.target_refresh_loss} "
                f"for {self._good_checks} cycles"
            )
            return True
        if self._flat_checks >= patience:
            self.stop_reason = (
                f"plateau - relative cycle-mean improvement below "
                f"{rel_delta:.4f} for {self._flat_checks} cycles"
            )
            return True
        if self._stagn_checks >= self.stagn_patience:
            self.stop_reason = (
                f"stagnation - no cycle beat the best cycle mean "
                f"{self._best_cycle_mean:.4f} by more than "
                f"{self.stagn_tolerance} for {self._stagn_checks} "
                f"cycles"
            )
            return True
        return False
