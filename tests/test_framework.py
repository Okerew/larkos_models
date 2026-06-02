"""
Larkos Testing Framework - 9 behavioural tests for the Larkos model.

Each test creates its own BackendState and manages checkpoints
independently.  Tests are self-contained and can be run individually
or via LarkosTestFramework.run_all().

Important constraints (from the testing spec):
  - The model is a state-model, not a pure transformer.  Observations
    must be made in the fused patterns, not in isolated layer outputs.
  - Tests that probe continual learning / adaptation work by loading a
    checkpoint, then retraining - matching the expected usage pattern.
  - Emotional / affective representations are monitored via the C-side
    backend state, captured once per epoch in EpochRecord.

Usage:
    from tests.test_framework import LarkosTestFramework
    fw = LarkosTestFramework()
    results = fw.run_all()
    fw.print_report(results)
"""

from __future__ import annotations

import shutil
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_ROOT = Path(__file__).parent.parent

from modules.config import TEST_DATA, CKPT_DIR

# Test-data domains
_TEST_DATA = _ROOT / TEST_DATA
_CKPT_DIR  = _ROOT / CKPT_DIR
_CKPT_DIR.mkdir(exist_ok=True)

from modules.backend_state import BackendState
from modules.training import TrainingLoop
from modules.checkpoint import save_checkpoint, load_checkpoint


from tests.metrics import (
    EpochRecord,
    learning_efficiency_score,
    transfer_score,
    retention_score,
    discovery_score,
    stability_score,
    world_model_score,
    adaptation_score,
    meta_learning_score,
    affective_score,
    format_metrics,
)


class TestTrainingLoop(TrainingLoop):
    """
    Subclass of TrainingLoop that collects one EpochRecord per epoch
    by hooking into _side_effects.  Everything else (gradient flow,
    MAML, fusion, C-backend triggers) runs unchanged.
    """

    def __init__(
        self,
        backend: BackendState,
        epochs: int = 30,
        initial_alpha: float = 0.5,
        data_dir: Path | None = None,
    ) -> None:
        super().__init__(backend, epochs, initial_alpha, data_dir)
        self.epoch_records: List[EpochRecord] = []

    def _side_effects(
        self,
        epoch, lr, alpha, loss_val,
        fused_np, fused_vec,
        input_tensor, model_pred_np,
        neuron_pred, region_scores,
    ) -> None:
        # Run the standard side-effects first (logging, C backend triggers, etc.)
        super()._side_effects(
            epoch=epoch, lr=lr, alpha=alpha, loss_val=loss_val,
            fused_np=fused_np, fused_vec=fused_vec,
            input_tensor=input_tensor, model_pred_np=model_pred_np,
            neuron_pred=neuron_pred, region_scores=region_scores,
        )

        # Collect per-epoch snapshot for test analysis
        try:
            emo  = self.backend.get_emotional_state()
            aff  = self.backend.get_affective_state()
            mask = self.backend.compute_mask_intensity()

            cur     = aff.get("current_state", {})
            valence = float(cur.get("valence",   0.0))
            arousal = float(cur.get("arousal",   0.0))
            stab    = float(cur.get("stability", 0.0))

            intensities = [
                float(e.get("intensity", 0.0))
                for e in emo.get("emotions", [])
            ]
            cog_impact = float(emo.get("cognitive_impact",    0.0))
            emo_reg    = float(emo.get("emotional_regulation", 0.0))
            mask_int   = float(mask.get("mask_intensity",     0.0))

            id_state = self.backend.get_identity_state()
            id_conf  = float(id_state.get("confidence_level",  0.0))
            id_cons  = float(id_state.get("consistency_score", 0.0))

            spec_eff  = self.backend.evaluate_specialization_effectiveness(
                network_performance=float(1.0 - min(loss_val, 1.0))
            )
            spec_score = float(spec_eff.get("effectiveness", 0.0))

        except Exception:
            valence = arousal = stab = 0.0
            intensities = []
            cog_impact = emo_reg = mask_int = 0.0
            id_conf = id_cons = spec_score = 0.0

        refreshed = (epoch - self._target_epoch) == 0
        try:
            inp_np = np.asarray(input_tensor, dtype=np.float32).ravel()
        except Exception:
            inp_np = None

        self.epoch_records.append(EpochRecord(
            epoch                      = epoch,
            loss                       = loss_val,
            fused                      = fused_np.copy(),
            input_vec                  = inp_np,
            input_refreshed            = refreshed,
            valence                    = valence,
            arousal                    = arousal,
            stability                  = stab,
            emotion_intensities        = intensities,
            cognitive_impact           = cog_impact,
            emotional_regulation       = emo_reg,
            mask_intensity             = mask_int,
            specialization_effectiveness = spec_score,
            identity_confidence        = id_conf,
            identity_consistency       = id_cons,
        ))


def _save_test_checkpoint(loop: TestTrainingLoop, tag: str) -> None:
    """Save torch checkpoint + C-side memory to test_checkpoints/<tag>.*"""
    pt_path  = str(_CKPT_DIR / f"{tag}.pt")
    mem_path = str(_CKPT_DIR / f"{tag}_memory.bin")
    save_checkpoint(loop, pt_path)
    loop.backend.save_memory(mem_path)
    loop.backend.save_network_states()
    net_src = _ROOT / "network_states.json"
    net_dst = _CKPT_DIR / f"{tag}_network_states.json"
    if net_src.exists():
        shutil.copy(net_src, net_dst)


def _load_test_checkpoint(
    loop:    TestTrainingLoop,
    tag:     str,
    use_ema: bool = True,
) -> None:
    """Restore a previously saved test checkpoint onto loop."""
    pt_path  = str(_CKPT_DIR / f"{tag}.pt")
    mem_path = str(_CKPT_DIR / f"{tag}_memory.bin")
    net_path = _CKPT_DIR / f"{tag}_network_states.json"

    load_checkpoint(loop, pt_path, use_ema=use_ema)
    loop.backend.load_memory(mem_path)

    # Restore network states JSON so C-side sees the same neuron history
    if net_path.exists():
        dst = _ROOT / "network_states.json"
        shutil.copy(net_path, dst)
        loop.backend.load_network_states()


@dataclass
class LarkosTestResult:
    test_id:      int
    name:         str
    metrics:      dict
    analysis:     str
    passed:       bool = True
    epoch_records: list = field(default_factory=list)


class LarkosTestFramework:
    """
    Runs the 9 behavioural tests defined in testing_framework_for_larkos.md.

    Parameters
    ----------
    epochs_short : int
        Epochs used for fast phases (learning, transfer, discovery, world model,
        affective).  Default 30.
    epochs_stability : int
        Epochs for the stability test.  Default 60.
    epochs_continual : int
        Epochs per phase in the continual-learning test.  Default 25.
    epochs_adaptation : int
        Epochs per environment in the adaptation test.  Default 25.
    epochs_meta : int
        Epochs per domain in the meta-learning test.  Default 20.
    """

    def __init__(
        self,
        epochs_short:     int = 30,
        epochs_stability: int = 60,
        epochs_continual: int = 25,
        epochs_adaptation:int = 25,
        epochs_meta:      int = 20,
    ) -> None:
        self.epochs_short      = epochs_short
        self.epochs_stability  = epochs_stability
        self.epochs_continual  = epochs_continual
        self.epochs_adaptation = epochs_adaptation
        self.epochs_meta       = epochs_meta

    def _new_loop(
        self,
        data_dir:      Path,
        epochs:        int,
        initial_alpha: float = 0.5,
    ) -> TestTrainingLoop:
        backend = BackendState()
        return TestTrainingLoop(
            backend        = backend,
            epochs         = epochs,
            initial_alpha  = initial_alpha,
            data_dir       = data_dir,
        )

    def _run(self, loop: TestTrainingLoop) -> List[EpochRecord]:
        loop.run()
        return loop.epoch_records

    def run_test_1_learning_efficiency(self) -> LarkosTestResult:
        """
        Train on a completely novel toy-physics domain and measure:
        - how many epochs until competence appears (loss halved)
        - whether loss decreases monotonically
        - whether fused patterns stabilise (converge to a world model)
        """
        print("\n" + "=" * 60)
        print("TEST 1 - Learning Efficiency (toy_physics)")
        print("=" * 60)

        loop    = self._new_loop(_TEST_DATA / "toy_physics", self.epochs_short)
        records = self._run(loop)
        m       = learning_efficiency_score(records)

        analysis_lines = [
            f"Initial loss:       {m.get('initial_loss', '?'):.4f}",
            f"Final loss:         {m.get('final_loss', '?'):.4f}",
            f"Total improvement:  {m.get('total_improvement', '?'):.4f}",
            f"Epochs to 50%:      {m.get('epochs_to_half', '?')}",
            f"Loss slope:         {m.get('mean_loss_slope', '?'):.5f}",
            f"Fused drift:        {m.get('fused_drift', '?'):.4f}",
            f"Fused convergence:  {m.get('fused_convergence', '?'):.4f}",
        ]
        passed = (
            m.get("total_improvement", 0) > 0
            and m.get("mean_loss_slope", 1) < 0
            and m.get("epochs_to_half", -1) != -1
        )
        analysis = "\n".join(analysis_lines)
        print(analysis)
        return LarkosTestResult(1, "Learning Efficiency", m, analysis, passed, records)

    def run_test_2_domain_transfer(self) -> LarkosTestResult:
        """
        Train on logic puzzles (domain A) then on math problems (domain B).
        Measure whether training A accelerates convergence on B.
        """
        print("\n" + "=" * 60)
        print("TEST 2 - Domain Transfer (logic -> math)")
        print("=" * 60)

        # Phase A: logic puzzles
        print("\n  Phase A: logic_puzzles")
        loop_a   = self._new_loop(_TEST_DATA / "logic_puzzles", self.epochs_short)
        records_a = self._run(loop_a)

        # Save checkpoint after A so we can continue from it
        _save_test_checkpoint(loop_a, "test2_phase_a")

        # Phase B: math problems, continuing from the A checkpoint
        print("\n  Phase B: math_problems (continuing from A checkpoint)")
        loop_b = self._new_loop(_TEST_DATA / "math_problems", self.epochs_short)
        _load_test_checkpoint(loop_b, "test2_phase_a", use_ema=True)
        records_b = self._run(loop_b)

        m = transfer_score(records_a, records_b)
        analysis_lines = [
            f"Domain A final loss:     {m.get('domain_a_final_loss', '?'):.4f}",
            f"Domain B initial loss:   {m.get('domain_b_initial_loss', '?'):.4f}",
            f"Domain B final loss:     {m.get('domain_b_final_loss', '?'):.4f}",
            f"A slope:                 {m.get('domain_a_slope', '?'):.5f}",
            f"B slope:                 {m.get('domain_b_slope', '?'):.5f}",
            f"Fused carry-over (start):{m.get('fused_cosine_a_to_b_start', '?'):.4f}",
            f"Transfer efficiency:     {m.get('transfer_efficiency', '?'):.4f}",
        ]
        # Transfer is meaningful if B converges significantly faster than A did cold
        passed = (
            m.get("domain_b_slope", 1) < 0
            and m.get("transfer_efficiency", 0.0) > 0.3
        )
        analysis = "\n".join(analysis_lines)
        print(analysis)
        return LarkosTestResult(2, "Domain Transfer", m, analysis, passed,
                                records_a + records_b)

    def run_test_3_continual_learning(self) -> LarkosTestResult:
        """
        Train A -> B -> C -> return to A.  Measure retention: does the model
        remember A after learning B and C?  Does it re-learn A faster than
        it learned it cold?
        """
        print("\n" + "=" * 60)
        print("TEST 3 - Continual Learning (A→B→C→A)")
        print("=" * 60)

        epochs = self.epochs_continual

        # Phase A
        print("\n  Phase A: toy_physics")
        loop_a    = self._new_loop(_TEST_DATA / "toy_physics", epochs)
        records_a = self._run(loop_a)
        _save_test_checkpoint(loop_a, "test3_phase_a")

        # Phase B
        print("\n  Phase B: logic_puzzles")
        loop_b = self._new_loop(_TEST_DATA / "logic_puzzles", epochs)
        _load_test_checkpoint(loop_b, "test3_phase_a")
        records_b = self._run(loop_b)
        _save_test_checkpoint(loop_b, "test3_phase_b")

        # Phase C
        print("\n  Phase C: abstract_patterns")
        loop_c = self._new_loop(_TEST_DATA / "abstract_patterns", epochs)
        _load_test_checkpoint(loop_c, "test3_phase_b")
        records_c = self._run(loop_c)
        _save_test_checkpoint(loop_c, "test3_phase_c")

        # Return to A
        print("\n  Return to A: toy_physics (testing retention)")
        loop_r = self._new_loop(_TEST_DATA / "toy_physics", epochs)
        _load_test_checkpoint(loop_r, "test3_phase_c")
        records_r = self._run(loop_r)

        m = retention_score(records_a, records_b, records_c, records_r)
        analysis_lines = [
            f"Phase A initial loss:    {m.get('phase_a_initial_loss', '?'):.4f}",
            f"Phase A final loss:      {m.get('phase_a_final_loss', '?'):.4f}",
            f"Return initial loss:     {m.get('return_to_a_initial_loss', '?'):.4f}",
            f"Return final loss:       {m.get('return_to_a_final_loss', '?'):.4f}",
            f"Forgetting index:        {m.get('forgetting_index', '?'):.4f}  (0=perfect, 1=total)",
            f"Recovery slope:          {m.get('recovery_slope', '?'):.5f}",
            f"Fused drift C→return:    {m.get('fused_drift_c_to_return', '?'):.4f}",
        ]
        passed = m.get("forgetting_index", 1.0) < 0.5
        analysis = "\n".join(analysis_lines)
        print(analysis)
        all_records = records_a + records_b + records_c + records_r
        return LarkosTestResult(3, "Continual Learning", m, analysis, passed, all_records)

    def run_test_4_discovery(self) -> LarkosTestResult:
        """
        Train on minimal_info: sparse, ambiguous statements.
        Measure whether internal representations gain structure over time,
        hinting at emergent pattern discovery not explicitly provided.
        """
        print("\n" + "=" * 60)
        print("TEST 4 - Discovery (minimal information)")
        print("=" * 60)

        loop    = self._new_loop(_TEST_DATA / "minimal_info", self.epochs_short)
        records = self._run(loop)
        m       = discovery_score(records)

        analysis_lines = [
            f"Fused variance early:   {m.get('fused_variance_early', '?'):.4f}",
            f"Fused variance late:    {m.get('fused_variance_late', '?'):.4f}",
            f"Variance ratio:         {m.get('variance_ratio', '?'):.4f}  (reference only)",
            f"Intra-run spread:       {m.get('intra_run_spread', '?'):.4f}",
            f"Input-repr alignment:   {m.get('input_repr_alignment', '?'):.4f}  (>0 = structure)",
            f"Alignment pairs:        {m.get('alignment_pairs', '?')}",
            f"Final loss:             {m.get('loss_final', '?'):.4f}",
        ]
        passed = (
            m.get("alignment_pairs", 0) >= 6
            and m.get("input_repr_alignment", 0.0) > 0.2
            and m.get("intra_run_spread", 0.0) > 0.02
        )
        analysis = "\n".join(analysis_lines)
        print(analysis)
        return LarkosTestResult(4, "Discovery", m, analysis, passed, records)

    def run_test_5_model_stability(self) -> LarkosTestResult:
        """
        Run the model on toy_physics for an extended period.
        Check that loss, fused patterns, and emotional state remain
        well-behaved: no NaN, no divergence, stable norms.
        """
        print("\n" + "=" * 60)
        print("TEST 5 - Model Stability (long run)")
        print("=" * 60)

        loop    = self._new_loop(_TEST_DATA / "toy_physics", self.epochs_stability)
        records = self._run(loop)
        m       = stability_score(records)

        analysis_lines = [
            f"Loss range:             {m.get('loss_range', '?'):.4f}",
            f"Late-epoch loss std:    {m.get('loss_late_std', '?'):.4f}",
            f"Fused mean norm:        {m.get('fused_mean_norm', '?'):.4f}",
            f"Fused norm std:         {m.get('fused_norm_std', '?'):.4f}",
            f"Valence range:          {m.get('valence_range', '?'):.4f}",
            f"Arousal range:          {m.get('arousal_range', '?'):.4f}",
            f"NaN epochs:             {m.get('nan_epochs', '?')}",
        ]
        passed = (
            m.get("nan_epochs", 1) == 0
            and m.get("fused_norm_std", 1e9) < 5.0
        )
        analysis = "\n".join(analysis_lines)
        print(analysis)
        return LarkosTestResult(5, "Model Stability", m, analysis, passed, records)

    def run_test_6_internal_world_model(self) -> LarkosTestResult:
        """
        Train on toy_physics and examine whether fused patterns form
        coherent internal representations: meaningful pairwise structure,
        active dimensions, correlation between representation magnitude
        and learning confidence.
        """
        print("\n" + "=" * 60)
        print("TEST 6 - Internal World Model (representation coherence)")
        print("=" * 60)

        loop    = self._new_loop(_TEST_DATA / "toy_physics", self.epochs_short)
        records = self._run(loop)
        m       = world_model_score(records)

        analysis_lines = [
            f"Mean pairwise sim:      {m.get('mean_pairwise_similarity', '?'):.4f}",
            f"Pairwise sim std:       {m.get('pairwise_sim_std', '?'):.4f}",
            f"Loss-fused norm corr:   {m.get('loss_fused_norm_correlation', '?'):.4f}",
            f"Active fused dims:      {m.get('fused_dimensionality', '?'):.2%}",
        ]
        # Model has a world model if representations are neither collapsed
        # (all similar, std≈0) nor totally random (std ≈ mean_sim)
        std_sim = m.get("pairwise_sim_std", 0.0)
        passed = std_sim > 0.01 and m.get("fused_dimensionality", 0.0) > 0.2
        analysis = "\n".join(analysis_lines)
        print(analysis)
        return LarkosTestResult(6, "Internal World Model", m, analysis, passed, records)

    def run_test_7_adaptation_speed(self) -> LarkosTestResult:
        """
        Train in toy_physics (original rules), then change one rule
        (gravity direction flips) and measure how quickly the model
        adapts from the saved checkpoint.
        """
        print("\n" + "=" * 60)
        print("TEST 7 - Adaptation Speed (rule change)")
        print("=" * 60)

        epochs = self.epochs_adaptation

        # Pre-change: original toy_physics
        print("\n  Pre-change: toy_physics")
        loop_pre    = self._new_loop(_TEST_DATA / "toy_physics", epochs)
        records_pre = self._run(loop_pre)
        _save_test_checkpoint(loop_pre, "test7_pre")

        # Post-change: toy_physics_modified (gravity reversed)
        print("\n  Post-change: toy_physics_modified (gravity now normal)")
        loop_post = self._new_loop(_TEST_DATA / "toy_physics_modified", epochs)
        _load_test_checkpoint(loop_post, "test7_pre")
        records_post = self._run(loop_post)

        m = adaptation_score(records_pre, records_post)
        analysis_lines = [
            f"Pre-change final loss:   {m.get('pre_final_loss', '?'):.4f}",
            f"Post-change initial:     {m.get('post_initial_loss', '?'):.4f}",
            f"Post-change final:       {m.get('post_final_loss', '?'):.4f}",
            f"Disruption magnitude:    {m.get('disruption_magnitude', '?'):.4f}",
            f"Adaptation slope:        {m.get('adaptation_slope', '?'):.5f}",
            f"Recovery epochs:         {m.get('recovery_epochs', '?')}",
            f"Fused shift:             {m.get('fused_shift', '?'):.4f}",
        ]
        passed = (
            m.get("adaptation_slope", 1) < 0
            and m.get("recovery_epochs", -1) != -1
        )
        analysis = "\n".join(analysis_lines)
        print(analysis)
        return LarkosTestResult(7, "Adaptation Speed", m, analysis, passed,
                                records_pre + records_post)

    def run_test_8_meta_learning(self) -> LarkosTestResult:
        """
        Train on three unrelated domains sequentially with checkpoint
        continuation.  Measure whether convergence rate improves across
        domains, indicating that learning how to learn is occurring.
        """
        print("\n" + "=" * 60)
        print("TEST 8 - Meta-Learning (efficiency across domains)")
        print("=" * 60)

        epochs  = self.epochs_meta
        domains = [
            ("toy_physics",    "test8_d1"),
            ("logic_puzzles",  "test8_d2"),
            ("math_problems",  "test8_d3"),
        ]

        all_phase_records = []
        prev_tag = None

        for domain_name, tag in domains:
            print(f"\n  Domain: {domain_name}")
            loop = self._new_loop(_TEST_DATA / domain_name, epochs)
            if prev_tag is not None:
                _load_test_checkpoint(loop, prev_tag)
            phase_records = self._run(loop)
            _save_test_checkpoint(loop, tag)
            all_phase_records.append(phase_records)
            prev_tag = tag

        m = meta_learning_score(all_phase_records)
        analysis_lines = [
            f"Per-phase slopes:        {[f'{s:.5f}' for s in m.get('per_phase_slopes', [])]}",
            f"Slope trend:             {m.get('slope_trend', '?'):.5f}  (negative = improving)",
            f"Per-phase initial loss:  {[f'{l:.4f}' for l in m.get('per_phase_initial_loss', [])]}",
            f"Per-phase final loss:    {[f'{l:.4f}' for l in m.get('per_phase_final_loss', [])]}",
        ]
        # Meta-learning is observed if later domains start & end at lower loss
        initials = m.get("per_phase_initial_loss", [])
        finals   = m.get("per_phase_final_loss", [])
        passed = (
            len(initials) >= 2
            and initials[-1] < initials[0]
            and finals[-1] < finals[0]
        )
        analysis = "\n".join(analysis_lines)
        print(analysis)
        all_records = [r for phase in all_phase_records for r in phase]
        return LarkosTestResult(8, "Meta-Learning", m, analysis, passed, all_records)

    def run_test_9_affective_representations(self) -> LarkosTestResult:
        """
        Train on toy_physics while monitoring the C-side emotional and
        affective state closely.  We want to see whether the emotional
        system encodes deeper structure than just reward/loss:
        - valence tracks improvement direction (not just magnitude)
        - arousal tracks difficulty (not just success)
        - emotion intensities are diverse (not all the same type)
        - mask_intensity correlates with identity confidence
        """
        print("\n" + "=" * 60)
        print("TEST 9 - Affective / Emotional Representations")
        print("=" * 60)

        loop    = self._new_loop(_TEST_DATA / "toy_physics", self.epochs_short)
        records = self._run(loop)
        m       = affective_score(records)

        analysis_lines = [
            f"Valence-loss delta corr: {m.get('valence_loss_delta_corr', '?'):.4f}  (ideally < 0)",
            f"Arousal-loss corr:       {m.get('arousal_loss_corr', '?'):.4f}  (ideally > 0)",
            f"Reg. trend:              {m.get('emotional_regulation_trend', '?'):.5f}  (ideally > 0)",
            f"Emotion diversity:       {m.get('emotion_diversity', '?'):.4f}  (>0 = varied emotions)",
            f"Mask-identity corr:      {m.get('mask_identity_corr', '?'):.4f}",
            f"Cognitive impact range:  {m.get('cognitive_impact_range', '?'):.4f}",
            f"Affective complexity:    {m.get('affective_complexity', '?'):.4f}",
        ]
        # Deeper encoding requires both emotional variety and measurable VA-space complexity
        passed = (
            m.get("emotion_diversity", 0.0) > 0.05
            and m.get("affective_complexity", 0.0) > 0.1
        )
        analysis = "\n".join(analysis_lines)
        print(analysis)
        return LarkosTestResult(9, "Affective Representations", m, analysis, passed, records)

    def run_all(self) -> List[LarkosTestResult]:
        results = []
        tests = [
            self.run_test_1_learning_efficiency,
            self.run_test_2_domain_transfer,
            self.run_test_3_continual_learning,
            self.run_test_4_discovery,
            self.run_test_5_model_stability,
            self.run_test_6_internal_world_model,
            self.run_test_7_adaptation_speed,
            self.run_test_8_meta_learning,
            self.run_test_9_affective_representations,
        ]
        for test_fn in tests:
            try:
                result = test_fn()
                results.append(result)
            except Exception as exc:
                import traceback
                print(f"\nTest {test_fn.__name__} FAILED with exception:")
                traceback.print_exc()
                results.append(LarkosTestResult(
                    test_id=len(results) + 1,
                    name=test_fn.__name__,
                    metrics={},
                    analysis=f"EXCEPTION: {exc}",
                    passed=False,
                ))
        return results

    def print_report(self, results: List[LarkosTestResult]) -> None:
        print("\n\n" + "=" * 60)
        print("LARKOS TEST FRAMEWORK - FINAL REPORT")
        print("=" * 60)
        passed_n = sum(1 for r in results if r.passed)
        print(f"  {passed_n}/{len(results)} tests passed\n")

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] Test {r.test_id}: {r.name}")
            print(format_metrics(f"T{r.test_id} metrics", r.metrics))
            print()

        print("=" * 60)

    def save_report(
        self,
        results: List[LarkosTestResult],
        path: str = "test_report.json",
    ) -> None:
        """Saves a JSON report (metrics only - fused arrays excluded)."""
        report = []
        for r in results:
            safe_metrics = {}
            for k, v in r.metrics.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    safe_metrics[k] = v
                elif isinstance(v, list):
                    safe_metrics[k] = [
                        x if isinstance(x, (int, float, str, bool, type(None))) else str(x)
                        for x in v
                    ]
                else:
                    safe_metrics[k] = str(v)
            report.append({
                "test_id":  r.test_id,
                "name":     r.name,
                "passed":   r.passed,
                "metrics":  safe_metrics,
                "analysis": r.analysis,
            })
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Larkos Testing Framework")
    parser.add_argument(
        "--tests", nargs="*", type=int,
        help="Which tests to run (1-9). Omit to run all.",
    )
    parser.add_argument(
        "--epochs-short",     type=int, default=30,
        help="Epochs for short training phases (default 30)"
    )
    parser.add_argument(
        "--epochs-stability", type=int, default=60,
        help="Epochs for stability test (default 60)"
    )
    parser.add_argument(
        "--epochs-continual", type=int, default=25,
        help="Epochs per phase in continual-learning test (default 25)"
    )
    parser.add_argument(
        "--epochs-adaptation",type=int, default=25,
        help="Epochs per env in adaptation test (default 25)"
    )
    parser.add_argument(
        "--epochs-meta",      type=int, default=20,
        help="Epochs per domain in meta-learning test (default 20)"
    )
    parser.add_argument(
        "--report", type=str, default="test_report.json",
        help="Path to save JSON report"
    )
    args = parser.parse_args()

    fw = LarkosTestFramework(
        epochs_short      = args.epochs_short,
        epochs_stability  = args.epochs_stability,
        epochs_continual  = args.epochs_continual,
        epochs_adaptation = args.epochs_adaptation,
        epochs_meta       = args.epochs_meta,
    )

    test_map = {
        1: fw.run_test_1_learning_efficiency,
        2: fw.run_test_2_domain_transfer,
        3: fw.run_test_3_continual_learning,
        4: fw.run_test_4_discovery,
        5: fw.run_test_5_model_stability,
        6: fw.run_test_6_internal_world_model,
        7: fw.run_test_7_adaptation_speed,
        8: fw.run_test_8_meta_learning,
        9: fw.run_test_9_affective_representations,
    }

    if args.tests:
        results = []
        for t in args.tests:
            if t in test_map:
                results.append(test_map[t]())
            else:
                print(f"Unknown test {t}, skipping.")
    else:
        results = fw.run_all()

    fw.print_report(results)
    fw.save_report(results, args.report)
