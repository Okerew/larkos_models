"""
Entity interaction world for testing affective bond formation.

Four entities with distinct, stable behavioral profiles allow us to test
whether bonds track real interaction quality rather than just loss magnitude.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from tests.simulation.config import EntityProfile, DEFAULT_ENTITIES

import numpy as np


class EntityWorld:
    """
    N entities with fixed behavioral profiles.

    Each call to interact(entity_id, action) returns a reward shaped by the
    entity's mean, variability and consistency.  The caller can supply an
    action derived from the model's output to make the interaction
    input-dependent; if omitted it defaults to 0.

    Interaction result keys:
        entity_id, entity_name, action, reward,
        trust_signal [0,1], valence [-1,1]
    """

    def __init__(
        self,
        profiles:  Optional[List[EntityProfile]] = None,
        n_actions: int = 4,
        seed:      int = 42,
    ) -> None:
        self.profiles  = profiles or DEFAULT_ENTITIES
        self._by_id: Dict[int, EntityProfile] = {
            p.entity_id: p for p in self.profiles
        }
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(seed)
        self.step_count  = 0
        self._history: Dict[int, List[float]] = {
            p.entity_id: [] for p in self.profiles
        }

    def select_entity(self) -> int:
        """Round-robin entity selection so each entity is visited equally."""
        return self.step_count % len(self.profiles)

    def interact(
        self,
        entity_id: int,
        action:    Optional[int] = None,
    ) -> dict:
        p = self._by_id[entity_id]
        if action is None:
            action = 0

        base  = p.reward_mean + (0.2 if action == p.preferred_action else 0.0)
        noise = float(self._rng.normal(0.0, p.reward_std))
        reward = base * p.consistency + noise * (1.0 - p.consistency)
        reward = float(np.clip(reward, -1.0, 1.0))

        self._history[entity_id].append(reward)
        self.step_count += 1

        return {
            "entity_id":    entity_id,
            "entity_name":  p.name,
            "action":       action,
            "reward":       reward,
            "trust_signal": float((reward + 1.0) / 2.0),   # [0, 1]
            "valence":      reward,                          # [-1, 1]
        }

    def to_text(self, entity_id: int, reward: float) -> str:
        name = self._by_id[entity_id].name
        tone = "positive" if reward > 0.15 else ("negative" if reward < -0.15 else "neutral")
        return (
            f"Step {self.step_count}: interacted with {name}. "
            f"Outcome {tone} (reward={reward:.3f})."
        )

    def entity_stats(self) -> Dict[str, dict]:
        stats: Dict[str, dict] = {}
        for p in self.profiles:
            hist = self._history[p.entity_id]
            if hist:
                stats[p.name] = {
                    "mean_reward":    float(np.mean(hist)),
                    "std_reward":     float(np.std(hist)),
                    "n_interactions": len(hist),
                }
        return stats
