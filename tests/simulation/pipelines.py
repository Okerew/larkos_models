"""
Simulation-backed data pipelines for Larkos testing.

Both implement the TextDataPipeline interface (next_sample() -> str) so they
drop into TrainingLoop unchanged.  They additionally expose the raw simulation
state so PhysicsTestLoop / BondTestLoop can record it in epoch records.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tests.simulation.physics_world import PhysicsWorld
from tests.simulation.entity_world import EntityWorld


class PhysicsDataPipeline:
    """
    Streams sequential physics-state observations as structured text.

    Attributes exposed for PhysicsTestLoop._side_effects:
        prev_state    : observation vector *before* the last step (s_t)
        current_state : observation vector *after*  the last step (s_{t+1})
    """

    def __init__(self, world: PhysicsWorld) -> None:
        self.world         = world
        self.prev_state:    Optional[np.ndarray] = None
        self.current_state: np.ndarray           = world.observe()

    def next_sample(self) -> str:
        self.prev_state    = self.current_state
        self.current_state = self.world.step()
        return self.world.to_text()


class EntityInteractionPipeline:
    """
    Streams entity-interaction descriptions as text.

    Attributes exposed for BondTestLoop._side_effects:
        last_interaction : result dict from the most recent interact() call
    """

    def __init__(self, world: EntityWorld) -> None:
        self.world             = world
        self.last_interaction: Optional[dict] = None

    def next_sample(self) -> str:
        entity_id = self.world.select_entity()
        result    = self.world.interact(entity_id)
        self.last_interaction = result
        return self.world.to_text(entity_id, result["reward"])
