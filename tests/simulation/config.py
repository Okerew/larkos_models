from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class EntityProfile:
    entity_id:        int
    name:             str
    reward_mean:      float  # expected reward; > 0 positive, < 0 negative
    reward_std:       float  # spread (low = predictable)
    consistency:      float  # 0 = random, 1 = perfectly consistent
    preferred_action: int    # action that earns maximum reward


DEFAULT_ENTITIES: List[EntityProfile] = [
    EntityProfile(0, "consistent_positive", reward_mean= 0.80, reward_std=0.05, consistency=0.95, preferred_action=0),
    EntityProfile(1, "consistent_negative", reward_mean=-0.60, reward_std=0.05, consistency=0.90, preferred_action=1),
    EntityProfile(2, "erratic",             reward_mean= 0.10, reward_std=0.70, consistency=0.15, preferred_action=2),
    EntityProfile(3, "neutral_steady",      reward_mean= 0.00, reward_std=0.10, consistency=0.80, preferred_action=3),
]

@dataclass
class PhysicsConfig:
    gravity:            float = -9.8    # m/s², negative = downward
    friction:           float = 0.995   # velocity retention per step
    restitution:        float = 0.8     # bounce factor at boundaries/collisions
    dt:                 float = 0.02    # timestep in seconds
    conserve_momentum:  bool  = True    # False → toy doubling rule on collision
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def variant(self, **overrides) -> "PhysicsConfig":
        """Return a copy of this config with selected fields overridden."""
        d = {
            "gravity":           self.gravity,
            "friction":          self.friction,
            "restitution":       self.restitution,
            "dt":                self.dt,
            "conserve_momentum": self.conserve_momentum,
            "bounds":            self.bounds,
        }
        d.update(overrides)
        return PhysicsConfig(**d)


NORMAL_GRAVITY    = PhysicsConfig(gravity=-9.8, conserve_momentum=True)
REVERSED_GRAVITY  = PhysicsConfig(gravity=+9.8, conserve_momentum=True)
DOUBLED_GRAVITY   = PhysicsConfig(gravity=-19.6, conserve_momentum=True)
TOY_PHYSICS       = PhysicsConfig(gravity=-9.8, conserve_momentum=False)
