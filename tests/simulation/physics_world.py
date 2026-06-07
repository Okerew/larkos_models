"""
2D rigid-body physics simulation for Larkos world-model testing.

Generates actual numeric state trajectories instead of pre-authored text so the
world-model tests evaluate whether the model can track real physical dynamics.
"""
from __future__ import annotations

from typing import Optional
from tests.simulation.config import NORMAL_GRAVITY, PhysicsConfig, REVERSED_GRAVITY

import numpy as np


class PhysicsWorld:
    """
    Minimal 2D rigid-body physics: gravity, friction, boundary reflection,
    elastic (or toy-doubled) pairwise collisions.

    Observation vector layout per object: [x, y, vx, vy, mass]  →  shape (5*N,)
    """

    N_FEATURES = 5  # features per object

    def __init__(
        self,
        config:    Optional[PhysicsConfig] = None,
        n_objects: int = 3,
        seed:      int = 42,
    ) -> None:
        self.config    = config or NORMAL_GRAVITY
        self.n_objects = n_objects
        self._rng      = np.random.RandomState(seed)
        self._pos      = np.zeros((n_objects, 2), dtype=np.float64)
        self._vel      = np.zeros((n_objects, 2), dtype=np.float64)
        self._mass     = np.ones(n_objects,        dtype=np.float64)
        self._radius   = np.full(n_objects, 0.04,  dtype=np.float64)
        self.step_count = 0
        self.reset()

    def reset(self) -> None:
        xmin, ymin, xmax, ymax = self.config.bounds
        for i in range(self.n_objects):
            self._pos[i] = [
                self._rng.uniform(xmin + 0.15, xmax - 0.15),
                self._rng.uniform(ymin + 0.15, ymax - 0.15),
            ]
            self._vel[i]  = self._rng.uniform(-0.3, 0.3, 2)
            self._mass[i] = self._rng.uniform(0.5, 2.0)
        self.step_count = 0

    def step(self) -> np.ndarray:
        """Advance one timestep. Returns the observation vector after the step."""
        cfg = self.config
        dt  = cfg.dt
        xmin, ymin, xmax, ymax = cfg.bounds

        self._vel[:, 1] += cfg.gravity * dt
        self._vel       *= cfg.friction
        self._pos       += self._vel * dt

        for i in range(self.n_objects):
            r = self._radius[i]
            for axis, lo, hi in ((0, xmin, xmax), (1, ymin, ymax)):
                if self._pos[i, axis] - r < lo:
                    self._pos[i, axis] = lo + r
                    self._vel[i, axis] = abs(self._vel[i, axis]) * cfg.restitution
                elif self._pos[i, axis] + r > hi:
                    self._pos[i, axis] = hi - r
                    self._vel[i, axis] = -abs(self._vel[i, axis]) * cfg.restitution

        for i in range(self.n_objects):
            for j in range(i + 1, self.n_objects):
                d    = self._pos[j] - self._pos[i]
                dist = float(np.linalg.norm(d)) + 1e-10
                min_d = self._radius[i] + self._radius[j]
                if dist < min_d:
                    n       = d / dist
                    overlap = min_d - dist
                    self._pos[i] -= n * overlap * 0.5
                    self._pos[j] += n * overlap * 0.5
                    if cfg.conserve_momentum:
                        dv  = self._vel[j] - self._vel[i]
                        dot = float(np.dot(dv, n))
                        if dot < 0:
                            mi  = self._mass[i]
                            mj  = self._mass[j]
                            imp = 2.0 * dot / (1.0 / mi + 1.0 / mj)
                            self._vel[i] += imp / mi * n
                            self._vel[j] -= imp / mj * n
                    else:
                        combined = self._vel[i] + self._vel[j]
                        self._vel[i] = combined * cfg.restitution
                        self._vel[j] = combined * cfg.restitution

        self.step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Flat observation: [x, y, vx, vy, mass] for each object. Shape (5*N,)."""
        parts: list[float] = []
        for i in range(self.n_objects):
            parts.extend([
                float(self._pos[i, 0]),
                float(self._pos[i, 1]),
                float(self._vel[i, 0]),
                float(self._vel[i, 1]),
                float(self._mass[i]),
            ])
        return np.array(parts, dtype=np.float32)

    def to_text(self) -> str:
        """Structured text description of the current state for the text encoder."""
        cfg   = self.config
        parts = [f"physics gravity={cfg.gravity:.1f}"]
        for i in range(self.n_objects):
            px, py = self._pos[i]
            vx, vy = self._vel[i]
            m      = self._mass[i]
            parts.append(
                f"obj{i} pos=({px:.3f},{py:.3f}) "
                f"vel=({vx:.3f},{vy:.3f}) mass={m:.2f}"
            )
        return ". ".join(parts) + "."

    @property
    def obs_dim(self) -> int:
        return self.n_objects * self.N_FEATURES
