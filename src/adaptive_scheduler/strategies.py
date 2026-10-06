from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class ObservationStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    last_selected_step: int = -1

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.25
        return self.m2 / (self.count - 1)

    @property
    def uncertainty(self) -> float:
        # Standard error-like term with a floor for unobserved tasks.
        if self.count == 0:
            return 1.0
        return min(1.0, math.sqrt(self.variance / self.count))


class UniformScheduler:
    """Cycles evenly through tasks."""

    def __init__(self, task_names: list[str], seed: int = 0) -> None:
        if not task_names:
            raise ValueError("task_names cannot be empty")
        self.task_names = list(task_names)
        self.rng = random.Random(seed)
        self.order = list(task_names)
        self.rng.shuffle(self.order)
        self.index = 0

    def select(self, observations: dict[str, float], step: int) -> str:
        task = self.order[self.index]
        self.index = (self.index + 1) % len(self.order)
        return task


class AdaptiveScheduler:
    """Interpretable adaptive scheduler using observed performance only.

    The policy balances:
      - room to improve (low observed proficiency),
      - uncertainty (few/noisy observations),
      - staleness (not selected recently),
      - epsilon-greedy exploration.
    """

    def __init__(
        self,
        task_names: list[str],
        seed: int = 0,
        epsilon: float = 0.08,
        room_weight: float = 0.20,
        uncertainty_weight: float = 0.10,
        staleness_weight: float = 0.70,
    ) -> None:
        if not task_names:
            raise ValueError("task_names cannot be empty")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.task_names = list(task_names)
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.room_weight = room_weight
        self.uncertainty_weight = uncertainty_weight
        self.staleness_weight = staleness_weight
        self.stats = {name: ObservationStats() for name in task_names}

    def observe(self, observations: dict[str, float]) -> None:
        for name, value in observations.items():
            self.stats[name].update(value)

    def select(self, observations: dict[str, float], step: int) -> str:
        if self.rng.random() < self.epsilon:
            choice = self.rng.choice(self.task_names)
            self.stats[choice].last_selected_step = step
            return choice

        max_age = max(1, step + 1)
        best_name = self.task_names[0]
        best_score = float("-inf")

        for name in self.task_names:
            stats = self.stats[name]
            observed = stats.mean if stats.count else 0.5
            room = 1.0 - observed
            uncertainty = stats.uncertainty
            age = step - stats.last_selected_step if stats.last_selected_step >= 0 else max_age
            staleness = min(1.0, age / max_age)

            score = (
                self.room_weight * room
                + self.uncertainty_weight * uncertainty
                + self.staleness_weight * staleness
            )

            if score > best_score:
                best_score = score
                best_name = name

        self.stats[best_name].last_selected_step = step
        return best_name
