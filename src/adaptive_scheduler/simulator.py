from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .models import TaskSpec


@dataclass
class TaskState:
    proficiency: float
    train_count: int = 0
    last_trained_step: int = -1


class TrainingEnvironment:
    """Synthetic environment with saturating, noisy learning dynamics."""

    def __init__(self, tasks: Iterable[TaskSpec], seed: int = 0) -> None:
        self.tasks = list(tasks)
        if not self.tasks:
            raise ValueError("At least one task is required")
        self.rng = random.Random(seed)
        self.states = {
            task.name: TaskState(proficiency=task.initial_proficiency)
            for task in self.tasks
        }
        self.step = 0

    def train(self, task_name: str) -> None:
        task = self._task(task_name)
        state = self.states[task_name]

        # Saturating improvement: large when proficiency is low, small near 1.0.
        effective_rate = task.learning_rate * (1.0 - 0.65 * task.difficulty)
        deterministic_gain = effective_rate * (1.0 - state.proficiency)

        # Small stochasticity prevents the environment from being perfectly smooth.
        jitter = self.rng.gauss(0.0, 0.08 * deterministic_gain) if deterministic_gain > 0 else 0.0
        state.proficiency = min(1.0, max(0.0, state.proficiency + deterministic_gain + jitter))
        state.train_count += 1
        state.last_trained_step = self.step
        self.step += 1

    def evaluate(self, task_name: str) -> float:
        task = self._task(task_name)
        latent = self.states[task_name].proficiency
        observed = latent + self.rng.gauss(0.0, task.eval_noise)
        return min(1.0, max(0.0, observed))

    def evaluate_all(self) -> dict[str, float]:
        return {task.name: self.evaluate(task.name) for task in self.tasks}

    def latent_proficiencies(self) -> dict[str, float]:
        return {name: state.proficiency for name, state in self.states.items()}

    def _task(self, task_name: str) -> TaskSpec:
        for task in self.tasks:
            if task.name == task_name:
                return task
        raise KeyError(f"Unknown task: {task_name}")
