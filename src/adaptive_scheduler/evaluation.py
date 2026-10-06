from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Callable

from .models import TaskSpec
from .simulator import TrainingEnvironment
from .strategies import AdaptiveScheduler, UniformScheduler


@dataclass(frozen=True)
class RunMetrics:
    mean_proficiency: float
    min_proficiency: float
    fraction_above_target: float


@dataclass(frozen=True)
class AggregateMetrics:
    mean_proficiency: float
    min_proficiency: float
    fraction_above_target: float


def default_tasks() -> list[TaskSpec]:
    return [
        TaskSpec("classification", difficulty=0.25, learning_rate=0.090, initial_proficiency=0.32),
        TaskSpec("reasoning", difficulty=0.75, learning_rate=0.055, initial_proficiency=0.18),
        TaskSpec("extraction", difficulty=0.20, learning_rate=0.075, initial_proficiency=0.58),
        TaskSpec("coding", difficulty=0.65, learning_rate=0.060, initial_proficiency=0.27),
        TaskSpec("summarisation", difficulty=0.30, learning_rate=0.070, initial_proficiency=0.52),
        TaskSpec("tool_use", difficulty=0.80, learning_rate=0.050, initial_proficiency=0.15),
        TaskSpec("planning", difficulty=0.70, learning_rate=0.052, initial_proficiency=0.21),
        TaskSpec("grounding", difficulty=0.55, learning_rate=0.062, initial_proficiency=0.40),
    ]


def load_tasks(path: str | Path) -> list[TaskSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TaskSpec(**item) for item in data]


def run_experiment(
    tasks: list[TaskSpec],
    scheduler_factory: Callable[[list[str], int], object],
    budget: int,
    seed: int,
    eval_every: int = 10,
    target: float = 0.80,
) -> RunMetrics:
    if budget <= 0:
        raise ValueError("budget must be > 0")

    env = TrainingEnvironment(tasks, seed=seed)
    scheduler = scheduler_factory([t.name for t in tasks], seed)
    observations = env.evaluate_all()
    if hasattr(scheduler, "observe"):
        scheduler.observe(observations)

    for step in range(budget):
        selected = scheduler.select(observations, step)
        env.train(selected)
        if (step + 1) % eval_every == 0:
            observations = env.evaluate_all()
            if hasattr(scheduler, "observe"):
                scheduler.observe(observations)

    prof = list(env.latent_proficiencies().values())
    return RunMetrics(
        mean_proficiency=mean(prof),
        min_proficiency=min(prof),
        fraction_above_target=sum(p >= target for p in prof) / len(prof),
    )


def aggregate(results: list[RunMetrics]) -> AggregateMetrics:
    if not results:
        raise ValueError("results cannot be empty")
    return AggregateMetrics(
        mean_proficiency=mean(r.mean_proficiency for r in results),
        min_proficiency=mean(r.min_proficiency for r in results),
        fraction_above_target=mean(r.fraction_above_target for r in results),
    )


def compare(
    tasks: list[TaskSpec],
    budget: int,
    seeds: int,
    target: float = 0.80,
) -> tuple[AggregateMetrics, AggregateMetrics]:
    uniform_runs = [
        run_experiment(tasks, lambda names, seed: UniformScheduler(names, seed), budget, seed, target=target)
        for seed in range(seeds)
    ]
    adaptive_runs = [
        run_experiment(tasks, lambda names, seed: AdaptiveScheduler(names, seed), budget, seed, target=target)
        for seed in range(seeds)
    ]
    return aggregate(uniform_runs), aggregate(adaptive_runs)
