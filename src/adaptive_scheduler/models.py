from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Configuration for one synthetic training task.

    Attributes
    ----------
    name:
        Human-readable task identifier.
    difficulty:
        Value in [0, 1]. Larger values reduce the effective learning rate.
    learning_rate:
        Base learning increment applied when the task is trained.
    initial_proficiency:
        Starting latent proficiency in [0, 1].
    eval_noise:
        Standard deviation of Gaussian evaluation noise.
    """

    name: str
    difficulty: float
    learning_rate: float
    initial_proficiency: float
    eval_noise: float = 0.04

    def __post_init__(self) -> None:
        for field_name in ("difficulty", "learning_rate", "initial_proficiency", "eval_noise"):
            value = getattr(self, field_name)
            if field_name == "learning_rate":
                if value <= 0:
                    raise ValueError("learning_rate must be > 0")
            elif field_name == "eval_noise":
                if value < 0:
                    raise ValueError("eval_noise must be >= 0")
            elif not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be in [0, 1]")
