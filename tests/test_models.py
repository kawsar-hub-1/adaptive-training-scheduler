import pytest

from adaptive_scheduler.models import TaskSpec


def test_task_spec_rejects_invalid_probability_fields() -> None:
    with pytest.raises(ValueError):
        TaskSpec("bad", difficulty=1.2, learning_rate=0.1, initial_proficiency=0.3)


def test_task_spec_requires_positive_learning_rate() -> None:
    with pytest.raises(ValueError):
        TaskSpec("bad", difficulty=0.5, learning_rate=0.0, initial_proficiency=0.3)
