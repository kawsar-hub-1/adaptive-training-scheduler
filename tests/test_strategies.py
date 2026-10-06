from adaptive_scheduler.strategies import AdaptiveScheduler, UniformScheduler


def test_uniform_visits_all_tasks() -> None:
    scheduler = UniformScheduler(["a", "b", "c"], seed=0)
    selected = [scheduler.select({}, step) for step in range(3)]
    assert set(selected) == {"a", "b", "c"}


def test_adaptive_prefers_lower_observed_task_when_exploration_disabled() -> None:
    scheduler = AdaptiveScheduler(["easy", "hard"], seed=0, epsilon=0.0)
    observations = {"easy": 0.9, "hard": 0.2}
    scheduler.observe(observations)
    assert scheduler.select(observations, step=10) == "hard"
