from adaptive_scheduler.models import TaskSpec
from adaptive_scheduler.simulator import TrainingEnvironment


def test_training_improves_proficiency_on_average() -> None:
    task = TaskSpec("x", difficulty=0.2, learning_rate=0.1, initial_proficiency=0.2, eval_noise=0.0)
    env = TrainingEnvironment([task], seed=1)
    before = env.latent_proficiencies()["x"]
    for _ in range(5):
        env.train("x")
    after = env.latent_proficiencies()["x"]
    assert after > before


def test_evaluation_is_bounded() -> None:
    task = TaskSpec("x", difficulty=0.2, learning_rate=0.1, initial_proficiency=0.95, eval_noise=0.5)
    env = TrainingEnvironment([task], seed=2)
    for _ in range(100):
        assert 0.0 <= env.evaluate("x") <= 1.0
