from adaptive_scheduler.evaluation import compare, default_tasks


def test_compare_returns_valid_metrics() -> None:
    uniform, adaptive = compare(default_tasks(), budget=80, seeds=3)
    for result in (uniform, adaptive):
        assert 0.0 <= result.mean_proficiency <= 1.0
        assert 0.0 <= result.min_proficiency <= 1.0
        assert 0.0 <= result.fraction_above_target <= 1.0
