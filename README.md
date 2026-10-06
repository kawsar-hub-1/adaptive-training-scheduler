# Adaptive Training Scheduler

A small, reproducible Python research-software project for studying how limited training rollouts can be allocated across tasks more efficiently than uniform sampling.

The project uses a synthetic learning environment so the experiments are self-contained and do not depend on proprietary datasets or assessment material. It compares a uniform baseline with an adaptive scheduler that prioritises tasks using three signals: **room to improve**, **uncertainty**, and **staleness**, while retaining a small amount of exploration.

## Research question

Given a fixed rollout budget and a bank of tasks with different difficulty and learning dynamics, can an adaptive scheduler improve final average proficiency compared with uniform allocation?

## What this repository demonstrates

- Python package structure suitable for research software
- Reproducible synthetic simulation
- Baseline and adaptive scheduling strategies
- Explicit evaluation metrics and seeded experiments
- Command-line interface for repeatable experiments
- Unit tests
- GitHub Actions CI configuration

## Method

Each task has a latent proficiency value in `[0, 1]`. Training on a task increases proficiency according to a saturating learning curve with task-specific difficulty and learning rate. Evaluation is noisy, so the scheduler must infer where additional rollouts are most useful from observations rather than latent state.

The adaptive score is:

```text
score = 0.20 * room_to_improve
      + 0.10 * uncertainty
      + 0.70 * staleness
```

A small epsilon-greedy exploration probability prevents the scheduler from permanently ignoring tasks that currently look unpromising.

This is intentionally simple and interpretable. It is designed as a compact research-software example rather than as a claim of a universally optimal training policy.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install

```bash
pip install -e .
pip install -r requirements-dev.txt
```

### 3. Run a comparison

```bash
adaptive-scheduler compare --budget 300 --seeds 20
```

### 4. Run tests

```bash
pytest
```

## Example task bank

`examples/task_bank.json` contains a small synthetic task bank. You can run it directly:

```bash
adaptive-scheduler compare --task-bank examples/task_bank.json --budget 300 --seeds 20
```


## Reproducibility note

This repository is independently implemented with synthetic data. It contains no proprietary datasets, recruitment-assessment files, hidden tests, or third-party confidential material.

## Example results

Using the bundled eight-task synthetic bank, a budget of 300 training rollouts, and 20 random seeds:

| Strategy | Mean proficiency | Mean minimum proficiency | Fraction of tasks ≥ 0.80 |
|---|---:|---:|---:|
| Uniform | 0.8325 | 0.6566 | 0.606 |
| Adaptive | 0.8381 | 0.7956 | 0.850 |

In this setting, the adaptive policy improves average proficiency modestly while substantially improving the weakest-task outcome. The exact values are reproducible with:

```bash
adaptive-scheduler compare --budget 300 --seeds 20
```

## Metrics

The CLI reports:

- final mean proficiency across tasks
- minimum task proficiency
- fraction of tasks above a target proficiency threshold
- relative improvement of the adaptive scheduler over uniform sampling

## Repository structure

```text
adaptive-training-scheduler/
├── src/adaptive_scheduler/
│   ├── __init__.py
│   ├── cli.py
│   ├── evaluation.py
│   ├── models.py
│   ├── simulator.py
│   └── strategies.py
├── tests/
├── examples/
├── .github/workflows/
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── README.md
```

## Design notes

The code separates the simulator from scheduling strategies so that alternative policies can be evaluated against the same environment. Randomness is controlled through explicit seeds, and the simulator exposes only observed evaluation scores to the scheduler.

## Limitations

- The environment is synthetic and deliberately small.
- Learning dynamics are simplified and independent across tasks.
- The adaptive heuristic is interpretable but not formally optimal.
- Real post-training systems can involve non-stationarity, transfer between tasks, reward-model error, and much richer cost constraints.

These limitations are useful directions for extending the project.

## Possible extensions

- contextual bandit policies
- Bayesian uncertainty estimates
- task-transfer graphs
- cost-aware scheduling
- non-stationary task difficulty
- richer fairness constraints across task groups

## License

MIT License. See `LICENSE`.
