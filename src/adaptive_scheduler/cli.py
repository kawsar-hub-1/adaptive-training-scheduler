from __future__ import annotations

import argparse

from .evaluation import compare, default_tasks, load_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare rollout allocation strategies.")
    sub = parser.add_subparsers(dest="command", required=True)

    compare_parser = sub.add_parser("compare", help="Compare uniform and adaptive scheduling.")
    compare_parser.add_argument("--budget", type=int, default=600, help="Training rollouts per run.")
    compare_parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds.")
    compare_parser.add_argument("--target", type=float, default=0.80, help="Target proficiency threshold.")
    compare_parser.add_argument("--task-bank", type=str, default=None, help="Optional JSON task bank.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = load_tasks(args.task_bank) if args.task_bank else default_tasks()

    uniform, adaptive = compare(tasks, budget=args.budget, seeds=args.seeds, target=args.target)
    improvement = 100.0 * (adaptive.mean_proficiency - uniform.mean_proficiency) / uniform.mean_proficiency

    print(f"Budget: {args.budget} | Seeds: {args.seeds} | Tasks: {len(tasks)}")
    print("\nUniform")
    print(f"  mean proficiency       : {uniform.mean_proficiency:.4f}")
    print(f"  mean minimum proficiency: {uniform.min_proficiency:.4f}")
    print(f"  fraction above target  : {uniform.fraction_above_target:.3f}")

    print("\nAdaptive")
    print(f"  mean proficiency       : {adaptive.mean_proficiency:.4f}")
    print(f"  mean minimum proficiency: {adaptive.min_proficiency:.4f}")
    print(f"  fraction above target  : {adaptive.fraction_above_target:.3f}")

    print(f"\nRelative mean-proficiency improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()
