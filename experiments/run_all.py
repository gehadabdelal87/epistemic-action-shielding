#!/usr/bin/env python3
"""Run the complete reproducibility pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/journal"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    root = Path(__file__).resolve().parents[1]
    primary_n = "50" if args.quick else "1000"
    scaling_reps = "10" if args.quick else "100"
    policy_train = "500" if args.quick else "5000"
    policy_eval = "20" if args.quick else "100"
    policy_seeds = "2026" if args.quick else "2026,2027,2028,2029,2030,2031,2032,2033,2034,2035"

    run([python, "-m", "pytest", str(root / "tests")])
    run(
        [
            python,
            str(root / "experiments" / "run_primary_diagnostic.py"),
            "--scenarios-per-seed",
            primary_n,
            "--output",
            str(args.output / "primary"),
        ]
    )
    run(
        [
            python,
            str(root / "experiments" / "run_scaling.py"),
            "--repetitions",
            scaling_reps,
            "--output",
            str(args.output / "scaling"),
        ]
    )
    run(
        [
            python,
            str(root / "experiments" / "run_sensitivity.py"),
            "--scenarios",
            primary_n,
            "--output",
            str(args.output / "sensitivity"),
        ]
    )
    run(
        [
            python,
            str(root / "experiments" / "run_update_revision_experiment.py"),
            "--repetitions",
            "20" if args.quick else "500",
            "--output",
            str(args.output / "update_revision"),
        ]
    )
    run(
        [
            python,
            str(root / "experiments" / "run_policy_integration.py"),
            "--training-seeds",
            policy_seeds,
            "--training-episodes",
            policy_train,
            "--evaluation-episodes",
            policy_eval,
            "--output",
            str(args.output / "policy"),
        ]
    )
    run(
        [
            python,
            str(root / "experiments" / "statistical_analysis.py"),
            "--records",
            str(args.output / "primary" / "decision_records.csv"),
            "--output",
            str(args.output / "statistics"),
        ]
    )
    run(
        [
            python,
            str(root / "experiments" / "make_figures.py"),
            "--results",
            str(args.output),
            "--output",
            str(args.output / "figures"),
        ]
    )


if __name__ == "__main__":
    main()
