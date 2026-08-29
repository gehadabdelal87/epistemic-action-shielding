#!/usr/bin/env python3
"""Execute the EAS experiment suite from a versioned JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/configured"))
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration field {key!r} must be an object")
    return value


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")

    version = config.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported or missing schema_version: expected {SCHEMA_VERSION}, got {version!r}"
        )

    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Archive the exact resolved experiment configuration with the results.
    (output / "config_used.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    primary = require_mapping(config, "primary")
    run([
        python,
        str(root / "experiments" / "run_primary_diagnostic.py"),
        "--seeds", ",".join(str(seed) for seed in primary["seeds"]),
        "--scenarios-per-seed", str(primary["scenarios_per_seed"]),
        "--extra-worlds", str(primary["extra_worlds"]),
        "--extra-propositions", str(primary["extra_propositions"]),
        "--extra-agents", str(primary["extra_agents"]),
        "--output", str(output / "primary"),
    ])

    scaling = require_mapping(config, "scaling")
    run([
        python,
        str(root / "experiments" / "run_scaling.py"),
        "--seed", str(scaling["seed"]),
        "--repetitions", str(scaling["repetitions"]),
        "--warmup", str(scaling["warmup"]),
        "--output", str(output / "scaling"),
    ])

    sensitivity = require_mapping(config, "sensitivity")
    run([
        python,
        str(root / "experiments" / "run_sensitivity.py"),
        "--seed", str(sensitivity["seed"]),
        "--scenarios", str(sensitivity["scenarios"]),
        "--output", str(output / "sensitivity"),
    ])

    update_revision = require_mapping(config, "update_revision")
    run([
        python,
        str(root / "experiments" / "run_update_revision_experiment.py"),
        "--seed", str(update_revision["seed"]),
        "--repetitions", str(update_revision["repetitions"]),
        "--output", str(output / "update_revision"),
    ])

    policy = require_mapping(config, "policy")
    run([
        python,
        str(root / "experiments" / "run_policy_integration.py"),
        "--training-seeds", ",".join(str(seed) for seed in policy["training_seeds"]),
        "--evaluation-seed", str(policy["evaluation_seed"]),
        "--training-episodes", str(policy["training_episodes"]),
        "--evaluation-episodes", str(policy["evaluation_episodes"]),
        "--confidence-threshold", str(policy["confidence_threshold"]),
        "--verification-probability", str(policy["verification_probability"]),
        "--safe-probability", str(policy["safe_probability"]),
        "--hidden-size", str(policy["hidden_size"]),
        "--batch-size", str(policy["batch_size"]),
        "--min-replay-size", str(policy["min_replay_size"]),
        "--epsilon-decay-steps", str(policy["epsilon_decay_steps"]),
        "--output", str(output / "policy"),
    ])

    statistics = require_mapping(config, "statistics")
    run([
        python,
        str(root / "experiments" / "statistical_analysis.py"),
        "--records", str(output / "primary" / "decision_records.csv"),
        "--output", str(output / "statistics"),
        "--reference", str(statistics["reference"]),
        "--bootstrap-resamples", str(statistics["bootstrap_resamples"]),
        "--seed", str(statistics["seed"]),
    ])

    figures = require_mapping(config, "figures")
    run([
        python,
        str(root / "experiments" / "make_figures.py"),
        "--results", str(output),
        "--output", str(output / "figures"),
        "--format", str(figures["format"]),
    ])


if __name__ == "__main__":
    main()
