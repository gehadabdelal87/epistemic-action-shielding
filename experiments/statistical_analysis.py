#!/usr/bin/env python3
"""Paired statistical analysis for the primary diagnostic experiment."""

from __future__ import annotations

# Allow this experiment to be run directly from a source checkout without
# requiring an editable install first. Installed-package execution still works.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import csv
import math
from pathlib import Path
import random
import statistics
from typing import Callable

from eas_shield.metrics import percentile, write_json


BOOL_FIELDS = {
    "epistemic_violation",
    "environmental_violation",
    "governance_violation",
    "full_admissibility_compliance",
    "agent_substitution_violation",
    "unsafe_open",
    "task_completed",
    "fallback",
    "halted",
}
FLOAT_FIELDS = {"utility", "constrained_regret", "authorization_ns"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("results/journal/primary/decision_records.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/journal/statistics"))
    parser.add_argument("--reference", default="eas")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def read_records(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in BOOL_FIELDS:
            row[field] = parse_bool(row[field])
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
        row["seed"] = int(row["seed"])
    return rows


def paired_rows(rows, condition_a: str, condition_b: str):
    index = {(row["scenario_id"], row["condition"]): row for row in rows}
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    pairs = []
    for scenario_id in scenario_ids:
        a = index.get((scenario_id, condition_a))
        b = index.get((scenario_id, condition_b))
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def exact_mcnemar(a_values, b_values):
    b_count = sum((not a) and b for a, b in zip(a_values, b_values))
    c_count = sum(a and (not b) for a, b in zip(a_values, b_values))
    n = b_count + c_count
    if n == 0:
        return 1.0, b_count, c_count
    tail = min(b_count, c_count)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2 * probability), b_count, c_count


def paired_bootstrap(values_a, values_b, resamples: int, seed: int):
    pairs = [
        (a, b)
        for a, b in zip(values_a, values_b)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if not pairs:
        return float("nan"), float("nan"), float("nan"), 0
    differences = [b - a for a, b in pairs]
    estimate = statistics.mean(differences)
    rng = random.Random(seed)
    n = len(pairs)
    samples = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        samples.append(statistics.mean(sample))
    samples.sort()
    return estimate, percentile(samples, 0.025), percentile(samples, 0.975), n


def holm_adjust(p_values):
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (index, value) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_records(args.records)
    conditions = sorted({row["condition"] for row in rows if row["condition"] != args.reference})
    binary_metrics = (
        "epistemic_violation",
        "environmental_violation",
        "governance_violation",
        "agent_substitution_violation",
        "task_completed",
        "fallback",
        "halted",
    )
    continuous_metrics = ("utility", "constrained_regret", "authorization_ns")
    output_rows = []
    p_value_positions = []
    p_values = []

    for condition in conditions:
        pairs = paired_rows(rows, args.reference, condition)
        for metric in binary_metrics:
            reference_values = [bool(a[metric]) for a, _ in pairs]
            condition_values = [bool(b[metric]) for _, b in pairs]
            p_value, b_count, c_count = exact_mcnemar(reference_values, condition_values)
            row = {
                "reference": args.reference,
                "condition": condition,
                "metric": metric,
                "outcome_type": "binary",
                "n": len(pairs),
                "reference_mean": statistics.mean(reference_values) if pairs else float("nan"),
                "condition_mean": statistics.mean(condition_values) if pairs else float("nan"),
                "paired_difference": (
                    statistics.mean(condition_values) - statistics.mean(reference_values)
                    if pairs
                    else float("nan")
                ),
                "ci_low": "",
                "ci_high": "",
                "test": "exact_mcnemar",
                "p_value": p_value,
                "discordant_reference0_condition1": b_count,
                "discordant_reference1_condition0": c_count,
            }
            p_value_positions.append(len(output_rows))
            p_values.append(p_value)
            output_rows.append(row)
        for metric in continuous_metrics:
            reference_values = [float(a[metric]) for a, _ in pairs]
            condition_values = [float(b[metric]) for _, b in pairs]
            difference, low, high, n = paired_bootstrap(
                reference_values,
                condition_values,
                args.bootstrap_resamples,
                args.seed + len(output_rows),
            )
            output_rows.append(
                {
                    "reference": args.reference,
                    "condition": condition,
                    "metric": metric,
                    "outcome_type": "continuous",
                    "n": n,
                    "reference_mean": statistics.mean([v for v in reference_values if math.isfinite(v)]) if any(math.isfinite(v) for v in reference_values) else float("nan"),
                    "condition_mean": statistics.mean([v for v in condition_values if math.isfinite(v)]) if any(math.isfinite(v) for v in condition_values) else float("nan"),
                    "paired_difference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "test": "paired_bootstrap_mean_difference",
                    "p_value": "",
                    "discordant_reference0_condition1": "",
                    "discordant_reference1_condition0": "",
                }
            )

    adjusted = holm_adjust(p_values)
    for position, value in zip(p_value_positions, adjusted):
        output_rows[position]["holm_adjusted_p"] = value
    for row in output_rows:
        row.setdefault("holm_adjusted_p", "")

    with (args.output / "paired_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    write_json(
        {
            "reference": args.reference,
            "conditions": conditions,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "comparisons": output_rows,
        },
        args.output / "paired_effects.json",
    )


if __name__ == "__main__":
    main()
