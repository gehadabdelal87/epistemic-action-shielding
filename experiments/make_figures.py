#!/usr/bin/env python3
"""Generate manuscript-ready figures from archived experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/journal"))
    parser.add_argument("--output", type=Path, default=Path("results/journal/figures"))
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: object) -> float:
    """Convert JSON/CSV numeric values to float while tolerating missing values."""
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def diagnostic_figure(results: Path, output: Path, extension: str) -> None:
    data = load_json(results / "primary" / "aggregate_metrics.json")
    conditions = list(data)
    labels = [condition.replace("_", " ") for condition in conditions]
    metrics = (
        "epistemic_violation_rate",
        "environmental_violation_rate",
        "governance_violation_rate",
    )
    x = list(range(len(conditions)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, metric in enumerate(metrics):
        values = [as_float(data[condition].get(metric)) for condition in conditions]
        ax.bar([value + (offset - 1) * width for value in x], values, width, label=metric.replace("_", " "))
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Layer-specific violation rates")
    ax.legend()
    save(fig, output / f"diagnostic_violation_rates.{extension}")


def usefulness_figure(results: Path, output: Path, extension: str) -> None:
    data = load_json(results / "primary" / "aggregate_metrics.json")
    conditions = list(data)
    labels = [condition.replace("_", " ") for condition in conditions]
    x = list(range(len(conditions)))
    width = 0.25
    metrics = ("task_completion_rate", "fallback_burden", "mean_constrained_regret")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, metric in enumerate(metrics):
        values = [as_float(data[condition].get(metric)) for condition in conditions]
        ax.bar([value + (offset - 1) * width for value in x], values, width, label=metric.replace("_", " "))
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Rate / mean regret")
    ax.set_title("Usefulness and intervention cost")
    ax.legend()
    save(fig, output / f"usefulness_tradeoff.{extension}")


def scaling_figure(results: Path, output: Path, extension: str) -> None:
    data = load_json(results / "scaling" / "scaling_summary.json")
    groups = {
        "worlds": [],
        "agents": [],
        "context_props": [],
        "actions": [],
    }
    for label, values in data.items():
        for prefix in groups:
            if label.startswith(prefix + "_"):
                x = int(label.rsplit("_", 1)[1])
                groups[prefix].append((x, as_float(values.get("median_authorization_ms"))))
    for prefix, points in groups.items():
        points.sort()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot([point[0] for point in points], [point[1] for point in points], marker="o")
        ax.set_xlabel(prefix.replace("_", " "))
        ax.set_ylabel("Median authorization time (ms)")
        ax.set_title(f"Controlled scaling: {prefix.replace('_', ' ')}")
        save(fig, output / f"scaling_{prefix}.{extension}")


def sensitivity_figures(results: Path, output: Path, extension: str) -> None:
    path = results / "sensitivity" / "sensitivity_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parameters = sorted({row["parameter"] for row in rows})
    metrics = ("open_rate", "task_completion_rate", "fallback_burden")
    for parameter in parameters:
        selected = sorted(
            (row for row in rows if row["parameter"] == parameter),
            key=lambda row: as_float(row.get("threshold")),
        )
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for metric in metrics:
            ax.plot(
                [as_float(row.get("threshold")) for row in selected],
                [as_float(row.get(metric)) for row in selected],
                marker="o",
                label=metric.replace("_", " "),
            )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Sensitivity: {parameter.replace('_', ' ')}")
        ax.legend()
        save(fig, output / f"sensitivity_{parameter}.{extension}")



def update_revision_figure(results: Path, output: Path, extension: str) -> None:
    path = results / "update_revision" / "update_revision_summary.json"
    if not path.exists():
        return
    data = load_json(path)
    conditions = list(data)
    labels = [condition.replace("_", " ") for condition in conditions]
    x = list(range(len(conditions)))
    width = 0.25
    metrics = ("robot_knowledge_rate", "operator_knowledge_rate", "autonomous_open_rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric in enumerate(metrics):
        values = [as_float(data[condition].get(metric)) for condition in conditions]
        ax.bar([value + (offset - 1) * width for value in x], values, width, label=metric.replace("_", " "))
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Informational update, agent knowledge, and action availability")
    ax.legend()
    save(fig, output / f"update_revision.{extension}")

def policy_figure(results: Path, output: Path, extension: str) -> None:
    path = results / "policy" / "dqn_summary.json"
    if not path.exists():
        return
    data = load_json(path)["results"]
    labels = ["unshielded", "confidence", "eas"]
    metrics = (
        "safe_completion_rate",
        "unsafe_action_rate",
        "epistemically_unsupported_action_rate",
        "epistemic_luck_rate",
    )
    for regime in ("in_distribution", "degraded_calibrated", "degraded_overconfident"):
        if regime not in data:
            continue
        x = list(range(len(labels)))
        width = 0.19
        fig, ax = plt.subplots(figsize=(9, 5))
        for offset, metric in enumerate(metrics):
            values = [as_float(data[regime][label].get(metric)) for label in labels]
            centered = offset - (len(metrics) - 1) / 2
            ax.bar(
                [value + centered * width for value in x],
                values,
                width,
                label=metric.replace("_", " "),
            )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title(f"DQN policy shielding: {regime.replace('_', ' ')}")
        ax.legend()
        save(fig, output / f"dqn_policy_{regime}.{extension}")

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        returns = [as_float(data[regime][label].get("mean_return")) for label in labels]
        ax.bar(x, returns)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Mean episode return")
        ax.set_title(f"DQN return: {regime.replace('_', ' ')}")
        save(fig, output / f"dqn_return_{regime}.{extension}")


def main() -> None:
    args = parse_args()
    diagnostic_figure(args.results, args.output, args.format)
    usefulness_figure(args.results, args.output, args.format)
    scaling_figure(args.results, args.output, args.format)
    sensitivity_figures(args.results, args.output, args.format)
    update_revision_figure(args.results, args.output, args.format)
    policy_figure(args.results, args.output, args.format)


if __name__ == "__main__":
    main()
