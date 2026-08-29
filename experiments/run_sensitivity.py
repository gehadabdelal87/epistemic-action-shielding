#!/usr/bin/env python3
"""One-factor sensitivity analysis with paired scenario-level monotonicity checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[0]
# When this file is placed in <repo>/experiments/, its project root is one level up.
if ROOT.name == "experiments":
    PROJECT_ROOT = ROOT.parent
else:
    # Standalone validation/download location; project root can still be inferred at runtime
    # when copied into the repository. This fallback does not affect repository execution.
    PROJECT_ROOT = ROOT
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eas_shield.metrics import aggregate_records, decision_record, write_json, write_records_csv
from eas_shield.scenario_generation import generate_gate_scenario, stratified_gate_parameters
from eas_shield.shield import DecisionMode, DecisionTrace, EASDecisionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/sensitivity"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenarios", type=int, default=1000)
    return parser.parse_args()


def resolve_output(path: Path) -> Path:
    if path.is_absolute():
        return path
    # In repository use, anchor relative paths to the project root.
    if (PROJECT_ROOT / "pyproject.toml").exists():
        return PROJECT_ROOT / path
    return path.resolve()


def _subset_check(
    *,
    parameter: str,
    lower_threshold: float,
    higher_threshold: float,
    action_sets: dict[float, list[frozenset[str]]],
    scenario_ids: dict[float, list[str]],
) -> dict[str, object]:
    """Check the expected adjacent-threshold admissible-set monotonicity relation."""

    if parameter in {"source_reliability", "observation_quality"}:
        stricter_threshold = higher_threshold
        permissive_threshold = lower_threshold
    elif parameter == "risk":
        stricter_threshold = lower_threshold
        permissive_threshold = higher_threshold
    else:  # defensive: analyses below are fixed, but fail loudly if extended incorrectly
        raise ValueError(f"Unknown sensitivity parameter: {parameter}")

    stricter_sets = action_sets[stricter_threshold]
    permissive_sets = action_sets[permissive_threshold]
    ids = scenario_ids[stricter_threshold]

    violations: list[dict[str, object]] = []
    for index, (strict_set, permissive_set) in enumerate(zip(stricter_sets, permissive_sets)):
        if not strict_set.issubset(permissive_set):
            violations.append(
                {
                    "scenario_index": index,
                    "scenario_id": ids[index],
                    "stricter_admissible": sorted(strict_set),
                    "permissive_admissible": sorted(permissive_set),
                    "unexpected_actions": sorted(strict_set - permissive_set),
                }
            )

    return {
        "parameter": parameter,
        "lower_threshold": lower_threshold,
        "higher_threshold": higher_threshold,
        "stricter_threshold": stricter_threshold,
        "permissive_threshold": permissive_threshold,
        "expected_relation": "A_adm(stricter) subseteq A_adm(permissive)",
        "scenario_count": len(stricter_sets),
        "violation_count": len(violations),
        "passed": not violations,
        "violation_examples": violations[:10],
    }


def main() -> None:
    args = parse_args()
    if args.scenarios <= 0:
        raise ValueError("--scenarios must be a positive integer")

    output = resolve_output(args.output)
    output.mkdir(parents=True, exist_ok=True)

    engine = EASDecisionEngine()

    # Generate the paired scenario parameter schedule exactly once and reuse it
    # at every threshold. This makes the subset comparison scenario-by-scenario.
    parameters = stratified_gate_parameters(args.seed, args.scenarios)

    analyses: dict[str, list[float]] = {
        "source_reliability": [round(x / 10, 1) for x in range(1, 10)],
        "observation_quality": [round(x / 10, 1) for x in range(1, 10)],
        "risk": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    }

    all_records = []
    summary_rows: list[dict[str, object]] = []
    monotonicity_checks: list[dict[str, object]] = []

    for parameter_name, values in analyses.items():
        action_sets: dict[float, list[frozenset[str]]] = {}
        scenario_ids: dict[float, list[str]] = {}

        for threshold in values:
            records = []
            threshold_action_sets: list[frozenset[str]] = []
            threshold_scenario_ids: list[str] = []

            for index, params in enumerate(parameters):
                kwargs = {
                    "source_threshold": 0.7,
                    "quality_threshold": 0.7,
                    "risk_threshold": 10.0,
                }
                if parameter_name == "source_reliability":
                    kwargs["source_threshold"] = threshold
                elif parameter_name == "observation_quality":
                    kwargs["quality_threshold"] = threshold
                elif parameter_name == "risk":
                    kwargs["risk_threshold"] = threshold

                scenario = generate_gate_scenario(
                    seed=args.seed,
                    index=index,
                    parameters=params,
                    **kwargs,
                )
                trace = DecisionTrace()
                started = time.perf_counter_ns()
                outcome = engine.authorize(
                    state=scenario.state,
                    action_library=scenario.action_library,
                    governance_policy=scenario.governance_policy,
                    trace=trace,
                    mode=DecisionMode.OPTIMIZE,
                    utility_by_action=scenario.utility_by_action,
                    fallback_priority=scenario.fallback_priority,
                    random_seed=args.seed,
                    decision_id=f"sensitivity-{parameter_name}-{threshold}-{index}",
                )
                elapsed = time.perf_counter_ns() - started

                record = decision_record(
                    scenario_id=scenario.scenario_id,
                    seed=args.seed,
                    condition=f"{parameter_name}:{threshold}",
                    outcome=outcome,
                    canonical_actions=scenario.action_library,
                    canonical_governance=scenario.governance_policy,
                    utility_by_action=scenario.utility_by_action,
                    authorization_ns=elapsed,
                )
                records.append(record)
                all_records.append(record)
                threshold_action_sets.append(frozenset(outcome.admissible))
                threshold_scenario_ids.append(scenario.scenario_id)

            action_sets[threshold] = threshold_action_sets
            scenario_ids[threshold] = threshold_scenario_ids

            aggregate = aggregate_records(records)
            summary_rows.append(
                {
                    "parameter": parameter_name,
                    "threshold": threshold,
                    **{
                        key: value
                        for key, value in aggregate.items()
                        if not isinstance(value, dict)
                    },
                }
            )

        for lower_threshold, higher_threshold in zip(values, values[1:]):
            monotonicity_checks.append(
                _subset_check(
                    parameter=parameter_name,
                    lower_threshold=lower_threshold,
                    higher_threshold=higher_threshold,
                    action_sets=action_sets,
                    scenario_ids=scenario_ids,
                )
            )

    monotonicity_passed = all(bool(check["passed"]) for check in monotonicity_checks)
    monotonicity_report = {
        "seed": args.seed,
        "scenarios": args.scenarios,
        "paired_scenario_schedule": True,
        "overall_passed": monotonicity_passed,
        "checks": monotonicity_checks,
    }

    write_records_csv(all_records, output / "sensitivity_records.csv")
    with (output / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    write_json(monotonicity_report, output / "sensitivity_monotonicity.json")
    write_json(
        {
            "seed": args.seed,
            "scenarios": args.scenarios,
            "analyses": analyses,
            "paired_scenario_schedule": True,
            "monotonicity_verified": True,
            "monotonicity_passed": monotonicity_passed,
            "monotonicity_report": "sensitivity_monotonicity.json",
        },
        output / "manifest.json",
    )

    if not monotonicity_passed:
        violations = sum(int(check["violation_count"]) for check in monotonicity_checks)
        raise RuntimeError(
            f"Sensitivity monotonicity verification failed with {violations} scenario-level violation(s). "
            f"See {output / 'sensitivity_monotonicity.json'}"
        )


if __name__ == "__main__":
    main()
