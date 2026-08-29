#!/usr/bin/env python3
"""Run the paired primary diagnostic experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eas_shield.metrics import aggregate_records, decision_record, write_json, write_records_csv
from eas_shield.replay import replay_authorization
from eas_shield.scenario_generation import generate_gate_scenario, stratified_gate_parameters
from eas_shield.shield import DecisionMode, DecisionTrace, EASDecisionEngine
from eas_shield.variants import CONDITIONS, action_library_for_condition, governance_policy_for_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--scenarios-per-seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "primary")
    parser.add_argument("--extra-worlds", type=int, default=6)
    parser.add_argument("--extra-propositions", type=int, default=2)
    parser.add_argument("--extra-agents", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    engine = EASDecisionEngine()
    records = []
    strata_rows = []
    trace_rows = []

    for seed in seeds:
        parameters = stratified_gate_parameters(seed, args.scenarios_per_seed)
        for index, params in enumerate(parameters):
            scenario = generate_gate_scenario(
                seed=seed,
                index=index,
                parameters=params,
                extra_worlds=args.extra_worlds,
                extra_propositions=args.extra_propositions,
                extra_agents=args.extra_agents,
            )
            strata_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "seed": seed,
                    **scenario.stratum,
                }
            )
            for condition in CONDITIONS:
                action_library = action_library_for_condition(
                    scenario.action_library, condition
                )
                governance = governance_policy_for_condition(
                    scenario.governance_policy, condition
                )
                trace = DecisionTrace()
                started = time.perf_counter_ns()
                outcome = engine.authorize(
                    state=scenario.state,
                    action_library=action_library,
                    governance_policy=governance,
                    trace=trace,
                    mode=DecisionMode.OPTIMIZE,
                    utility_by_action=scenario.utility_by_action,
                    fallback_priority=scenario.fallback_priority,
                    random_seed=seed,
                    decision_id=f"{scenario.scenario_id}-{condition}",
                )
                elapsed = time.perf_counter_ns() - started
                replay = replay_authorization(outcome.trace_entry)
                records.append(
                    decision_record(
                        scenario_id=scenario.scenario_id,
                        seed=seed,
                        condition=condition,
                        outcome=outcome,
                        canonical_actions=scenario.action_library,
                        canonical_governance=scenario.governance_policy,
                        utility_by_action=scenario.utility_by_action,
                        authorization_ns=elapsed,
                        trace_replay_success=replay.success,
                    )
                )
                trace_rows.append(
                    {
                        "decision_id": outcome.decision_id,
                        "scenario_id": scenario.scenario_id,
                        "condition": condition,
                        "trace_hash": outcome.trace_entry.trace_hash,
                        "replay_success": replay.success,
                        "trace": outcome.trace_entry.to_dict(),
                    }
                )

    write_records_csv(records, args.output / "decision_records.csv")
    grouped = {
        condition: aggregate_records(
            [record for record in records if record.condition == condition]
        )
        for condition in CONDITIONS
    }
    write_json(grouped, args.output / "aggregate_metrics.json")
    write_json(trace_rows, args.output / "authorization_traces.json")

    if strata_rows:
        with (args.output / "scenario_strata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(strata_rows[0]))
            writer.writeheader()
            writer.writerows(strata_rows)

    manifest = {
        "seeds": seeds,
        "scenarios_per_seed": args.scenarios_per_seed,
        "conditions": list(CONDITIONS),
        "records": len(records),
        "extra_worlds": args.extra_worlds,
        "extra_propositions": args.extra_propositions,
        "extra_agents": args.extra_agents,
        "outputs": [
            "decision_records.csv",
            "aggregate_metrics.json",
            "authorization_traces.json",
            "scenario_strata.csv",
        ],
    }
    write_json(manifest, args.output / "manifest.json")
    print(json.dumps(grouped, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
