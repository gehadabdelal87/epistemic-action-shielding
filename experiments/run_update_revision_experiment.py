#!/usr/bin/env python3
"""Evaluate event-indexed update and evidence-driven partition refinement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eas_shield.environment import OPERATOR, ROBOT, SAFE
from eas_shield.events import public_announcement_event_model
from eas_shield.formulas import Knows
from eas_shield.metrics import write_json
from eas_shield.revision import EvidencePolicy, Observation
from eas_shield.scenario_generation import GateScenarioParameters, generate_gate_scenario
from eas_shield.shield import DecisionMode, DecisionTrace, EASDecisionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "update_revision")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser.parse_args()


def parameters() -> GateScenarioParameters:
    return GateScenarioParameters(
        safe=True,
        robot_knows_safe=False,
        operator_knows_safe=False,
        confidence=0.9,
        source_reliability=0.95,
        observation_quality=0.95,
        provenance_known=True,
        autonomous_risk=5.0,
        coordinated_risk=5.0,
        communication_available=True,
        evidence_source_reachable=True,
        waiting_safe=True,
        shutdown_available=True,
        reviewer_authorized=True,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    engine = EASDecisionEngine()
    rows = []
    conditions = (
        "no_information",
        "public_announcement",
        "strong_robot_observation",
        "weak_robot_observation",
        "strong_operator_observation",
    )
    for repetition in range(args.repetitions):
        scenario = generate_gate_scenario(
            seed=args.seed,
            index=repetition,
            parameters=parameters(),
        )
        for condition in conditions:
            event_model = None
            designated_event = None
            observations = ()
            if condition == "public_announcement":
                event_model = public_announcement_event_model(
                    scenario.state.model.agents, SAFE, event_name="safe_announcement"
                )
                designated_event = "safe_announcement"
            elif condition in {
                "strong_robot_observation",
                "weak_robot_observation",
                "strong_operator_observation",
            }:
                recipient = OPERATOR if condition == "strong_operator_observation" else ROBOT
                strength = 0.4 if condition == "weak_robot_observation" else 0.95
                observations = (
                    Observation(
                        observation_id=f"{condition}-{repetition}",
                        recipient=recipient,
                        formula=SAFE,
                        observed_value=True,
                        confidence=strength,
                        source="verified_sensor",
                        source_reliability=0.95,
                        quality=0.95,
                        provenance_known=True,
                    ),
                )
            trace = DecisionTrace()
            outcome = engine.authorize(
                state=scenario.state,
                action_library=scenario.action_library,
                governance_policy=scenario.governance_policy,
                trace=trace,
                mode=DecisionMode.OPTIMIZE,
                utility_by_action=scenario.utility_by_action,
                fallback_priority=scenario.fallback_priority,
                event_model=event_model,
                designated_event=designated_event,
                observations=observations,
                evidence_policy=EvidencePolicy(),
                decision_id=f"update-{condition}-{repetition}",
            )
            checker = outcome.authorization_state.model.checker()
            rows.append(
                {
                    "repetition": repetition,
                    "condition": condition,
                    "robot_knows_safe": checker.satisfies(
                        outcome.authorization_state.world, Knows(ROBOT, SAFE)
                    ),
                    "operator_knows_safe": checker.satisfies(
                        outcome.authorization_state.world, Knows(OPERATOR, SAFE)
                    ),
                    "selected_action": outcome.selected_action or "halt",
                    "status": outcome.status.value,
                    "model_worlds": len(outcome.authorization_state.model.worlds),
                    "model_edges": outcome.authorization_state.model.edge_count,
                    "revision_count": len(outcome.trace_entry.revision_records),
                    "event_applied": outcome.trace_entry.event_record.get("status") == "applied",
                }
            )

    with (args.output / "update_revision_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            "n": len(selected),
            "robot_knowledge_rate": sum(row["robot_knows_safe"] for row in selected) / len(selected),
            "operator_knowledge_rate": sum(row["operator_knows_safe"] for row in selected) / len(selected),
            "autonomous_open_rate": sum(row["selected_action"] == "autonomous_open" for row in selected) / len(selected),
            "fallback_rate": sum(row["selected_action"] in {"request_evidence", "request_review", "wait", "safe_shutdown"} for row in selected) / len(selected),
        }
    write_json(summary, args.output / "update_revision_summary.json")


if __name__ == "__main__":
    main()
