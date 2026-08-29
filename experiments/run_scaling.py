#!/usr/bin/env python3
"""Controlled scaling probes for explicit finite EAS models.

The probe uses deterministic scenario indexing, records authorization latency and
resource-size diagnostics, and reports robust dispersion statistics (Q1/Q3/IQR)
in addition to median, mean, and p95 latency.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
import zlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eas_shield.actions import ActionLibrary
from eas_shield.metrics import percentile, write_json
from eas_shield.scenario_generation import generate_gate_scenario
from eas_shield.shield import DecisionMode, DecisionTrace, EASDecisionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "scaling")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _stable_label_index(label: str) -> int:
    """Return a process-independent scenario index for a configuration label."""
    return zlib.crc32(label.encode("utf-8")) % 10_000


def expand_actions(library: ActionLibrary, target_count: int) -> ActionLibrary:
    if target_count <= len(library.actions):
        return ActionLibrary(library.actions[:target_count], version=f"{library.version}-n{target_count}")
    actions = list(library.actions)
    source = library.get("autonomous_open")
    index = 0
    while len(actions) < target_count:
        actions.append(
            replace(
                source,
                name=f"diagnostic_action_{index}",
                transition_id="open_gate",
                cost=source.cost + index * 0.01,
                risk=source.risk,
                version=f"{source.version}-clone-{index}",
            )
        )
        index += 1
    return ActionLibrary(tuple(actions), version=f"{library.version}-n{target_count}")


def run_configuration(
    *,
    label: str,
    seed: int,
    repetitions: int,
    warmup: int,
    worlds: int = 16,
    extra_agents: int = 0,
    extra_propositions: int = 2,
    action_count: int = 6,
) -> list[dict[str, object]]:
    scenario = generate_gate_scenario(
        seed=seed,
        index=_stable_label_index(label),
        extra_worlds=max(0, worlds - 2),
        extra_agents=extra_agents,
        extra_propositions=extra_propositions,
    )
    actions = expand_actions(scenario.action_library, action_count)
    metadata = dict(scenario.state.metadata)
    metadata["authorized_actions"] = [action.name for action in actions.actions]
    state = scenario.state.with_metadata(**metadata)
    engine = EASDecisionEngine()

    utility_by_action = {
        **scenario.utility_by_action,
        **{action.name: 0.0 for action in actions.actions},
    }
    fallback_priority = tuple(
        name for name in scenario.fallback_priority if name in actions.by_name
    )

    for index in range(warmup):
        engine.authorize(
            state=state,
            action_library=actions,
            governance_policy=scenario.governance_policy,
            trace=DecisionTrace(),
            mode=DecisionMode.OPTIMIZE,
            utility_by_action=utility_by_action,
            fallback_priority=fallback_priority,
            decision_id=f"warmup-{label}-{index}",
        )

    formula_count = 2 * len(actions.actions)
    formula_size_total = sum(
        action.pre_epi.size + action.pre_env.size for action in actions.actions
    )

    rows: list[dict[str, object]] = []
    for repetition in range(repetitions):
        trace = DecisionTrace()
        tracemalloc.start()
        started = time.perf_counter_ns()
        outcome = engine.authorize(
            state=state,
            action_library=actions,
            governance_policy=scenario.governance_policy,
            trace=trace,
            mode=DecisionMode.OPTIMIZE,
            utility_by_action=utility_by_action,
            fallback_priority=fallback_priority,
            decision_id=f"scale-{label}-{repetition}",
        )
        elapsed = time.perf_counter_ns() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append(
            {
                "label": label,
                "repetition": repetition,
                "worlds": len(state.model.worlds),
                "agents": len(state.model.agents),
                "propositions": len(state.model.propositions),
                "actions": len(actions.actions),
                "edges": state.model.edge_count,
                "formula_count": formula_count,
                "formula_modal_depth": max(action.pre_epi.modal_depth for action in actions.actions),
                "formula_size_total": formula_size_total,
                "authorization_ns": elapsed,
                "authorization_ms": elapsed / 1_000_000,
                "peak_memory_bytes": peak,
                "trace_bytes": len(str(outcome.trace_entry.to_dict()).encode("utf-8")),
                "selected_action": outcome.selected_action or "halt",
                "status": outcome.status.value,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["label"]), []).append(row)

    summary: dict[str, object] = {}
    for label, group in groups.items():
        times = sorted(float(row["authorization_ms"]) for row in group)
        memory = sorted(float(row["peak_memory_bytes"]) for row in group)
        trace_sizes = sorted(float(row["trace_bytes"]) for row in group)

        q1 = percentile(times, 0.25)
        q3 = percentile(times, 0.75)
        summary[label] = {
            "n": len(group),
            "worlds": group[0]["worlds"],
            "agents": group[0]["agents"],
            "propositions": group[0]["propositions"],
            "actions": group[0]["actions"],
            "edges": group[0]["edges"],
            "formula_count": group[0]["formula_count"],
            "formula_modal_depth": group[0]["formula_modal_depth"],
            "formula_size_total": group[0]["formula_size_total"],
            "q1_authorization_ms": q1,
            "median_authorization_ms": statistics.median(times),
            "mean_authorization_ms": statistics.mean(times),
            "q3_authorization_ms": q3,
            "iqr_authorization_ms": q3 - q1,
            "p95_authorization_ms": percentile(times, 0.95),
            "median_peak_memory_bytes": statistics.median(memory),
            "p95_peak_memory_bytes": percentile(memory, 0.95),
            "median_trace_bytes": statistics.median(trace_sizes),
            "p95_trace_bytes": percentile(trace_sizes, 0.95),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for worlds in (8, 16, 32, 64, 128, 256):
        rows.extend(
            run_configuration(
                label=f"worlds_{worlds}",
                seed=args.seed,
                repetitions=args.repetitions,
                warmup=args.warmup,
                worlds=worlds,
                extra_agents=0,
                extra_propositions=2,
                action_count=6,
            )
        )
    for total_agents in (2, 3, 4, 6, 8):
        rows.extend(
            run_configuration(
                label=f"agents_{total_agents}",
                seed=args.seed + 1,
                repetitions=args.repetitions,
                warmup=args.warmup,
                worlds=32,
                extra_agents=total_agents - 2,
                extra_propositions=2,
                action_count=6,
            )
        )
    for extra_props in (0, 2, 4, 8, 16):
        rows.extend(
            run_configuration(
                label=f"context_props_{extra_props}",
                seed=args.seed + 2,
                repetitions=args.repetitions,
                warmup=args.warmup,
                worlds=32,
                extra_agents=0,
                extra_propositions=extra_props,
                action_count=6,
            )
        )
    for action_count in (3, 6, 12, 24, 48):
        rows.extend(
            run_configuration(
                label=f"actions_{action_count}",
                seed=args.seed + 3,
                repetitions=args.repetitions,
                warmup=args.warmup,
                worlds=32,
                extra_agents=0,
                extra_propositions=2,
                action_count=action_count,
            )
        )

    with (args.output / "scaling_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_json(summarize(rows), args.output / "scaling_summary.json")
    write_json(
        {
            "seed": args.seed,
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "scenario_indexing": "crc32(label) modulo 10000",
            "timing_excludes": ["scenario_generation", "file_output"],
            "dispersion_statistics": ["q1", "q3", "iqr"],
            "resource_diagnostics": [
                "peak_memory_bytes",
                "trace_bytes",
                "formula_count",
                "formula_size_total",
                "edges",
            ],
        },
        args.output / "manifest.json",
    )


if __name__ == "__main__":
    main()
