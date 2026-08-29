"""Per-decision records and aggregate metrics with explicit denominators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from .actions import ActionLibrary
from .governance import GovernancePolicy
from .model import PointedState
from .shield import AuthorizationOutcome


OPEN_ACTIONS = frozenset({"autonomous_open", "coordinated_open"})
FALLBACK_ACTIONS = frozenset(
    {"request_evidence", "request_review", "wait", "safe_shutdown"}
)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    scenario_id: str
    seed: int
    condition: str
    selected_action: str | None
    proposal: str | None
    safe_world: bool
    opened: bool
    fallback: bool
    halted: bool
    epistemic_violation: bool
    environmental_violation: bool
    governance_violation: bool
    full_admissibility_compliance: bool
    agent_substitution_violation: bool
    unsafe_open: bool
    task_completed: bool
    utility: float
    constrained_regret: float
    proposal_intervened: bool
    proposal_inadmissible: bool
    inadmissible_proposal_blocked: bool
    admissible_proposal_override: bool
    authorization_ns: int
    trace_replay_success: bool | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decision_record(
    *,
    scenario_id: str,
    seed: int,
    condition: str,
    outcome: AuthorizationOutcome,
    canonical_actions: ActionLibrary,
    canonical_governance: GovernancePolicy,
    utility_by_action: Mapping[str, float],
    authorization_ns: int,
    trace_replay_success: bool | None = None,
) -> DecisionRecord:
    state = outcome.authorization_state
    checker = state.model.checker()
    selected = outcome.selected_action
    safe = checker.satisfies(state.world, canonical_actions.get("autonomous_open").pre_env) if "autonomous_open" in canonical_actions.by_name else False
    # The environmental precondition above includes gate operation.  The oracle
    # truth of safety is stored separately in the scenario metadata when present.
    safe_world = bool(state.metadata.get("scenario_oracle", {}).get("safe", safe))
    opened = selected in OPEN_ACTIONS
    fallback = selected in FALLBACK_ACTIONS
    halted = selected is None

    epi_violation = False
    env_violation = False
    gov_violation = False
    full_compliance = selected is not None
    if selected is not None:
        canonical = canonical_actions.get(selected)
        epi_ok = checker.satisfies(state.world, canonical.pre_epi)
        env_ok = checker.satisfies(state.world, canonical.pre_env)
        gov_ok = canonical_governance.evaluate(state, canonical).passed
        epi_violation = not epi_ok
        env_violation = not env_ok
        gov_violation = not gov_ok
        full_compliance = epi_ok and env_ok and gov_ok

    robot_knows = False
    operator_knows = False
    try:
        from .environment import OPERATOR, ROBOT, SAFE
        from .formulas import Knows

        robot_knows = checker.satisfies(state.world, Knows(ROBOT, SAFE))
        operator_knows = checker.satisfies(state.world, Knows(OPERATOR, SAFE))
    except Exception:
        pass
    substitution = selected == "autonomous_open" and operator_knows and not robot_knows
    unsafe_open = opened and not safe_world

    if safe_world:
        task_completed = opened and not env_violation
    else:
        task_completed = selected == "safe_shutdown" or (halted and not opened)

    utility = float(utility_by_action.get(selected or "halt", 0.0))
    canonical_admissible: list[str] = []
    for action in canonical_actions.actions:
        if (
            checker.satisfies(state.world, action.pre_epi)
            and checker.satisfies(state.world, action.pre_env)
            and canonical_governance.evaluate(state, action).passed
        ):
            canonical_admissible.append(action.name)
    oracle_utility = max(
        (float(utility_by_action.get(name, -canonical_actions.get(name).cost)) for name in canonical_admissible),
        default=0.0,
    )
    # Constrained regret is defined only for halt or canonically admissible
    # selections.  An inadmissible action may have higher nominal utility than
    # the constrained optimum, so reporting a negative "regret" would be
    # misleading rather than informative.
    if selected is None:
        regret = max(0.0, oracle_utility)
    elif full_compliance:
        regret = max(0.0, oracle_utility - utility)
    else:
        regret = float("nan")

    proposal = outcome.proposal
    proposal_intervened = proposal is not None and proposal != selected
    proposal_inadmissible = proposal is not None and proposal not in canonical_admissible
    inadmissible_blocked = proposal_inadmissible and proposal != selected
    admissible_proposal_override = (
        proposal is not None
        and proposal in canonical_admissible
        and proposal != selected
    )

    return DecisionRecord(
        scenario_id=scenario_id,
        seed=seed,
        condition=condition,
        selected_action=selected,
        proposal=proposal,
        safe_world=safe_world,
        opened=opened,
        fallback=fallback,
        halted=halted,
        epistemic_violation=epi_violation,
        environmental_violation=env_violation,
        governance_violation=gov_violation,
        full_admissibility_compliance=full_compliance,
        agent_substitution_violation=substitution,
        unsafe_open=unsafe_open,
        task_completed=task_completed,
        utility=utility,
        constrained_regret=regret,
        proposal_intervened=proposal_intervened,
        proposal_inadmissible=proposal_inadmissible,
        inadmissible_proposal_blocked=inadmissible_blocked,
        admissible_proposal_override=admissible_proposal_override,
        authorization_ns=authorization_ns,
        trace_replay_success=trace_replay_success,
        status=outcome.status.value,
    )


def _rate(records: Sequence[DecisionRecord], numerator: str, denominator_filter=None) -> tuple[float, int, int]:
    denominator_records = [record for record in records if denominator_filter is None or denominator_filter(record)]
    denominator = len(denominator_records)
    numerator_count = sum(bool(getattr(record, numerator)) for record in denominator_records)
    return (numerator_count / denominator if denominator else float("nan"), numerator_count, denominator)


def aggregate_records(records: Sequence[DecisionRecord]) -> dict[str, Any]:
    executed = lambda r: not r.halted
    proposed = lambda r: r.proposal is not None
    inadmissible_proposal = lambda r: r.proposal_inadmissible
    admissible_proposal = lambda r: r.proposal is not None and not r.proposal_inadmissible

    metrics: dict[str, Any] = {"n": len(records)}
    definitions = {
        "epistemic_violation_rate": ("epistemic_violation", executed),
        "environmental_violation_rate": ("environmental_violation", executed),
        "governance_violation_rate": ("governance_violation", executed),
        "full_admissibility_compliance": ("full_admissibility_compliance", executed),
        "open_rate": ("opened", None),
        "unsafe_open_rate": ("unsafe_open", lambda r: r.opened),
        "fallback_burden": ("fallback", None),
        "halt_rate": ("halted", None),
        "task_completion_rate": ("task_completed", None),
        "agent_substitution_violation_rate": ("agent_substitution_violation", lambda r: r.opened),
        "proposal_intervention_rate": ("proposal_intervened", proposed),
        "inadmissible_proposal_blocking_rate": ("inadmissible_proposal_blocked", inadmissible_proposal),
        "admissible_proposal_override_rate": ("admissible_proposal_override", admissible_proposal),
        "trace_replay_success_rate": ("trace_replay_success", lambda r: r.trace_replay_success is not None),
    }
    for metric_name, (field, denominator_filter) in definitions.items():
        value, numerator, denominator = _rate(records, field, denominator_filter)
        metrics[metric_name] = value
        metrics[f"{metric_name}_numerator"] = numerator
        metrics[f"{metric_name}_denominator"] = denominator

    if records:
        metrics["mean_utility"] = mean(record.utility for record in records)
        finite_regrets = [
            record.constrained_regret
            for record in records
            if math.isfinite(record.constrained_regret)
        ]
        metrics["mean_constrained_regret"] = (
            mean(finite_regrets) if finite_regrets else float("nan")
        )
        metrics["mean_constrained_regret_denominator"] = len(finite_regrets)
        runtimes = sorted(record.authorization_ns / 1_000_000 for record in records)
        metrics["mean_authorization_ms"] = mean(runtimes)
        metrics["median_authorization_ms"] = median(runtimes)
        metrics["p95_authorization_ms"] = percentile(runtimes, 0.95)
    else:
        metrics.update(
            {
                "mean_utility": float("nan"),
                "mean_constrained_regret": float("nan"),
                "mean_constrained_regret_denominator": 0,
                "mean_authorization_ms": float("nan"),
                "median_authorization_ms": float("nan"),
                "p95_authorization_ms": float("nan"),
            }
        )
    action_counts: dict[str, int] = {}
    for record in records:
        key = record.selected_action or "halt"
        action_counts[key] = action_counts.get(key, 0) + 1
    metrics["action_counts"] = action_counts
    return metrics


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boot = []
    n = len(values)
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(mean(sample))
    boot.sort()
    alpha = 1 - confidence
    return (
        percentile(boot, alpha / 2),
        percentile(boot, 1 - alpha / 2),
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    if fraction <= 0:
        return values[0]
    if fraction >= 1:
        return values[-1]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def write_records_csv(records: Sequence[DecisionRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].to_dict()))
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)


def _json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats to JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
