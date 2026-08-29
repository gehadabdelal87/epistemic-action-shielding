"""Deterministic replay of stored authorization traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .actions import ActionLibrary
from .governance import GovernancePolicy
from .model import PointedState
from .shield import DecisionMode, EASDecisionEngine
from .trace import AuthorizationTraceEntry, DecisionTrace


@dataclass(frozen=True, slots=True)
class ReplayResult:
    success: bool
    differences: Mapping[str, Any]
    recomputed_entry: AuthorizationTraceEntry


def replay_authorization(entry: AuthorizationTraceEntry) -> ReplayResult:
    bundle = entry.replay_bundle
    state = PointedState.from_dict(bundle["authorization_state"])
    actions = ActionLibrary.from_dict(bundle["action_library"])
    policy = GovernancePolicy.from_dict(bundle["governance_policy"])
    trace = DecisionTrace()
    engine = EASDecisionEngine()
    outcome = engine.authorize(
        state=state,
        action_library=actions,
        governance_policy=policy,
        trace=trace,
        mode=DecisionMode(str(bundle["mode"])),
        proposal=bundle.get("proposal"),
        utility_by_action={
            str(k): float(v)
            for k, v in bundle.get("utility_by_action", {}).items()
        },
        fallback_priority=tuple(bundle.get("fallback_priority", [])),
        random_seed=bundle.get("random_seed"),
        decision_id=entry.decision_id,
    )
    replayed = outcome.trace_entry
    fields = (
        "epistemically_permitted",
        "epistemically_blocked",
        "environmentally_permitted",
        "environmentally_blocked",
        "governance_permitted",
        "governance_rejected",
        "selected_action",
        "status",
    )
    differences: dict[str, Any] = {}
    for field_name in fields:
        original_value = getattr(entry, field_name)
        replay_value = getattr(replayed, field_name)
        if original_value != replay_value:
            differences[field_name] = {
                "original": original_value,
                "replayed": replay_value,
            }
    return ReplayResult(not differences, differences, replayed)
