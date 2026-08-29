"""Structured authorization and execution traces with deterministic replay data."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .model import canonicalize


@dataclass(frozen=True, slots=True)
class AuthorizationTraceEntry:
    decision_id: str
    prior_state_id: str
    authorization_state_id: str
    prior_model_id: str
    authorization_model_id: str
    event_record: Mapping[str, Any]
    revision_records: tuple[Mapping[str, Any], ...]
    action_library_id: str
    governance_policy_id: str
    mode: str
    proposal: str | None
    epistemically_permitted: tuple[str, ...]
    epistemically_blocked: tuple[str, ...]
    epistemic_witnesses: Mapping[str, Any]
    environmentally_permitted: tuple[str, ...]
    environmentally_blocked: tuple[str, ...]
    governance_permitted: tuple[str, ...]
    governance_rejected: Mapping[str, Any]
    selected_action: str | None
    status: str
    fallback_priority: tuple[str, ...]
    utility_by_action: Mapping[str, float]
    random_seed: int | None
    replay_bundle: Mapping[str, Any]

    @property
    def trace_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "decision_id": self.decision_id,
                "prior_state_id": self.prior_state_id,
                "authorization_state_id": self.authorization_state_id,
                "prior_model_id": self.prior_model_id,
                "authorization_model_id": self.authorization_model_id,
                "event_record": dict(self.event_record),
                "revision_records": list(self.revision_records),
                "action_library_id": self.action_library_id,
                "governance_policy_id": self.governance_policy_id,
                "mode": self.mode,
                "proposal": self.proposal,
                "epistemically_permitted": list(self.epistemically_permitted),
                "epistemically_blocked": list(self.epistemically_blocked),
                "epistemic_witnesses": dict(self.epistemic_witnesses),
                "environmentally_permitted": list(self.environmentally_permitted),
                "environmentally_blocked": list(self.environmentally_blocked),
                "governance_permitted": list(self.governance_permitted),
                "governance_rejected": dict(self.governance_rejected),
                "selected_action": self.selected_action,
                "status": self.status,
                "fallback_priority": list(self.fallback_priority),
                "utility_by_action": dict(self.utility_by_action),
                "random_seed": self.random_seed,
                "replay_bundle": dict(self.replay_bundle),
            }
        )


@dataclass(frozen=True, slots=True)
class ExecutionTraceEntry:
    decision_id: str
    action: str | None
    transition_id: str | None
    status: str
    prior_state_id: str
    resulting_state_id: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "decision_id": self.decision_id,
                "action": self.action,
                "transition_id": self.transition_id,
                "status": self.status,
                "prior_state_id": self.prior_state_id,
                "resulting_state_id": self.resulting_state_id,
                "details": dict(self.details),
            }
        )


@dataclass(slots=True)
class DecisionTrace:
    authorization_entries: list[AuthorizationTraceEntry] = field(default_factory=list)
    execution_entries: list[ExecutionTraceEntry] = field(default_factory=list)

    def append_authorization(self, entry: AuthorizationTraceEntry) -> None:
        self.authorization_entries.append(entry)

    def append_execution(self, entry: ExecutionTraceEntry) -> None:
        self.execution_entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorization_entries": [entry.to_dict() for entry in self.authorization_entries],
            "execution_entries": [entry.to_dict() for entry in self.execution_entries],
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "DecisionTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        trace = cls()
        for item in data.get("authorization_entries", []):
            trace.authorization_entries.append(authorization_entry_from_dict(item))
        for item in data.get("execution_entries", []):
            trace.execution_entries.append(execution_entry_from_dict(item))
        return trace


def authorization_entry_from_dict(data: Mapping[str, Any]) -> AuthorizationTraceEntry:
    return AuthorizationTraceEntry(
        decision_id=str(data["decision_id"]),
        prior_state_id=str(data["prior_state_id"]),
        authorization_state_id=str(data["authorization_state_id"]),
        prior_model_id=str(data["prior_model_id"]),
        authorization_model_id=str(data["authorization_model_id"]),
        event_record=dict(data.get("event_record", {})),
        revision_records=tuple(dict(item) for item in data.get("revision_records", [])),
        action_library_id=str(data["action_library_id"]),
        governance_policy_id=str(data["governance_policy_id"]),
        mode=str(data["mode"]),
        proposal=data.get("proposal"),
        epistemically_permitted=tuple(data.get("epistemically_permitted", [])),
        epistemically_blocked=tuple(data.get("epistemically_blocked", [])),
        epistemic_witnesses=dict(data.get("epistemic_witnesses", {})),
        environmentally_permitted=tuple(data.get("environmentally_permitted", [])),
        environmentally_blocked=tuple(data.get("environmentally_blocked", [])),
        governance_permitted=tuple(data.get("governance_permitted", [])),
        governance_rejected=dict(data.get("governance_rejected", {})),
        selected_action=data.get("selected_action"),
        status=str(data["status"]),
        fallback_priority=tuple(data.get("fallback_priority", [])),
        utility_by_action={str(k): float(v) for k, v in data.get("utility_by_action", {}).items()},
        random_seed=data.get("random_seed"),
        replay_bundle=dict(data.get("replay_bundle", {})),
    )


def execution_entry_from_dict(data: Mapping[str, Any]) -> ExecutionTraceEntry:
    return ExecutionTraceEntry(
        decision_id=str(data["decision_id"]),
        action=data.get("action"),
        transition_id=data.get("transition_id"),
        status=str(data["status"]),
        prior_state_id=str(data["prior_state_id"]),
        resulting_state_id=str(data["resulting_state_id"]),
        details=dict(data.get("details", {})),
    )
