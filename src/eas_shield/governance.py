"""Machine-checkable operational governance constraints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Protocol

from .actions import ActionSchema
from .model import PointedState


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    constraint_id: str
    passed: bool
    reason: str
    evidence: Mapping[str, Any]
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "policy_version": self.policy_version,
        }


class GovernanceConstraint(Protocol):
    constraint_id: str
    version: str

    def evaluate(
        self,
        state: PointedState,
        action: ActionSchema,
        trace_context: Mapping[str, Any],
    ) -> ConstraintResult: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SourceReliabilityConstraint:
    threshold: float
    applies_to: frozenset[str] = frozenset()
    constraint_id: str = "source_reliability"
    version: str = "1"

    def evaluate(self, state: PointedState, action: ActionSchema, trace_context: Mapping[str, Any]) -> ConstraintResult:
        if self.applies_to and action.name not in self.applies_to:
            return ConstraintResult(self.constraint_id, True, "not_applicable", {}, self.version)
        value = float(state.metadata.get("source_reliability", 0.0))
        passed = value >= self.threshold
        return ConstraintResult(
            self.constraint_id,
            passed,
            "passed" if passed else "source_reliability_below_threshold",
            {"value": value, "threshold": self.threshold},
            self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "source_reliability",
            "threshold": self.threshold,
            "applies_to": sorted(self.applies_to),
            "constraint_id": self.constraint_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ObservationQualityConstraint:
    threshold: float
    applies_to: frozenset[str] = frozenset()
    constraint_id: str = "observation_quality"
    version: str = "1"

    def evaluate(self, state: PointedState, action: ActionSchema, trace_context: Mapping[str, Any]) -> ConstraintResult:
        if self.applies_to and action.name not in self.applies_to:
            return ConstraintResult(self.constraint_id, True, "not_applicable", {}, self.version)
        value = float(state.metadata.get("observation_quality", 0.0))
        passed = value >= self.threshold
        return ConstraintResult(
            self.constraint_id,
            passed,
            "passed" if passed else "observation_quality_below_threshold",
            {"value": value, "threshold": self.threshold},
            self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "observation_quality",
            "threshold": self.threshold,
            "applies_to": sorted(self.applies_to),
            "constraint_id": self.constraint_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceConstraint:
    applies_to: frozenset[str] = frozenset()
    constraint_id: str = "known_provenance"
    version: str = "1"

    def evaluate(self, state: PointedState, action: ActionSchema, trace_context: Mapping[str, Any]) -> ConstraintResult:
        if self.applies_to and action.name not in self.applies_to:
            return ConstraintResult(self.constraint_id, True, "not_applicable", {}, self.version)
        value = bool(state.metadata.get("provenance_known", False))
        return ConstraintResult(
            self.constraint_id,
            value,
            "passed" if value else "provenance_unknown",
            {"provenance_known": value},
            self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "known_provenance",
            "applies_to": sorted(self.applies_to),
            "constraint_id": self.constraint_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class MaxRiskConstraint:
    threshold: float
    applies_to: frozenset[str] = frozenset()
    constraint_id: str = "max_action_risk"
    version: str = "1"

    def evaluate(self, state: PointedState, action: ActionSchema, trace_context: Mapping[str, Any]) -> ConstraintResult:
        if self.applies_to and action.name not in self.applies_to:
            return ConstraintResult(self.constraint_id, True, "not_applicable", {}, self.version)
        passed = action.risk <= self.threshold
        return ConstraintResult(
            self.constraint_id,
            passed,
            "passed" if passed else "action_risk_above_threshold",
            {"value": action.risk, "threshold": self.threshold},
            self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "max_action_risk",
            "threshold": self.threshold,
            "applies_to": sorted(self.applies_to),
            "constraint_id": self.constraint_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class AuthorizationConstraint:
    metadata_key: str = "authorized_actions"
    applies_to: frozenset[str] = frozenset()
    constraint_id: str = "actor_authorized"
    version: str = "1"

    def evaluate(self, state: PointedState, action: ActionSchema, trace_context: Mapping[str, Any]) -> ConstraintResult:
        if self.applies_to and action.name not in self.applies_to:
            return ConstraintResult(self.constraint_id, True, "not_applicable", {}, self.version)
        authorized = state.metadata.get(self.metadata_key, ())
        if isinstance(authorized, Mapping):
            allowed = bool(authorized.get(action.actor, {}).get(action.name, False))
        else:
            allowed = action.name in set(authorized)
        return ConstraintResult(
            self.constraint_id,
            allowed,
            "passed" if allowed else "actor_not_authorized",
            {"actor": action.actor, "action": action.name},
            self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "authorization",
            "metadata_key": self.metadata_key,
            "applies_to": sorted(self.applies_to),
            "constraint_id": self.constraint_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ReviewCompletedConstraint:
    applies_to: frozenset[str]
    metadata_key: str = "human_review_completed"
    constraint_id: str = "human_review_completed"
    version: str = "1"

    def evaluate(self, state: PointedState, action: ActionSchema, trace_context: Mapping[str, Any]) -> ConstraintResult:
        if action.name not in self.applies_to:
            return ConstraintResult(self.constraint_id, True, "not_applicable", {}, self.version)
        completed = bool(state.metadata.get(self.metadata_key, False))
        return ConstraintResult(
            self.constraint_id,
            completed,
            "passed" if completed else "human_review_required",
            {"completed": completed},
            self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "review_completed",
            "metadata_key": self.metadata_key,
            "applies_to": sorted(self.applies_to),
            "constraint_id": self.constraint_id,
            "version": self.version,
        }


ConstraintType = (
    SourceReliabilityConstraint
    | ObservationQualityConstraint
    | ProvenanceConstraint
    | MaxRiskConstraint
    | AuthorizationConstraint
    | ReviewCompletedConstraint
)


@dataclass(frozen=True, slots=True)
class GovernanceEvaluation:
    passed: bool
    results: tuple[ConstraintResult, ...]

    @property
    def failed_constraints(self) -> tuple[str, ...]:
        return tuple(result.constraint_id for result in self.results if not result.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class GovernancePolicy:
    constraints: tuple[ConstraintType, ...]
    version: str = "1"

    @property
    def policy_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def evaluate(
        self,
        state: PointedState,
        action: ActionSchema,
        trace_context: Mapping[str, Any] | None = None,
    ) -> GovernanceEvaluation:
        context = trace_context or {}
        results = tuple(
            constraint.evaluate(state, action, context)
            for constraint in self.constraints
        )
        return GovernanceEvaluation(all(result.passed for result in results), results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernancePolicy":
        constraints: list[ConstraintType] = []
        for item in data.get("constraints", []):
            kind = item["type"]
            applies = frozenset(str(x) for x in item.get("applies_to", []))
            common = {
                "constraint_id": str(item.get("constraint_id", kind)),
                "version": str(item.get("version", "1")),
            }
            if kind == "source_reliability":
                constraints.append(SourceReliabilityConstraint(float(item["threshold"]), applies, **common))
            elif kind == "observation_quality":
                constraints.append(ObservationQualityConstraint(float(item["threshold"]), applies, **common))
            elif kind == "known_provenance":
                constraints.append(ProvenanceConstraint(applies, **common))
            elif kind == "max_action_risk":
                constraints.append(MaxRiskConstraint(float(item["threshold"]), applies, **common))
            elif kind == "authorization":
                constraints.append(
                    AuthorizationConstraint(
                        metadata_key=str(item.get("metadata_key", "authorized_actions")),
                        applies_to=applies,
                        **common,
                    )
                )
            elif kind == "review_completed":
                constraints.append(
                    ReviewCompletedConstraint(
                        applies_to=applies,
                        metadata_key=str(item.get("metadata_key", "human_review_completed")),
                        **common,
                    )
                )
            else:
                raise ValueError(f"Unknown governance constraint type: {kind}")
        return cls(tuple(constraints), version=str(data.get("version", "1")))
