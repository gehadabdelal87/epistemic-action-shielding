"""Evidence admission and information-partition refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .formulas import Formula, Not
from .model import EpistemicModel, PointedState


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    recipient: str
    formula: Formula
    observed_value: bool
    confidence: float
    source: str
    source_reliability: float
    quality: float
    provenance_known: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("confidence", self.confidence),
            ("source_reliability", self.source_reliability),
            ("quality", self.quality),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content(self) -> Formula:
        return self.formula if self.observed_value else Not(self.formula)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "recipient": self.recipient,
            "formula": self.formula.to_dict(),
            "observed_value": self.observed_value,
            "confidence": self.confidence,
            "source": self.source,
            "source_reliability": self.source_reliability,
            "quality": self.quality,
            "provenance_known": self.provenance_known,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    confidence_threshold: float = 0.7
    reliability_threshold: float = 0.7
    quality_threshold: float = 0.7
    require_known_provenance: bool = True
    version: str = "1"

    def admits(self, observation: Observation) -> tuple[bool, tuple[str, ...]]:
        failures: list[str] = []
        if observation.confidence < self.confidence_threshold:
            failures.append("confidence_below_threshold")
        if observation.source_reliability < self.reliability_threshold:
            failures.append("source_reliability_below_threshold")
        if observation.quality < self.quality_threshold:
            failures.append("observation_quality_below_threshold")
        if self.require_known_provenance and not observation.provenance_known:
            failures.append("provenance_unknown")
        return (not failures, tuple(failures))

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "reliability_threshold": self.reliability_threshold,
            "quality_threshold": self.quality_threshold,
            "require_known_provenance": self.require_known_provenance,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    observation: Observation
    admitted: bool
    failures: tuple[str, ...]
    prior_model_id: str
    revised_model_id: str
    changed_edges: int
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "admitted": self.admitted,
            "failures": list(self.failures),
            "prior_model_id": self.prior_model_id,
            "revised_model_id": self.revised_model_id,
            "changed_edges": self.changed_edges,
            "policy_version": self.policy_version,
        }


def refine_information_partition(
    state: PointedState,
    observation: Observation,
    policy: EvidencePolicy,
) -> tuple[PointedState, RevisionRecord]:
    """Refine one agent's S5 partition by the truth value of admitted evidence.

    The operator intersects the existing equivalence relation with agreement on
    the observation content.  This preserves equivalence and does not consult
    an external oracle.
    """
    if observation.recipient not in state.model.agents:
        raise KeyError(f"Unknown observation recipient: {observation.recipient}")
    admitted, failures = policy.admits(observation)
    if not admitted:
        return state, RevisionRecord(
            observation=observation,
            admitted=False,
            failures=failures,
            prior_model_id=state.model.model_id,
            revised_model_id=state.model.model_id,
            changed_edges=0,
            policy_version=policy.version,
        )

    checker = state.model.checker()
    content = observation.content
    truth = {
        world: checker.satisfies(world, content) for world in state.model.worlds
    }
    agent = observation.recipient
    old_relation = state.model.relations[agent]
    new_relation: dict[str, frozenset[str]] = {}
    changed_edges = 0
    for world in state.model.worlds:
        targets = frozenset(
            other
            for other in old_relation[world]
            if truth[other] == truth[world]
        )
        # Reflexivity ensures this cell cannot be empty.
        new_relation[world] = targets
        changed_edges += len(old_relation[world].symmetric_difference(targets))

    revised_model = state.model.with_relation(agent, new_relation)
    metadata = dict(state.metadata)
    metadata.update(
        {
            "last_observation_id": observation.observation_id,
            "last_evidence_policy_version": policy.version,
            "last_observation_admitted": True,
            "source_reliability": observation.source_reliability,
            "observation_quality": observation.quality,
            "provenance_known": observation.provenance_known,
            "confidence": observation.confidence,
        }
    )
    revised_state = PointedState(revised_model, state.world, metadata)
    return revised_state, RevisionRecord(
        observation=observation,
        admitted=True,
        failures=(),
        prior_model_id=state.model.model_id,
        revised_model_id=revised_model.model_id,
        changed_edges=changed_edges,
        policy_version=policy.version,
    )
