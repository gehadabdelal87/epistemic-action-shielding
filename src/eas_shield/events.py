"""Dynamic epistemic event models and product update."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .formulas import Atom, Formula, formula_from_dict
from .model import EpistemicModel, ModelChecker, ModelValidationError, PointedState, equivalence_relation_from_partition


class EventValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EventModel:
    events: tuple[str, ...]
    agents: tuple[str, ...]
    preconditions: Mapping[str, Formula]
    postconditions: Mapping[str, Mapping[str, Formula]]
    indistinguishability: Mapping[str, Mapping[str, frozenset[str]]]
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(dict.fromkeys(self.events)))
        object.__setattr__(self, "agents", tuple(dict.fromkeys(self.agents)))
        object.__setattr__(self, "preconditions", dict(self.preconditions))
        object.__setattr__(
            self,
            "postconditions",
            {
                event: dict(assignments)
                for event, assignments in self.postconditions.items()
            },
        )
        object.__setattr__(
            self,
            "indistinguishability",
            {
                agent: {
                    event: frozenset(targets)
                    for event, targets in relation.items()
                }
                for agent, relation in self.indistinguishability.items()
            },
        )

    @property
    def event_model_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def validate(self) -> None:
        event_set = set(self.events)
        if not event_set:
            raise EventValidationError("An event model must contain at least one event.")
        if set(self.preconditions) != event_set:
            raise EventValidationError("Each event must have exactly one precondition.")
        unknown_post_events = set(self.postconditions) - event_set
        if unknown_post_events:
            raise EventValidationError(
                f"Postconditions reference unknown events: {sorted(unknown_post_events)}"
            )
        if set(self.indistinguishability) != set(self.agents):
            raise EventValidationError(
                "An indistinguishability relation is required for every agent."
            )
        for agent, relation in self.indistinguishability.items():
            if set(relation) != event_set:
                raise EventValidationError(
                    f"Event relation for {agent} must define every event."
                )
            for event, targets in relation.items():
                if not targets or not set(targets).issubset(event_set):
                    raise EventValidationError(
                        f"Invalid event cell for {agent}@{event}: {sorted(targets)}"
                    )
                if event not in targets:
                    raise EventValidationError(
                        f"Event relation for {agent} is not reflexive at {event}."
                    )
            for event, targets in relation.items():
                for other in targets:
                    if event not in relation[other]:
                        raise EventValidationError(
                            f"Event relation for {agent} is not symmetric."
                        )
                    if not relation[other].issubset(targets):
                        raise EventValidationError(
                            f"Event relation for {agent} is not transitive."
                        )

    def postcondition(self, event: str, proposition: str) -> Formula:
        return self.postconditions.get(event, {}).get(proposition, Atom(proposition))

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": list(self.events),
            "agents": list(self.agents),
            "preconditions": {
                event: formula.to_dict()
                for event, formula in sorted(self.preconditions.items())
            },
            "postconditions": {
                event: {
                    proposition: formula.to_dict()
                    for proposition, formula in sorted(assignments.items())
                }
                for event, assignments in sorted(self.postconditions.items())
            },
            "indistinguishability": {
                agent: {
                    event: sorted(targets)
                    for event, targets in sorted(relation.items())
                }
                for agent, relation in sorted(self.indistinguishability.items())
            },
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventModel":
        model = cls(
            events=tuple(str(x) for x in data["events"]),
            agents=tuple(str(x) for x in data["agents"]),
            preconditions={
                str(event): formula_from_dict(formula)
                for event, formula in data["preconditions"].items()
            },
            postconditions={
                str(event): {
                    str(prop): formula_from_dict(formula)
                    for prop, formula in assignments.items()
                }
                for event, assignments in data.get("postconditions", {}).items()
            },
            indistinguishability={
                str(agent): {
                    str(event): frozenset(str(target) for target in targets)
                    for event, targets in relation.items()
                }
                for agent, relation in data["indistinguishability"].items()
            },
            version=str(data.get("version", "1")),
        )
        model.validate()
        return model


def product_update(
    state: PointedState,
    event_model: EventModel,
    designated_event: str,
) -> PointedState:
    """Apply a DEL product update and return the updated pointed state."""
    event_model.validate()
    state.model.validate()
    if tuple(event_model.agents) != tuple(state.model.agents):
        if set(event_model.agents) != set(state.model.agents):
            raise EventValidationError(
                "Event and epistemic models must reference the same agents."
            )
    if designated_event not in event_model.events:
        raise EventValidationError(f"Unknown designated event: {designated_event}")

    checker = state.model.checker()
    if not checker.satisfies(
        state.world, event_model.preconditions[designated_event]
    ):
        raise EventValidationError(
            f"Precondition for event {designated_event!r} does not hold."
        )

    pairs: list[tuple[str, str]] = []
    for world in state.model.worlds:
        for event in event_model.events:
            if checker.satisfies(world, event_model.preconditions[event]):
                pairs.append((world, event))

    world_name = {(world, event): f"{world}::{event}" for world, event in pairs}
    updated_worlds = tuple(world_name[pair] for pair in pairs)
    if not updated_worlds:
        raise EventValidationError("Product update produced an empty model.")

    updated_valuation: dict[str, frozenset[str]] = {}
    for world, event in pairs:
        truths = {
            proposition
            for proposition in state.model.propositions
            if checker.satisfies(
                world, event_model.postcondition(event, proposition)
            )
        }
        updated_valuation[world_name[(world, event)]] = frozenset(truths)

    updated_relations: dict[str, dict[str, frozenset[str]]] = {}
    pair_set = set(pairs)
    for agent in state.model.agents:
        relation: dict[str, frozenset[str]] = {}
        for world, event in pairs:
            targets: set[str] = set()
            for other in state.model.relations[agent][world]:
                for other_event in event_model.indistinguishability[agent][event]:
                    if (other, other_event) in pair_set:
                        targets.add(world_name[(other, other_event)])
            relation[world_name[(world, event)]] = frozenset(targets)
        updated_relations[agent] = relation

    updated_model = EpistemicModel(
        worlds=updated_worlds,
        agents=state.model.agents,
        propositions=state.model.propositions,
        valuation=updated_valuation,
        relations=updated_relations,
        frame_class=state.model.frame_class,
    )
    updated_model.validate()
    designated_world = world_name[(state.world, designated_event)]
    metadata = dict(state.metadata)
    metadata.update(
        {
            "last_event_id": designated_event,
            "last_event_model_id": event_model.event_model_id,
        }
    )
    return PointedState(updated_model, designated_world, metadata)


def public_announcement_event_model(
    agents: tuple[str, ...],
    proposition: Formula,
    event_name: str = "announce",
) -> EventModel:
    """Build a one-event public-announcement model."""
    indistinguishability = {
        agent: equivalence_relation_from_partition((event_name,), ((event_name,),))
        for agent in agents
    }
    model = EventModel(
        events=(event_name,),
        agents=agents,
        preconditions={event_name: proposition},
        postconditions={},
        indistinguishability=indistinguishability,
    )
    model.validate()
    return model
