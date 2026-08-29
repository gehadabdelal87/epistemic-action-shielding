"""Finite S5 epistemic models and model checking."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

from .formulas import And, Atom, Formula, Knows, Not, Or, Top, formula_from_dict


class ModelValidationError(ValueError):
    """Raised when an epistemic model violates required structural conditions."""


@dataclass(frozen=True, slots=True)
class Counterexample:
    formula: Formula
    world: str
    reason: str
    children: tuple["Counterexample", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula.to_dict(),
            "world": self.world,
            "reason": self.reason,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(slots=True)
class ModelChecker:
    """Per-model cached satisfaction evaluator."""

    model: "EpistemicModel"
    _cache: dict[tuple[str, Formula], bool] = field(default_factory=dict)

    def satisfies(self, world: str, formula: Formula) -> bool:
        key = (world, formula)
        if key in self._cache:
            return self._cache[key]
        result = self._evaluate(world, formula)
        self._cache[key] = result
        return result

    def _evaluate(self, world: str, formula: Formula) -> bool:
        if world not in self.model.worlds:
            raise KeyError(f"Unknown world: {world}")
        if isinstance(formula, Top):
            return True
        if isinstance(formula, Atom):
            return formula.name in self.model.valuation[world]
        if isinstance(formula, Not):
            return not self.satisfies(world, formula.inner)
        if isinstance(formula, And):
            return all(self.satisfies(world, part) for part in formula.parts)
        if isinstance(formula, Or):
            return any(self.satisfies(world, part) for part in formula.parts)
        if isinstance(formula, Knows):
            if formula.agent not in self.model.agents:
                raise KeyError(f"Unknown agent: {formula.agent}")
            return all(
                self.satisfies(other, formula.inner)
                for other in self.model.relations[formula.agent][world]
            )
        raise TypeError(f"Unsupported formula: {formula!r}")

    def counterexample(self, world: str, formula: Formula) -> Counterexample | None:
        """Return a compact witness explaining why ``formula`` is false."""
        if self.satisfies(world, formula):
            return None
        if isinstance(formula, Atom):
            return Counterexample(formula, world, "atom_false")
        if isinstance(formula, Top):
            return None
        if isinstance(formula, Not):
            return Counterexample(formula, world, "negated_formula_true")
        if isinstance(formula, And):
            failed = tuple(
                child
                for part in formula.parts
                if (child := self.counterexample(world, part)) is not None
            )
            return Counterexample(formula, world, "conjunct_failed", failed)
        if isinstance(formula, Or):
            failed = tuple(
                child
                for part in formula.parts
                if (child := self.counterexample(world, part)) is not None
            )
            return Counterexample(formula, world, "all_disjuncts_failed", failed)
        if isinstance(formula, Knows):
            for other in sorted(self.model.relations[formula.agent][world]):
                if not self.satisfies(other, formula.inner):
                    child = self.counterexample(other, formula.inner)
                    return Counterexample(
                        formula,
                        other,
                        f"accessible_counterexample_for_{formula.agent}",
                        (child,) if child is not None else (),
                    )
        return Counterexample(formula, world, "formula_false")


@dataclass(frozen=True, slots=True)
class EpistemicModel:
    worlds: tuple[str, ...]
    agents: tuple[str, ...]
    propositions: tuple[str, ...]
    valuation: Mapping[str, frozenset[str]]
    relations: Mapping[str, Mapping[str, frozenset[str]]]
    frame_class: str = "S5"

    def __post_init__(self) -> None:
        object.__setattr__(self, "worlds", tuple(dict.fromkeys(self.worlds)))
        object.__setattr__(self, "agents", tuple(dict.fromkeys(self.agents)))
        object.__setattr__(self, "propositions", tuple(dict.fromkeys(self.propositions)))
        object.__setattr__(
            self,
            "valuation",
            {world: frozenset(values) for world, values in self.valuation.items()},
        )
        object.__setattr__(
            self,
            "relations",
            {
                agent: {
                    world: frozenset(targets)
                    for world, targets in relation.items()
                }
                for agent, relation in self.relations.items()
            },
        )

    @property
    def model_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    @property
    def edge_count(self) -> int:
        return sum(
            len(targets)
            for agent in self.agents
            for targets in self.relations[agent].values()
        )

    def checker(self) -> ModelChecker:
        return ModelChecker(self)

    def satisfies(self, world: str, formula: Formula) -> bool:
        return self.checker().satisfies(world, formula)

    def validate(self, require_s5: bool = True) -> None:
        if not self.worlds:
            raise ModelValidationError("A model must contain at least one world.")
        world_set = set(self.worlds)
        agent_set = set(self.agents)
        proposition_set = set(self.propositions)
        if len(world_set) != len(self.worlds):
            raise ModelValidationError("World identifiers must be unique.")
        if len(agent_set) != len(self.agents):
            raise ModelValidationError("Agent identifiers must be unique.")
        if len(proposition_set) != len(self.propositions):
            raise ModelValidationError("Proposition identifiers must be unique.")
        if set(self.valuation) != world_set:
            raise ModelValidationError("Valuation must define every world exactly once.")
        for world, values in self.valuation.items():
            unknown = set(values) - proposition_set
            if unknown:
                raise ModelValidationError(
                    f"Valuation at {world} contains unknown propositions: {sorted(unknown)}"
                )
        if set(self.relations) != agent_set:
            raise ModelValidationError("A relation must be provided for every agent.")
        for agent, relation in self.relations.items():
            if set(relation) != world_set:
                raise ModelValidationError(
                    f"Relation for {agent} must define successors for every world."
                )
            for world, targets in relation.items():
                unknown = set(targets) - world_set
                if unknown:
                    raise ModelValidationError(
                        f"Relation {agent}@{world} references unknown worlds: {sorted(unknown)}"
                    )
                if not targets:
                    raise ModelValidationError(
                        f"Relation {agent}@{world} may not have an empty information cell."
                    )
            if require_s5 or self.frame_class.upper() == "S5":
                self._validate_equivalence(agent, relation)

    def _validate_equivalence(
        self, agent: str, relation: Mapping[str, frozenset[str]]
    ) -> None:
        for world in self.worlds:
            if world not in relation[world]:
                raise ModelValidationError(
                    f"Relation for {agent} is not reflexive at {world}."
                )
        for world in self.worlds:
            for other in relation[world]:
                if world not in relation[other]:
                    raise ModelValidationError(
                        f"Relation for {agent} is not symmetric: {world}->{other}."
                    )
        for world in self.worlds:
            for middle in relation[world]:
                if not relation[middle].issubset(relation[world]):
                    raise ModelValidationError(
                        f"Relation for {agent} is not transitive from {world} via {middle}."
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "worlds": list(self.worlds),
            "agents": list(self.agents),
            "propositions": list(self.propositions),
            "valuation": {
                world: sorted(self.valuation[world]) for world in sorted(self.worlds)
            },
            "relations": {
                agent: {
                    world: sorted(self.relations[agent][world])
                    for world in sorted(self.worlds)
                }
                for agent in sorted(self.agents)
            },
            "frame_class": self.frame_class,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpistemicModel":
        model = cls(
            worlds=tuple(str(x) for x in data["worlds"]),
            agents=tuple(str(x) for x in data["agents"]),
            propositions=tuple(str(x) for x in data["propositions"]),
            valuation={
                str(world): frozenset(str(p) for p in values)
                for world, values in data["valuation"].items()
            },
            relations={
                str(agent): {
                    str(world): frozenset(str(target) for target in targets)
                    for world, targets in relation.items()
                }
                for agent, relation in data["relations"].items()
            },
            frame_class=str(data.get("frame_class", "S5")),
        )
        model.validate()
        return model

    def with_relation(
        self, agent: str, relation: Mapping[str, Iterable[str]]
    ) -> "EpistemicModel":
        if agent not in self.agents:
            raise KeyError(agent)
        relations = {
            name: {world: frozenset(targets) for world, targets in rel.items()}
            for name, rel in self.relations.items()
        }
        relations[agent] = {
            world: frozenset(targets) for world, targets in relation.items()
        }
        model = EpistemicModel(
            worlds=self.worlds,
            agents=self.agents,
            propositions=self.propositions,
            valuation=self.valuation,
            relations=relations,
            frame_class=self.frame_class,
        )
        model.validate()
        return model

    def with_valuation(
        self, valuation: Mapping[str, Iterable[str]]
    ) -> "EpistemicModel":
        model = EpistemicModel(
            worlds=self.worlds,
            agents=self.agents,
            propositions=self.propositions,
            valuation={world: frozenset(values) for world, values in valuation.items()},
            relations=self.relations,
            frame_class=self.frame_class,
        )
        model.validate()
        return model


@dataclass(frozen=True, slots=True)
class PointedState:
    model: EpistemicModel
    world: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.world not in self.model.worlds:
            raise ModelValidationError(
                f"Designated world {self.world!r} is not in the model."
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def state_id(self) -> str:
        payload = {
            "model_id": self.model.model_id,
            "world": self.world,
            "metadata": canonicalize(self.metadata),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "world": self.world,
            "metadata": canonicalize(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PointedState":
        return cls(
            model=EpistemicModel.from_dict(data["model"]),
            world=str(data["world"]),
            metadata=dict(data.get("metadata", {})),
        )

    def with_model(self, model: EpistemicModel, world: str | None = None) -> "PointedState":
        return PointedState(model, world or self.world, self.metadata)

    def with_metadata(self, **updates: Any) -> "PointedState":
        metadata = dict(self.metadata)
        metadata.update(updates)
        return PointedState(self.model, self.world, metadata)


def canonicalize(value: Any) -> Any:
    if isinstance(value, Formula):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): canonicalize(val) for key, val in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(canonicalize(item) for item in value)
    if hasattr(value, "to_dict"):
        return canonicalize(value.to_dict())
    return value


def equivalence_relation_from_partition(
    worlds: Iterable[str], blocks: Iterable[Iterable[str]]
) -> dict[str, frozenset[str]]:
    world_set = set(worlds)
    relation: dict[str, frozenset[str]] = {}
    seen: set[str] = set()
    for block_raw in blocks:
        block = frozenset(block_raw)
        if not block:
            continue
        if not block.issubset(world_set):
            raise ModelValidationError("Partition contains an unknown world.")
        if seen.intersection(block):
            raise ModelValidationError("Partition blocks overlap.")
        seen.update(block)
        for world in block:
            relation[world] = block
    if seen != world_set:
        missing = world_set - seen
        raise ModelValidationError(f"Partition does not cover worlds: {sorted(missing)}")
    return relation
