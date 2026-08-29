"""Epistemic formula language used by EAS.

The classes in this module are immutable, hashable, and serializable.  They are
kept deliberately small so the model checker can cache subformula results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypeAlias


class Formula:
    """Base class for all formulas."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def modal_depth(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Top(Formula):
    def to_dict(self) -> dict[str, Any]:
        return {"type": "top"}

    @property
    def size(self) -> int:
        return 1

    @property
    def modal_depth(self) -> int:
        return 0

    def __str__(self) -> str:
        return "⊤"


@dataclass(frozen=True, slots=True)
class Atom(Formula):
    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Atomic proposition names must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "atom", "name": self.name}

    @property
    def size(self) -> int:
        return 1

    @property
    def modal_depth(self) -> int:
        return 0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Not(Formula):
    inner: Formula

    def to_dict(self) -> dict[str, Any]:
        return {"type": "not", "inner": self.inner.to_dict()}

    @property
    def size(self) -> int:
        return 1 + self.inner.size

    @property
    def modal_depth(self) -> int:
        return self.inner.modal_depth

    def __str__(self) -> str:
        return f"¬{parenthesize(self.inner)}"


@dataclass(frozen=True, slots=True)
class And(Formula):
    parts: tuple[Formula, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("And requires at least two operands.")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "and", "parts": [part.to_dict() for part in self.parts]}

    @property
    def size(self) -> int:
        return 1 + sum(part.size for part in self.parts)

    @property
    def modal_depth(self) -> int:
        return max(part.modal_depth for part in self.parts)

    def __str__(self) -> str:
        return " ∧ ".join(parenthesize(part) for part in self.parts)


@dataclass(frozen=True, slots=True)
class Or(Formula):
    parts: tuple[Formula, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("Or requires at least two operands.")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "or", "parts": [part.to_dict() for part in self.parts]}

    @property
    def size(self) -> int:
        return 1 + sum(part.size for part in self.parts)

    @property
    def modal_depth(self) -> int:
        return max(part.modal_depth for part in self.parts)

    def __str__(self) -> str:
        return " ∨ ".join(parenthesize(part) for part in self.parts)


@dataclass(frozen=True, slots=True)
class Knows(Formula):
    agent: str
    inner: Formula

    def __post_init__(self) -> None:
        if not self.agent or not self.agent.strip():
            raise ValueError("Agent identifiers must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "knows",
            "agent": self.agent,
            "inner": self.inner.to_dict(),
        }

    @property
    def size(self) -> int:
        return 1 + self.inner.size

    @property
    def modal_depth(self) -> int:
        return 1 + self.inner.modal_depth

    def __str__(self) -> str:
        return f"K_{self.agent} {parenthesize(self.inner)}"


FormulaLike: TypeAlias = Formula


def conjunction(*parts: Formula) -> Formula:
    """Construct a flattened conjunction, returning the sole item when possible."""
    flattened: list[Formula] = []
    for part in parts:
        if isinstance(part, Top):
            continue
        if isinstance(part, And):
            flattened.extend(part.parts)
        else:
            flattened.append(part)
    if not flattened:
        return Top()
    if len(flattened) == 1:
        return flattened[0]
    return And(tuple(flattened))


def disjunction(*parts: Formula) -> Formula:
    """Construct a flattened disjunction, returning the sole item when possible."""
    flattened: list[Formula] = []
    for part in parts:
        if isinstance(part, Or):
            flattened.extend(part.parts)
        else:
            flattened.append(part)
    if len(flattened) == 1:
        return flattened[0]
    if not flattened:
        raise ValueError("A disjunction requires at least one operand.")
    return Or(tuple(flattened))


def parenthesize(formula: Formula) -> str:
    if isinstance(formula, (Atom, Top, Knows)):
        return str(formula)
    return f"({formula})"


def formula_from_dict(data: Mapping[str, Any]) -> Formula:
    kind = str(data.get("type", ""))
    if kind == "top":
        return Top()
    if kind == "atom":
        return Atom(str(data["name"]))
    if kind == "not":
        return Not(formula_from_dict(data["inner"]))
    if kind == "and":
        return And(tuple(formula_from_dict(item) for item in data["parts"]))
    if kind == "or":
        return Or(tuple(formula_from_dict(item) for item in data["parts"]))
    if kind == "knows":
        return Knows(str(data["agent"]), formula_from_dict(data["inner"]))
    raise ValueError(f"Unsupported formula type: {kind!r}")


def collect_atoms(formula: Formula) -> frozenset[str]:
    atoms: set[str] = set()

    def visit(node: Formula) -> None:
        if isinstance(node, Atom):
            atoms.add(node.name)
        elif isinstance(node, Not):
            visit(node.inner)
        elif isinstance(node, (And, Or)):
            for part in node.parts:
                visit(part)
        elif isinstance(node, Knows):
            visit(node.inner)

    visit(formula)
    return frozenset(atoms)


def collect_agents(formula: Formula) -> frozenset[str]:
    agents: set[str] = set()

    def visit(node: Formula) -> None:
        if isinstance(node, Not):
            visit(node.inner)
        elif isinstance(node, (And, Or)):
            for part in node.parts:
                visit(part)
        elif isinstance(node, Knows):
            agents.add(node.agent)
            visit(node.inner)

    visit(formula)
    return frozenset(agents)


def unique_subformulas(formulas: Iterable[Formula]) -> frozenset[Formula]:
    result: set[Formula] = set()

    def visit(node: Formula) -> None:
        if node in result:
            return
        result.add(node)
        if isinstance(node, Not):
            visit(node.inner)
        elif isinstance(node, (And, Or)):
            for part in node.parts:
                visit(part)
        elif isinstance(node, Knows):
            visit(node.inner)

    for formula in formulas:
        visit(formula)
    return frozenset(result)
