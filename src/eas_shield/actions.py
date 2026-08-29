"""Action schemas and versioned action libraries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from .formulas import Formula, formula_from_dict


class ActionValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ActionSchema:
    name: str
    actor: str
    pre_env: Formula
    pre_epi: Formula
    transition_id: str
    cost: float = 0.0
    risk: float = 0.0
    category: str = "ordinary"
    version: str = "1"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ActionValidationError("Action name must be non-empty.")
        if not self.actor.strip():
            raise ActionValidationError("Action actor must be non-empty.")
        if not self.transition_id.strip():
            raise ActionValidationError("Action transition_id must be non-empty.")
        if self.risk < 0:
            raise ActionValidationError("Action risk may not be negative.")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def is_fallback(self) -> bool:
        return self.category == "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "actor": self.actor,
            "pre_env": self.pre_env.to_dict(),
            "pre_epi": self.pre_epi.to_dict(),
            "transition_id": self.transition_id,
            "cost": self.cost,
            "risk": self.risk,
            "category": self.category,
            "version": self.version,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionSchema":
        return cls(
            name=str(data["name"]),
            actor=str(data["actor"]),
            pre_env=formula_from_dict(data["pre_env"]),
            pre_epi=formula_from_dict(data["pre_epi"]),
            transition_id=str(data["transition_id"]),
            cost=float(data.get("cost", 0.0)),
            risk=float(data.get("risk", 0.0)),
            category=str(data.get("category", "ordinary")),
            version=str(data.get("version", "1")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ActionLibrary:
    actions: tuple[ActionSchema, ...]
    version: str = "1"

    def __post_init__(self) -> None:
        names = [action.name for action in self.actions]
        if len(names) != len(set(names)):
            raise ActionValidationError("Action names must be unique.")
        if not self.actions:
            raise ActionValidationError("An action library may not be empty.")

    @property
    def by_name(self) -> dict[str, ActionSchema]:
        return {action.name: action for action in self.actions}

    @property
    def ordinary(self) -> tuple[ActionSchema, ...]:
        return tuple(action for action in self.actions if not action.is_fallback)

    @property
    def fallbacks(self) -> tuple[ActionSchema, ...]:
        return tuple(action for action in self.actions if action.is_fallback)

    @property
    def library_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def get(self, name: str) -> ActionSchema:
        try:
            return self.by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown action: {name}") from exc

    def replace(self, replacements: Mapping[str, ActionSchema]) -> "ActionLibrary":
        unknown = set(replacements) - set(self.by_name)
        if unknown:
            raise KeyError(f"Cannot replace unknown actions: {sorted(unknown)}")
        return ActionLibrary(
            tuple(replacements.get(action.name, action) for action in self.actions),
            version=f"{self.version}+variant",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionLibrary":
        return cls(
            actions=tuple(ActionSchema.from_dict(item) for item in data["actions"]),
            version=str(data.get("version", "1")),
        )
