"""Controlled diagnostic variants used in the experiments."""

from __future__ import annotations

from dataclasses import replace

from .actions import ActionLibrary
from .formulas import Atom, Knows, Top
from .governance import GovernancePolicy


CONDITIONS = (
    "eas",
    "state_truth",
    "confidence",
    "operator_knowledge",
    "no_gate",
    "no_governance",
)


def action_library_for_condition(
    canonical: ActionLibrary,
    condition: str,
) -> ActionLibrary:
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition: {condition}")
    if condition in {"eas", "no_governance"}:
        return canonical

    replacements = {}
    for name in ("autonomous_open", "coordinated_open"):
        action = canonical.get(name)
        if condition == "state_truth":
            pre_epi = Atom("safe")
        elif condition == "confidence":
            pre_epi = Atom("confidence_high")
        elif condition == "operator_knowledge":
            pre_epi = Knows("operator", Atom("safe"))
        elif condition == "no_gate":
            pre_epi = Top()
        else:
            raise AssertionError(condition)
        replacements[name] = replace(
            action,
            pre_epi=pre_epi,
            version=f"{action.version}+{condition}",
        )
    return canonical.replace(replacements)


def governance_policy_for_condition(
    canonical: GovernancePolicy,
    condition: str,
) -> GovernancePolicy:
    if condition == "no_governance":
        return GovernancePolicy((), version=f"{canonical.version}+disabled")
    return canonical
