"""Generation of logically valid finite S5 scenarios."""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import random
from typing import Any, Iterable, Mapping, Sequence

from .actions import ActionLibrary, ActionSchema
from .environment import (
    GATE_PROPOSITIONS,
    OPERATOR,
    ROBOT,
    build_gate_action_library,
)
from .formulas import Atom, Knows, Top
from .governance import (
    AuthorizationConstraint,
    GovernancePolicy,
    MaxRiskConstraint,
    ObservationQualityConstraint,
    ProvenanceConstraint,
    SourceReliabilityConstraint,
)
from .model import EpistemicModel, PointedState, equivalence_relation_from_partition


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    seed: int
    state: PointedState
    action_library: ActionLibrary
    governance_policy: GovernancePolicy
    utility_by_action: Mapping[str, float]
    fallback_priority: tuple[str, ...]
    stratum: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GateScenarioParameters:
    safe: bool
    robot_knows_safe: bool | None
    operator_knows_safe: bool | None
    confidence: float
    source_reliability: float
    observation_quality: float
    provenance_known: bool
    autonomous_risk: float
    coordinated_risk: float
    communication_available: bool
    evidence_source_reachable: bool
    waiting_safe: bool
    shutdown_available: bool
    reviewer_authorized: bool


def _gate_worlds(extra_worlds: int, rng: random.Random) -> tuple[str, ...]:
    if extra_worlds < 0:
        raise ValueError("extra_worlds may not be negative")
    return tuple(["w_safe_0", "w_unsafe_0"] + [f"w_extra_{i}" for i in range(extra_worlds)])


def _partition_with_designated_cell(
    worlds: Sequence[str],
    designated: str,
    designated_cell: set[str],
    rng: random.Random,
    max_block_size: int = 4,
) -> dict[str, frozenset[str]]:
    if designated not in designated_cell:
        raise ValueError("Designated cell must contain the designated world")
    remaining = [world for world in worlds if world not in designated_cell]
    rng.shuffle(remaining)
    blocks: list[tuple[str, ...]] = [tuple(sorted(designated_cell))]
    while remaining:
        size = rng.randint(1, min(max_block_size, len(remaining)))
        block = tuple(sorted(remaining[:size]))
        remaining = remaining[size:]
        blocks.append(block)
    return equivalence_relation_from_partition(worlds, blocks)


def _knowledge_cell(
    worlds: Sequence[str],
    valuation: Mapping[str, frozenset[str]],
    designated: str,
    proposition: str,
    knows_positive: bool | None,
    rng: random.Random,
) -> set[str]:
    designated_truth = proposition in valuation[designated]
    positive = [world for world in worlds if proposition in valuation[world]]
    negative = [world for world in worlds if proposition not in valuation[world]]

    if knows_positive is True:
        if not designated_truth:
            raise ValueError("Positive knowledge is impossible at a negative designated world in S5")
        candidates = positive
        count = rng.randint(1, max(1, min(4, len(candidates))))
        cell = set(rng.sample(candidates, count))
        cell.add(designated)
        return cell
    if knows_positive is False:
        # Explicitly require failure of K_i p by including an opposite-valued world.
        opposite = negative if designated_truth else positive
        if not opposite:
            raise ValueError("Cannot generate uncertainty without an opposite-valued world")
        same = positive if designated_truth else negative
        cell = {designated, rng.choice(opposite)}
        if len(same) > 1 and rng.random() < 0.5:
            cell.add(rng.choice([world for world in same if world != designated]))
        return cell
    # None means knowledge status not stratified; generate either precise or uncertain.
    if rng.random() < 0.5:
        same = positive if designated_truth else negative
        return {designated} if len(same) == 1 else {designated, rng.choice(same)}
    opposite = negative if designated_truth else positive
    return {designated, rng.choice(opposite)}


def generate_gate_scenario(
    *,
    seed: int,
    index: int,
    parameters: GateScenarioParameters | None = None,
    extra_worlds: int = 6,
    extra_propositions: int = 2,
    extra_agents: int = 0,
    source_threshold: float = 0.7,
    quality_threshold: float = 0.7,
    risk_threshold: float = 10.0,
) -> Scenario:
    rng = random.Random(seed * 1_000_003 + index)
    if parameters is None:
        safe = rng.random() < 0.5
        parameters = GateScenarioParameters(
            safe=safe,
            robot_knows_safe=(rng.random() < 0.5) if safe else False,
            operator_knows_safe=(rng.random() < 0.5) if safe else False,
            confidence=rng.random(),
            source_reliability=rng.random(),
            observation_quality=rng.random(),
            provenance_known=rng.random() < 0.7,
            autonomous_risk=rng.uniform(1.0, 14.0),
            coordinated_risk=rng.uniform(1.0, 12.0),
            communication_available=rng.random() < 0.85,
            evidence_source_reachable=rng.random() < 0.9,
            waiting_safe=rng.random() < 0.9,
            shutdown_available=rng.random() < 0.95,
            reviewer_authorized=rng.random() < 0.85,
        )

    worlds = _gate_worlds(extra_worlds, rng)
    designated = "w_safe_0" if parameters.safe else "w_unsafe_0"
    extra_props = tuple(f"context_{i}" for i in range(extra_propositions))
    propositions = tuple(dict.fromkeys(GATE_PROPOSITIONS + extra_props))

    valuation: dict[str, frozenset[str]] = {}
    for world in worlds:
        truths: set[str] = {
            "gate_operational",
            "waiting_safe" if parameters.waiting_safe else "",
            "shutdown_available" if parameters.shutdown_available else "",
            "reviewer_authorized" if parameters.reviewer_authorized else "",
            "communication_available" if parameters.communication_available else "",
            "evidence_source_reachable" if parameters.evidence_source_reachable else "",
            "confidence_high" if parameters.confidence >= 0.7 else "",
        }
        truths.discard("")
        if world.startswith("w_safe") or (
            world.startswith("w_extra") and rng.random() < 0.5
        ):
            truths.add("safe")
        for proposition in extra_props:
            if rng.random() < 0.5:
                truths.add(proposition)
        valuation[world] = frozenset(truths)

    # Ensure the two anchor worlds differ on safety.
    valuation["w_safe_0"] = frozenset(set(valuation["w_safe_0"]) | {"safe"})
    valuation["w_unsafe_0"] = frozenset(set(valuation["w_unsafe_0"]) - {"safe"})

    robot_cell = _knowledge_cell(
        worlds,
        valuation,
        designated,
        "safe",
        parameters.robot_knows_safe,
        rng,
    )
    operator_cell = _knowledge_cell(
        worlds,
        valuation,
        designated,
        "safe",
        parameters.operator_knows_safe,
        rng,
    )
    relations: dict[str, dict[str, frozenset[str]]] = {
        ROBOT: _partition_with_designated_cell(worlds, designated, robot_cell, rng),
        OPERATOR: _partition_with_designated_cell(worlds, designated, operator_cell, rng),
    }
    agents = [ROBOT, OPERATOR]
    for agent_index in range(extra_agents):
        agent = f"agent_{agent_index}"
        agents.append(agent)
        # Random S5 partition.
        shuffled = list(worlds)
        rng.shuffle(shuffled)
        blocks: list[tuple[str, ...]] = []
        while shuffled:
            size = rng.randint(1, min(4, len(shuffled)))
            blocks.append(tuple(shuffled[:size]))
            shuffled = shuffled[size:]
        relations[agent] = equivalence_relation_from_partition(worlds, blocks)

    model = EpistemicModel(
        worlds=tuple(worlds),
        agents=tuple(agents),
        propositions=propositions,
        valuation=valuation,
        relations=relations,
    )
    model.validate()

    actions = build_gate_action_library(
        autonomous_risk=parameters.autonomous_risk,
        coordinated_risk=parameters.coordinated_risk,
        version=f"gate-{seed}-{index}",
    )
    open_actions = frozenset({"autonomous_open", "coordinated_open"})
    governance = GovernancePolicy(
        constraints=(
            SourceReliabilityConstraint(source_threshold, open_actions),
            ObservationQualityConstraint(quality_threshold, open_actions),
            ProvenanceConstraint(open_actions),
            MaxRiskConstraint(risk_threshold, open_actions),
            AuthorizationConstraint(applies_to=frozenset(action.name for action in actions.actions)),
        ),
        version=f"gov-s{source_threshold}-q{quality_threshold}-r{risk_threshold}",
    )
    authorized_actions = [action.name for action in actions.actions]
    metadata = {
        "confidence": parameters.confidence,
        "source_reliability": parameters.source_reliability,
        "observation_quality": parameters.observation_quality,
        "provenance_known": parameters.provenance_known,
        "authorized_actions": authorized_actions,
        "human_review_completed": parameters.operator_knows_safe is True,
        "predicted_value": {
            "autonomous_open": parameters.confidence * 10.0 - (1 - parameters.confidence) * 25.0,
            "coordinated_open": parameters.confidence * 9.0 - (1 - parameters.confidence) * 25.0,
            "request_evidence": 3.0 if parameters.evidence_source_reachable else -5.0,
            "request_review": 2.5 if parameters.communication_available else -5.0,
            "wait": 1.0 if parameters.waiting_safe else -5.0,
            "safe_shutdown": 4.0 if not parameters.safe else -2.0,
        },
        "scenario_oracle": {
            "safe": parameters.safe,
            "robot_knows_safe_requested": parameters.robot_knows_safe,
            "operator_knows_safe_requested": parameters.operator_knows_safe,
        },
    }
    state = PointedState(model, designated, metadata)
    checker = model.checker()
    robot_knows = checker.satisfies(designated, Knows(ROBOT, Atom("safe")))
    operator_knows = checker.satisfies(designated, Knows(OPERATOR, Atom("safe")))
    stratum = {
        "safe": parameters.safe,
        "robot_knows_safe": robot_knows,
        "operator_knows_safe": operator_knows,
        "confidence_high": parameters.confidence >= 0.7,
        "source_reliable": parameters.source_reliability >= source_threshold,
        "observation_quality_sufficient": parameters.observation_quality >= quality_threshold,
        "provenance_known": parameters.provenance_known,
        "autonomous_risk_admissible": parameters.autonomous_risk <= risk_threshold,
        "coordinated_risk_admissible": parameters.coordinated_risk <= risk_threshold,
        "communication_available": parameters.communication_available,
        "fallback_available": any(
            [
                parameters.evidence_source_reachable,
                parameters.communication_available and parameters.reviewer_authorized,
                parameters.waiting_safe,
                parameters.shutdown_available,
            ]
        ),
    }
    utility = {
        "autonomous_open": 10.0,
        "coordinated_open": 9.0,
        "request_evidence": 4.0,
        "request_review": 3.0,
        "wait": 1.0,
        "safe_shutdown": 0.5,
    }
    return Scenario(
        scenario_id=f"seed{seed}-scenario{index}",
        seed=seed,
        state=state,
        action_library=actions,
        governance_policy=governance,
        utility_by_action=utility,
        fallback_priority=(
            "request_evidence",
            "request_review",
            "wait",
            "safe_shutdown",
        ),
        stratum=stratum,
    )


def stratified_gate_parameters(seed: int, count: int) -> list[GateScenarioParameters]:
    """Generate a balanced, logically valid factorial scenario schedule.

    Epistemic strata are crossed with high/low confidence, source reliability,
    observation quality, provenance, risk, communication, and evidence access.
    The factorial schedule is shuffled once per seed and cycled when ``count``
    exceeds the number of unique combinations.
    """
    rng = random.Random(seed)
    epistemic_strata = [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, False, False),
    ]
    combinations = list(
        itertools.product(
            epistemic_strata,
            (False, True),  # confidence high
            (False, True),  # source reliable
            (False, True),  # observation quality sufficient
            (False, True),  # provenance known
            (False, True),  # action risk admissible
            (False, True),  # communication available
            (False, True),  # evidence source reachable
        )
    )
    rng.shuffle(combinations)
    result: list[GateScenarioParameters] = []
    for index in range(count):
        (
            (safe, robot_knows, operator_knows),
            confidence_high,
            source_high,
            quality_high,
            provenance_known,
            risk_admissible,
            communication_available,
            evidence_reachable,
        ) = combinations[index % len(combinations)]
        # Jitter values within non-overlapping bands to avoid identical numeric
        # records while preserving the intended threshold stratum.
        confidence = rng.uniform(0.78, 0.95) if confidence_high else rng.uniform(0.25, 0.62)
        source = rng.uniform(0.78, 0.95) if source_high else rng.uniform(0.25, 0.62)
        quality = rng.uniform(0.78, 0.95) if quality_high else rng.uniform(0.25, 0.62)
        autonomous_risk = rng.uniform(3.0, 8.5) if risk_admissible else rng.uniform(10.5, 14.0)
        coordinated_risk = rng.uniform(2.5, 8.0) if risk_admissible else rng.uniform(10.5, 13.0)
        # Include a small but explicit set of fail-closed cases where every
        # ordinary fallback is unavailable.
        no_fallback = (
            not communication_available
            and not evidence_reachable
            and index % 11 == 0
        )
        result.append(
            GateScenarioParameters(
                safe=safe,
                robot_knows_safe=robot_knows,
                operator_knows_safe=operator_knows,
                confidence=confidence,
                source_reliability=source,
                observation_quality=quality,
                provenance_known=provenance_known,
                autonomous_risk=autonomous_risk,
                coordinated_risk=coordinated_risk,
                communication_available=communication_available,
                evidence_source_reachable=evidence_reachable,
                waiting_safe=not no_fallback,
                shutdown_available=not no_fallback,
                reviewer_authorized=communication_available,
            )
        )
    return result
