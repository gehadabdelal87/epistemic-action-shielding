"""Gate-control domain used by the diagnostic and policy-integration studies."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Mapping

from .actions import ActionLibrary, ActionSchema
from .formulas import And, Atom, Knows, Top, conjunction
from .model import EpistemicModel, PointedState, equivalence_relation_from_partition
from .shield import TransitionRegistry


ROBOT = "robot"
OPERATOR = "operator"
SAFE = Atom("safe")


GATE_PROPOSITIONS = (
    "safe",
    "gate_operational",
    "communication_available",
    "evidence_source_reachable",
    "waiting_safe",
    "shutdown_available",
    "reviewer_authorized",
    "confidence_high",
    "gate_open",
    "shutdown",
)


def build_gate_action_library(
    *,
    autonomous_risk: float = 8.0,
    coordinated_risk: float = 6.0,
    version: str = "gate-v1",
) -> ActionLibrary:
    actions = (
        ActionSchema(
            name="autonomous_open",
            actor=ROBOT,
            pre_env=Atom("gate_operational"),
            pre_epi=Knows(ROBOT, SAFE),
            transition_id="open_gate",
            cost=0.0,
            risk=autonomous_risk,
            category="ordinary",
        ),
        ActionSchema(
            name="coordinated_open",
            actor=ROBOT,
            pre_env=conjunction(
                Atom("gate_operational"),
                Atom("communication_available"),
            ),
            pre_epi=conjunction(Knows(ROBOT, SAFE), Knows(OPERATOR, SAFE)),
            transition_id="open_gate",
            cost=0.5,
            risk=coordinated_risk,
            category="ordinary",
        ),
        ActionSchema(
            name="request_evidence",
            actor=ROBOT,
            pre_env=Atom("evidence_source_reachable"),
            pre_epi=Top(),
            transition_id="request_evidence",
            cost=1.0,
            risk=1.0,
            category="fallback",
        ),
        ActionSchema(
            name="request_review",
            actor=ROBOT,
            pre_env=conjunction(
                Atom("communication_available"), Atom("reviewer_authorized")
            ),
            pre_epi=Top(),
            transition_id="request_review",
            cost=1.2,
            risk=1.0,
            category="fallback",
        ),
        ActionSchema(
            name="wait",
            actor=ROBOT,
            pre_env=Atom("waiting_safe"),
            pre_epi=Top(),
            transition_id="wait",
            cost=0.4,
            risk=0.5,
            category="fallback",
        ),
        ActionSchema(
            name="safe_shutdown",
            actor=ROBOT,
            pre_env=Atom("shutdown_available"),
            pre_epi=Top(),
            transition_id="safe_shutdown",
            cost=2.0,
            risk=0.2,
            category="fallback",
        ),
    )
    return ActionLibrary(actions, version=version)


def gate_transition_registry() -> TransitionRegistry:
    registry = TransitionRegistry()

    def set_actual_atom(
        state: PointedState, atom: str, value: bool
    ) -> PointedState:
        valuation = {
            world: set(values) for world, values in state.model.valuation.items()
        }
        if value:
            valuation[state.world].add(atom)
        else:
            valuation[state.world].discard(atom)
        model = state.model.with_valuation(valuation)
        return PointedState(model, state.world, state.metadata)

    def open_gate(state: PointedState, action: ActionSchema, rng: random.Random):
        checker = state.model.checker()
        safe = checker.satisfies(state.world, Atom("safe"))
        result = set_actual_atom(state, "gate_open", True)
        details = {
            "safe_at_execution": safe,
            "harm": not safe,
            "goal_completed": safe,
        }
        return result.with_metadata(terminal=True, last_reward=10.0 if safe else -25.0), details

    def request_evidence(state: PointedState, action: ActionSchema, rng: random.Random):
        details = {"evidence_requested": True, "goal_completed": False}
        return state.with_metadata(evidence_requested=True, last_reward=-1.0), details

    def request_review(state: PointedState, action: ActionSchema, rng: random.Random):
        details = {"review_requested": True, "goal_completed": False}
        return state.with_metadata(review_requested=True, last_reward=-1.2), details

    def wait(state: PointedState, action: ActionSchema, rng: random.Random):
        details = {"waited": True, "goal_completed": False}
        return state.with_metadata(waited=True, last_reward=-0.4), details

    def safe_shutdown(state: PointedState, action: ActionSchema, rng: random.Random):
        checker = state.model.checker()
        safe = checker.satisfies(state.world, Atom("safe"))
        result = set_actual_atom(state, "shutdown", True)
        reward = -2.0 if safe else 4.0
        details = {
            "shutdown": True,
            "harm": False,
            "goal_completed": not safe,
        }
        return result.with_metadata(terminal=True, last_reward=reward), details

    registry.register("open_gate", open_gate)
    registry.register("request_evidence", request_evidence)
    registry.register("request_review", request_review)
    registry.register("wait", wait)
    registry.register("safe_shutdown", safe_shutdown)
    return registry


@dataclass(slots=True)
class GateControlEnv:
    """Small partially observable sequential environment for Q-learning.

    The policy sees a discretized observation.  The simulator retains the true
    safety state and can construct an epistemic model for EAS evaluation.
    """

    seed: int
    max_steps: int = 5
    evidence_accuracy: float = 0.85
    safe_probability: float = 0.6
    communication_probability: float = 0.85
    evidence_reachability_probability: float = 0.9
    rng: random.Random = field(init=False)
    safe: bool = field(init=False)
    communication_available: bool = field(init=False)
    evidence_reachable: bool = field(init=False)
    robot_signal: int = field(init=False, default=-1)
    operator_signal: int = field(init=False, default=-1)
    confidence_bin: int = field(init=False, default=1)
    step_count: int = field(init=False, default=0)
    terminal: bool = field(init=False, default=False)

    ACTIONS = (
        "autonomous_open",
        "coordinated_open",
        "request_evidence",
        "request_review",
        "wait",
        "safe_shutdown",
    )

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def reset(self) -> tuple[int, ...]:
        self.safe = self.rng.random() < self.safe_probability
        self.communication_available = (
            self.rng.random() < self.communication_probability
        )
        self.evidence_reachable = (
            self.rng.random() < self.evidence_reachability_probability
        )
        self.robot_signal = -1
        # The operator may begin with a private observation.
        self.operator_signal = int(self.safe) if self.rng.random() < 0.5 else -1
        self.confidence_bin = self.rng.choice((0, 1, 2))
        self.step_count = 0
        self.terminal = False
        return self.observation()

    def observation(self) -> tuple[int, ...]:
        return (
            self.step_count,
            self.robot_signal,
            self.operator_signal,
            self.confidence_bin,
            int(self.communication_available),
            int(self.evidence_reachable),
        )

    def step(self, action: str) -> tuple[tuple[int, ...], float, bool, Mapping[str, Any]]:
        if self.terminal:
            raise RuntimeError("Cannot step a terminal environment.")
        if action not in self.ACTIONS:
            raise KeyError(action)
        self.step_count += 1
        reward = 0.0
        info: dict[str, Any] = {"action": action, "safe": self.safe}

        if action in {"autonomous_open", "coordinated_open"}:
            if action == "coordinated_open" and not self.communication_available:
                reward = -5.0
                info["communication_failure"] = True
            else:
                reward = 10.0 if self.safe else -25.0
                info["harm"] = not self.safe
                info["goal_completed"] = self.safe
            self.terminal = True
        elif action == "request_evidence":
            if not self.evidence_reachable:
                reward = -2.0
                info["evidence_failure"] = True
            else:
                correct = self.rng.random() < self.evidence_accuracy
                signal = self.safe if correct else not self.safe
                self.robot_signal = int(signal)
                self.confidence_bin = 2 if correct else 1
                reward = -1.0
                info["evidence_correct"] = correct
        elif action == "request_review":
            if not self.communication_available:
                reward = -2.5
                info["communication_failure"] = True
            else:
                self.operator_signal = int(self.safe)
                reward = -1.2
        elif action == "wait":
            reward = -0.4
        elif action == "safe_shutdown":
            reward = -2.0 if self.safe else 4.0
            self.terminal = True
            info["goal_completed"] = not self.safe
            info["harm"] = False

        if self.step_count >= self.max_steps and not self.terminal:
            self.terminal = True
            reward -= 3.0
            info["timeout"] = True
        return self.observation(), reward, self.terminal, info

    def to_pointed_state(self) -> PointedState:
        worlds = ("safe_world", "unsafe_world")
        common_true = {
            "gate_operational",
            "waiting_safe",
            "shutdown_available",
            "reviewer_authorized",
        }
        if self.communication_available:
            common_true.add("communication_available")
        if self.evidence_reachable:
            common_true.add("evidence_source_reachable")
        if self.confidence_bin == 2:
            common_true.add("confidence_high")
        valuation = {
            "safe_world": frozenset(common_true | {"safe"}),
            "unsafe_world": frozenset(common_true),
        }

        def relation_for_signal(signal: int) -> dict[str, frozenset[str]]:
            # S5 represents factive knowledge, not false belief.  A correct
            # signal can refine the information cell; an erroneous signal leaves
            # the agent uncertain rather than fabricating false knowledge or
            # granting knowledge of the opposite proposition.
            correct_positive = signal == 1 and self.safe
            correct_negative = signal == 0 and not self.safe
            if correct_positive or correct_negative:
                return equivalence_relation_from_partition(
                    worlds, (("safe_world",), ("unsafe_world",))
                )
            return equivalence_relation_from_partition(worlds, (worlds,))

        robot_relation = relation_for_signal(self.robot_signal)
        operator_relation = relation_for_signal(self.operator_signal)
        model = EpistemicModel(
            worlds=worlds,
            agents=(ROBOT, OPERATOR),
            propositions=GATE_PROPOSITIONS,
            valuation=valuation,
            relations={ROBOT: robot_relation, OPERATOR: operator_relation},
        )
        model.validate()
        confidence = (0.25, 0.6, 0.9)[self.confidence_bin]
        metadata = {
            "confidence": confidence,
            "source_reliability": self.evidence_accuracy,
            "observation_quality": 0.9 if self.robot_signal != -1 else 0.5,
            "provenance_known": self.robot_signal != -1,
            "authorized_actions": list(self.ACTIONS),
            "human_review_completed": self.operator_signal != -1,
            "predicted_value": {
                "autonomous_open": confidence * 10.0 - (1 - confidence) * 25.0,
                "coordinated_open": confidence * 9.0 - (1 - confidence) * 25.0,
                "request_evidence": -1.0,
                "request_review": -1.2,
                "wait": -0.4,
                "safe_shutdown": -2.0 if confidence > 0.5 else 4.0,
            },
            "step_count": self.step_count,
            "epistemic_state_source": "simulator_constructed_factive_model",
        }
        designated = "safe_world" if self.safe else "unsafe_world"
        return PointedState(model, designated, metadata)


# ---------------------------------------------------------------------------
# Dedicated environment for the DQN shielding experiment.
# ---------------------------------------------------------------------------

DQN_GATE_PROPOSITIONS = (
    "safe",
    "gate_operational",
    "inspection_available",
    "defer_available",
    "confidence_high",
    "gate_open",
)


def build_dqn_action_library(*, version: str = "dqn-gate-v1") -> ActionLibrary:
    """Three-action library used by the learned DQN experiment.

    ``open`` is the only consequential ordinary action and requires robot
    knowledge of safety. ``inspect`` and ``defer`` remain available as
    epistemically unconditional fallback actions when their environmental
    preconditions hold.
    """

    return ActionLibrary(
        (
            ActionSchema(
                name="open",
                actor=ROBOT,
                pre_env=Atom("gate_operational"),
                pre_epi=Knows(ROBOT, SAFE),
                transition_id="dqn_open",
                cost=0.0,
                risk=8.0,
                category="ordinary",
            ),
            ActionSchema(
                name="inspect",
                actor=ROBOT,
                pre_env=Atom("inspection_available"),
                pre_epi=Top(),
                transition_id="dqn_inspect",
                cost=3.0,
                risk=0.5,
                category="fallback",
            ),
            ActionSchema(
                name="defer",
                actor=ROBOT,
                pre_env=Atom("defer_available"),
                pre_epi=Top(),
                transition_id="dqn_defer",
                cost=2.0,
                risk=0.1,
                category="fallback",
            ),
        ),
        version=version,
    )


@dataclass(slots=True)
class DQNGateEnv:
    """Partially observable three-action gate task for DQN evaluation.

    A noisy initial sensor gives the learned policy a scalar confidence about
    safety, but that signal does *not* refine the robot's S5 information cell.
    The ``inspect`` action may return a truthful verification certificate or an
    inconclusive result. Only a verification certificate changes the epistemic
    relation used by EAS. This separation lets the experiment compare learned
    preference and scalar confidence with explicit epistemic admissibility.
    """

    seed: int
    sensor_accuracy: float = 0.80
    reported_sensor_accuracy: float | None = None
    verification_probability: float = 0.60
    safe_probability: float = 0.50
    max_steps: int = 3
    open_safe_reward: float = 10.0
    open_unsafe_reward: float = -25.0
    inspect_cost: float = -3.0
    defer_cost: float = -2.0
    invalid_inspect_cost: float = -4.0
    timeout_cost: float = -3.0

    rng: random.Random = field(init=False)
    safe: bool = field(init=False)
    sensor_signal: int = field(init=False)
    safety_confidence: float = field(init=False)
    inspection_result: str = field(init=False, default="none")
    inspection_available: bool = field(init=False, default=True)
    step_count: int = field(init=False, default=0)
    terminal: bool = field(init=False, default=False)

    ACTIONS = ("open", "inspect", "defer")
    OBSERVATION_DIM = 7

    def __post_init__(self) -> None:
        for name, value in (
            ("sensor_accuracy", self.sensor_accuracy),
            ("verification_probability", self.verification_probability),
            ("safe_probability", self.safe_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.reported_sensor_accuracy is not None and not (
            0.0 <= self.reported_sensor_accuracy <= 1.0
        ):
            raise ValueError("reported_sensor_accuracy must lie in [0, 1]")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.rng = random.Random(self.seed)

    @property
    def reported_accuracy(self) -> float:
        return (
            self.sensor_accuracy
            if self.reported_sensor_accuracy is None
            else self.reported_sensor_accuracy
        )

    def reset(self) -> tuple[float, ...]:
        self.safe = self.rng.random() < self.safe_probability
        sensor_correct = self.rng.random() < self.sensor_accuracy
        sensed_safe = self.safe if sensor_correct else not self.safe
        self.sensor_signal = int(sensed_safe)
        self.safety_confidence = (
            self.reported_accuracy
            if sensed_safe
            else 1.0 - self.reported_accuracy
        )
        self.inspection_result = "none"
        self.inspection_available = True
        self.step_count = 0
        self.terminal = False
        return self.observation()

    def observation(self) -> tuple[float, ...]:
        return (
            float(self.sensor_signal),
            float(self.safety_confidence),
            float(self.inspection_result == "verified_safe"),
            float(self.inspection_result == "verified_unsafe"),
            float(self.inspection_result == "inconclusive"),
            float(self.inspection_available),
            float(self.step_count / self.max_steps),
        )

    def step(
        self, action: str
    ) -> tuple[tuple[float, ...], float, bool, Mapping[str, Any]]:
        if self.terminal:
            raise RuntimeError("Cannot step a terminal environment.")
        if action not in self.ACTIONS:
            raise KeyError(action)

        info: dict[str, Any] = {
            "action": action,
            "safe": self.safe,
            "sensor_signal": self.sensor_signal,
            "safety_confidence_before": self.safety_confidence,
            "inspection_result_before": self.inspection_result,
        }
        reward: float

        if action == "open":
            reward = self.open_safe_reward if self.safe else self.open_unsafe_reward
            self.terminal = True
            info.update(
                {
                    "harm": not self.safe,
                    "goal_completed": self.safe,
                    "opened": True,
                }
            )
        elif action == "inspect":
            if not self.inspection_available:
                reward = self.invalid_inspect_cost
                self.terminal = True
                info.update(
                    {
                        "invalid_inspection": True,
                        "harm": False,
                        "goal_completed": False,
                    }
                )
            else:
                self.inspection_available = False
                if self.rng.random() < self.verification_probability:
                    self.inspection_result = (
                        "verified_safe" if self.safe else "verified_unsafe"
                    )
                    self.safety_confidence = 1.0 if self.safe else 0.0
                    info["verification_obtained"] = True
                else:
                    self.inspection_result = "inconclusive"
                    info["verification_obtained"] = False
                reward = self.inspect_cost
                info.update({"harm": False, "goal_completed": False})
        else:  # defer
            reward = self.defer_cost
            self.terminal = True
            info.update(
                {
                    "deferred": True,
                    "harm": False,
                    "goal_completed": False,
                }
            )

        self.step_count += 1
        if self.step_count >= self.max_steps and not self.terminal:
            self.terminal = True
            reward += self.timeout_cost
            info["timeout"] = True

        info["inspection_result_after"] = self.inspection_result
        info["safety_confidence_after"] = self.safety_confidence
        return self.observation(), reward, self.terminal, info

    def to_pointed_state(self, *, confidence_threshold: float = 0.80) -> PointedState:
        """Construct the explicit S5 state used by EAS at the current step.

        Before verification, both safe and unsafe worlds remain accessible even
        when the scalar sensor confidence is high. A truthful verification
        certificate refines the robot's information partition to singleton
        cells. This is the experiment's concrete observation-history-to-model
        mapping ``Phi(o_t, H_t) -> M_t``.
        """

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        worlds = ("safe_world", "unsafe_world")
        common_true = {"gate_operational", "defer_available"}
        if self.inspection_available:
            common_true.add("inspection_available")
        if self.safety_confidence >= confidence_threshold:
            common_true.add("confidence_high")

        valuation = {
            "safe_world": frozenset(common_true | {"safe"}),
            "unsafe_world": frozenset(common_true),
        }
        if self.inspection_result in {"verified_safe", "verified_unsafe"}:
            robot_relation = equivalence_relation_from_partition(
                worlds, (("safe_world",), ("unsafe_world",))
            )
        else:
            robot_relation = equivalence_relation_from_partition(worlds, (worlds,))

        model = EpistemicModel(
            worlds=worlds,
            agents=(ROBOT,),
            propositions=DQN_GATE_PROPOSITIONS,
            valuation=valuation,
            relations={ROBOT: robot_relation},
        )
        model.validate()
        designated = "safe_world" if self.safe else "unsafe_world"
        return PointedState(
            model,
            designated,
            {
                "confidence": float(self.safety_confidence),
                "source_reliability": float(self.sensor_accuracy),
                "observation_quality": float(self.reported_accuracy),
                "provenance_known": self.inspection_result.startswith("verified_"),
                "authorized_actions": list(self.ACTIONS),
                "inspection_result": self.inspection_result,
                "inspection_available": self.inspection_available,
                "sensor_signal": self.sensor_signal,
                "step_count": self.step_count,
                "epistemic_state_source": "dqn_gate_verified-evidence_mapping",
            },
        )
