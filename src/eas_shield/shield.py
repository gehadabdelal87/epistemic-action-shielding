"""Executable Epistemic Action Shielding decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
import uuid
from typing import Any, Callable, Mapping, Sequence

from .actions import ActionLibrary, ActionSchema
from .events import EventModel, EventValidationError, product_update
from .formulas import collect_agents, collect_atoms
from .governance import GovernanceEvaluation, GovernancePolicy
from .model import ModelValidationError, PointedState
from .revision import EvidencePolicy, Observation, RevisionRecord, refine_information_partition
from .trace import AuthorizationTraceEntry, DecisionTrace, ExecutionTraceEntry


class DecisionMode(str, Enum):
    OPTIMIZE = "optimize"
    SHIELD = "shield"


class DecisionStatus(str, Enum):
    CONFIGURATION_ERROR = "configuration_error"
    UPDATE_ERROR = "update_error"
    PROPOSAL_AUTHORIZED = "proposal_authorized"
    ORDINARY_ACTION_SELECTED = "ordinary_action_selected"
    REVIEW_SELECTED = "review_selected"
    EVIDENCE_REQUEST_SELECTED = "evidence_request_selected"
    WAIT_SELECTED = "wait_selected"
    SAFE_SHUTDOWN_SELECTED = "safe_shutdown_selected"
    FALLBACK_SELECTED = "fallback_selected"
    NO_ADMISSIBLE_ACTION = "no_admissible_action"
    EXECUTION_SUCCEEDED = "execution_succeeded"
    EXECUTION_FAILED = "execution_failed"
    HALTED = "halted"


@dataclass(frozen=True, slots=True)
class AuthorizationOutcome:
    decision_id: str
    prior_state: PointedState
    authorization_state: PointedState
    selected_action: str | None
    status: DecisionStatus
    epistemically_permitted: tuple[str, ...]
    epistemically_blocked: tuple[str, ...]
    environmentally_permitted: tuple[str, ...]
    environmentally_blocked: tuple[str, ...]
    admissible: tuple[str, ...]
    governance_rejected: tuple[str, ...]
    proposal: str | None
    trace_entry: AuthorizationTraceEntry

    @property
    def halted(self) -> bool:
        return self.selected_action is None


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    authorization: AuthorizationOutcome
    resulting_state: PointedState
    status: DecisionStatus
    details: Mapping[str, Any]


TransitionHandler = Callable[
    [PointedState, ActionSchema, random.Random],
    tuple[PointedState, Mapping[str, Any]],
]


@dataclass(slots=True)
class TransitionRegistry:
    handlers: dict[str, TransitionHandler] = field(default_factory=dict)

    def register(self, transition_id: str, handler: TransitionHandler) -> None:
        if transition_id in self.handlers:
            raise ValueError(f"Transition already registered: {transition_id}")
        self.handlers[transition_id] = handler

    def execute(
        self,
        state: PointedState,
        action: ActionSchema,
        rng: random.Random,
    ) -> tuple[PointedState, Mapping[str, Any]]:
        try:
            handler = self.handlers[action.transition_id]
        except KeyError as exc:
            raise KeyError(
                f"No transition handler registered for {action.transition_id!r}."
            ) from exc
        return handler(state, action, rng)


@dataclass(slots=True)
class EASDecisionEngine:
    """Fail-closed implementation of the EAS authorization pipeline."""

    def validate_configuration(
        self,
        state: PointedState,
        action_library: ActionLibrary,
        governance_policy: GovernancePolicy,
        fallback_priority: Sequence[str],
    ) -> tuple[str, ...]:
        failures: list[str] = []
        try:
            state.model.validate()
        except ModelValidationError as exc:
            failures.append(f"model:{exc}")
        action_names = set(action_library.by_name)
        if not action_names:
            failures.append("empty_action_library")
        for action in action_library.actions:
            if action.actor not in state.model.agents:
                failures.append(f"unknown_actor:{action.name}:{action.actor}")
            unknown_atoms = (
                collect_atoms(action.pre_epi) | collect_atoms(action.pre_env)
            ) - set(state.model.propositions)
            if unknown_atoms:
                failures.append(
                    f"unknown_atoms:{action.name}:{','.join(sorted(unknown_atoms))}"
                )
            unknown_agents = (
                collect_agents(action.pre_epi) | collect_agents(action.pre_env)
            ) - set(state.model.agents)
            if unknown_agents:
                failures.append(
                    f"unknown_agents:{action.name}:{','.join(sorted(unknown_agents))}"
                )
        unknown_fallbacks = set(fallback_priority) - action_names
        if unknown_fallbacks:
            failures.append(
                f"unknown_fallbacks:{','.join(sorted(unknown_fallbacks))}"
            )
        fallback_set = {action.name for action in action_library.fallbacks}
        wrongly_categorized = set(fallback_priority) - fallback_set
        if wrongly_categorized:
            failures.append(
                f"fallback_priority_contains_ordinary:{','.join(sorted(wrongly_categorized))}"
            )
        return tuple(failures)

    def authorize(
        self,
        *,
        state: PointedState,
        action_library: ActionLibrary,
        governance_policy: GovernancePolicy,
        trace: DecisionTrace,
        mode: DecisionMode = DecisionMode.OPTIMIZE,
        proposal: str | None = None,
        utility_by_action: Mapping[str, float] | None = None,
        fallback_priority: Sequence[str] = (),
        event_model: EventModel | None = None,
        designated_event: str | None = None,
        observations: Sequence[Observation] = (),
        evidence_policy: EvidencePolicy | None = None,
        random_seed: int | None = None,
        decision_id: str | None = None,
    ) -> AuthorizationOutcome:
        decision_id = decision_id or uuid.uuid4().hex
        utility = {str(k): float(v) for k, v in (utility_by_action or {}).items()}
        fallback_priority = tuple(fallback_priority)
        prior_state = state
        failures = self.validate_configuration(
            state, action_library, governance_policy, fallback_priority
        )
        if mode is DecisionMode.SHIELD:
            if proposal is None:
                failures += ("missing_external_proposal",)
            elif proposal not in action_library.by_name:
                failures += (f"invalid_external_proposal:{proposal}",)

        if failures:
            entry = self._trace_entry(
                decision_id=decision_id,
                prior_state=prior_state,
                authorization_state=prior_state,
                action_library=action_library,
                governance_policy=governance_policy,
                mode=mode,
                proposal=proposal,
                epi_permitted=(),
                epi_blocked=tuple(action_library.by_name),
                witnesses={"configuration": list(failures)},
                env_permitted=(),
                env_blocked=(),
                admissible=(),
                gov_rejected={},
                selected=None,
                status=DecisionStatus.CONFIGURATION_ERROR,
                fallback_priority=fallback_priority,
                utility=utility,
                random_seed=random_seed,
                event_record={},
                revision_records=(),
            )
            trace.append_authorization(entry)
            return AuthorizationOutcome(
                decision_id,
                prior_state,
                prior_state,
                None,
                DecisionStatus.CONFIGURATION_ERROR,
                (),
                tuple(action_library.by_name),
                (),
                (),
                (),
                (),
                proposal,
                entry,
            )

        authorization_state = state
        event_record: dict[str, Any] = {}
        revision_records: list[RevisionRecord] = []
        force_fallback_only = False
        update_error: str | None = None

        if event_model is not None or designated_event is not None:
            if event_model is None or designated_event is None:
                update_error = "event_model_and_designated_event_must_be_supplied_together"
            else:
                try:
                    authorization_state = product_update(
                        authorization_state, event_model, designated_event
                    )
                    event_record = {
                        "event_model_id": event_model.event_model_id,
                        "designated_event": designated_event,
                        "status": "applied",
                    }
                except (EventValidationError, ModelValidationError, KeyError) as exc:
                    update_error = str(exc)
            if update_error is not None:
                force_fallback_only = True
                event_record = {
                    "designated_event": designated_event,
                    "status": "failed",
                    "error": update_error,
                }

        if observations:
            policy = evidence_policy or EvidencePolicy()
            for observation in observations:
                authorization_state, record = refine_information_partition(
                    authorization_state, observation, policy
                )
                revision_records.append(record)

        checker = authorization_state.model.checker()
        epi_permitted: list[str] = []
        epi_blocked: list[str] = []
        witnesses: dict[str, Any] = {}
        for action in action_library.actions:
            if checker.satisfies(authorization_state.world, action.pre_epi):
                epi_permitted.append(action.name)
            else:
                epi_blocked.append(action.name)
                witness = checker.counterexample(
                    authorization_state.world, action.pre_epi
                )
                witnesses[action.name] = (
                    witness.to_dict() if witness is not None else {"reason": "false"}
                )

        env_permitted: list[str] = []
        env_blocked: list[str] = []
        for name in epi_permitted:
            action = action_library.get(name)
            if checker.satisfies(authorization_state.world, action.pre_env):
                env_permitted.append(name)
            else:
                env_blocked.append(name)

        admissible: list[str] = []
        gov_rejected: dict[str, Any] = {}
        governance_results: dict[str, GovernanceEvaluation] = {}
        for name in env_permitted:
            action = action_library.get(name)
            evaluation = governance_policy.evaluate(
                authorization_state, action, {"decision_id": decision_id}
            )
            governance_results[name] = evaluation
            if evaluation.passed:
                admissible.append(name)
            else:
                gov_rejected[name] = evaluation.to_dict()

        ordinary_admissible = [
            name
            for name in admissible
            if not action_library.get(name).is_fallback
        ]
        fallback_admissible = [
            name
            for name in fallback_priority
            if name in admissible and action_library.get(name).is_fallback
        ]

        selected: str | None = None
        status: DecisionStatus
        if force_fallback_only:
            selected = fallback_admissible[0] if fallback_admissible else None
            status = (
                self._fallback_status(selected)
                if selected is not None
                else DecisionStatus.UPDATE_ERROR
            )
        elif mode is DecisionMode.SHIELD and proposal in admissible:
            selected = proposal
            status = DecisionStatus.PROPOSAL_AUTHORIZED
        elif mode is DecisionMode.OPTIMIZE and ordinary_admissible:
            selected = max(
                ordinary_admissible,
                key=lambda name: (
                    utility.get(name, -action_library.get(name).cost),
                    -action_library.get(name).risk,
                    name,
                ),
            )
            status = DecisionStatus.ORDINARY_ACTION_SELECTED
        elif fallback_admissible:
            selected = fallback_admissible[0]
            status = self._fallback_status(selected)
        else:
            status = DecisionStatus.NO_ADMISSIBLE_ACTION

        entry = self._trace_entry(
            decision_id=decision_id,
            prior_state=prior_state,
            authorization_state=authorization_state,
            action_library=action_library,
            governance_policy=governance_policy,
            mode=mode,
            proposal=proposal,
            epi_permitted=tuple(sorted(epi_permitted)),
            epi_blocked=tuple(sorted(epi_blocked)),
            witnesses=witnesses,
            env_permitted=tuple(sorted(env_permitted)),
            env_blocked=tuple(sorted(env_blocked)),
            admissible=tuple(sorted(admissible)),
            gov_rejected=gov_rejected,
            selected=selected,
            status=status,
            fallback_priority=fallback_priority,
            utility=utility,
            random_seed=random_seed,
            event_record=event_record,
            revision_records=tuple(record.to_dict() for record in revision_records),
        )
        trace.append_authorization(entry)
        return AuthorizationOutcome(
            decision_id=decision_id,
            prior_state=prior_state,
            authorization_state=authorization_state,
            selected_action=selected,
            status=status,
            epistemically_permitted=tuple(sorted(epi_permitted)),
            epistemically_blocked=tuple(sorted(epi_blocked)),
            environmentally_permitted=tuple(sorted(env_permitted)),
            environmentally_blocked=tuple(sorted(env_blocked)),
            admissible=tuple(sorted(admissible)),
            governance_rejected=tuple(sorted(gov_rejected)),
            proposal=proposal,
            trace_entry=entry,
        )

    def execute(
        self,
        authorization: AuthorizationOutcome,
        action_library: ActionLibrary,
        registry: TransitionRegistry,
        trace: DecisionTrace,
        *,
        random_seed: int | None = None,
    ) -> ExecutionOutcome:
        if authorization.selected_action is None:
            entry = ExecutionTraceEntry(
                decision_id=authorization.decision_id,
                action=None,
                transition_id=None,
                status=DecisionStatus.HALTED.value,
                prior_state_id=authorization.authorization_state.state_id,
                resulting_state_id=authorization.authorization_state.state_id,
                details={"reason": authorization.status.value},
            )
            trace.append_execution(entry)
            return ExecutionOutcome(
                authorization,
                authorization.authorization_state,
                DecisionStatus.HALTED,
                entry.details,
            )

        action = action_library.get(authorization.selected_action)
        rng = random.Random(random_seed)
        try:
            resulting_state, details = registry.execute(
                authorization.authorization_state, action, rng
            )
            status = DecisionStatus.EXECUTION_SUCCEEDED
        except Exception as exc:  # fail visibly; the authorization remains recorded
            resulting_state = authorization.authorization_state
            details = {"error": repr(exc)}
            status = DecisionStatus.EXECUTION_FAILED
        entry = ExecutionTraceEntry(
            decision_id=authorization.decision_id,
            action=action.name,
            transition_id=action.transition_id,
            status=status.value,
            prior_state_id=authorization.authorization_state.state_id,
            resulting_state_id=resulting_state.state_id,
            details=details,
        )
        trace.append_execution(entry)
        return ExecutionOutcome(authorization, resulting_state, status, details)

    def _fallback_status(self, action: str | None) -> DecisionStatus:
        mapping = {
            "request_review": DecisionStatus.REVIEW_SELECTED,
            "request_evidence": DecisionStatus.EVIDENCE_REQUEST_SELECTED,
            "wait": DecisionStatus.WAIT_SELECTED,
            "safe_shutdown": DecisionStatus.SAFE_SHUTDOWN_SELECTED,
        }
        return mapping.get(action, DecisionStatus.FALLBACK_SELECTED)

    def _trace_entry(
        self,
        *,
        decision_id: str,
        prior_state: PointedState,
        authorization_state: PointedState,
        action_library: ActionLibrary,
        governance_policy: GovernancePolicy,
        mode: DecisionMode,
        proposal: str | None,
        epi_permitted: tuple[str, ...],
        epi_blocked: tuple[str, ...],
        witnesses: Mapping[str, Any],
        env_permitted: tuple[str, ...],
        env_blocked: tuple[str, ...],
        admissible: tuple[str, ...],
        gov_rejected: Mapping[str, Any],
        selected: str | None,
        status: DecisionStatus,
        fallback_priority: tuple[str, ...],
        utility: Mapping[str, float],
        random_seed: int | None,
        event_record: Mapping[str, Any],
        revision_records: tuple[Mapping[str, Any], ...],
    ) -> AuthorizationTraceEntry:
        replay_bundle = {
            "authorization_state": authorization_state.to_dict(),
            "action_library": action_library.to_dict(),
            "governance_policy": governance_policy.to_dict(),
            "mode": mode.value,
            "proposal": proposal,
            "fallback_priority": list(fallback_priority),
            "utility_by_action": dict(utility),
            "random_seed": random_seed,
        }
        return AuthorizationTraceEntry(
            decision_id=decision_id,
            prior_state_id=prior_state.state_id,
            authorization_state_id=authorization_state.state_id,
            prior_model_id=prior_state.model.model_id,
            authorization_model_id=authorization_state.model.model_id,
            event_record=dict(event_record),
            revision_records=revision_records,
            action_library_id=action_library.library_id,
            governance_policy_id=governance_policy.policy_id,
            mode=mode.value,
            proposal=proposal,
            epistemically_permitted=epi_permitted,
            epistemically_blocked=epi_blocked,
            epistemic_witnesses=dict(witnesses),
            environmentally_permitted=env_permitted,
            environmentally_blocked=env_blocked,
            governance_permitted=admissible,
            governance_rejected=dict(gov_rejected),
            selected_action=selected,
            status=status.value,
            fallback_priority=fallback_priority,
            utility_by_action=dict(utility),
            random_seed=random_seed,
            replay_bundle=replay_bundle,
        )
