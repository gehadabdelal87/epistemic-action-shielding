from dataclasses import replace

from eas_shield.replay import replay_authorization
from eas_shield.scenario_generation import GateScenarioParameters, generate_gate_scenario
from eas_shield.shield import DecisionMode, DecisionTrace, EASDecisionEngine


def test_trace_replays_exactly():
    scenario = generate_gate_scenario(seed=10, index=1)
    outcome = EASDecisionEngine().authorize(
        state=scenario.state,
        action_library=scenario.action_library,
        governance_policy=scenario.governance_policy,
        trace=DecisionTrace(),
        mode=DecisionMode.OPTIMIZE,
        utility_by_action=scenario.utility_by_action,
        fallback_priority=scenario.fallback_priority,
        decision_id="replay-test",
    )
    replay = replay_authorization(outcome.trace_entry)
    assert replay.success, replay.differences


def test_shielded_external_proposal_replays_exactly():
    parameters = GateScenarioParameters(
        safe=True,
        robot_knows_safe=False,
        operator_knows_safe=True,
        confidence=0.9,
        source_reliability=0.9,
        observation_quality=0.9,
        provenance_known=True,
        autonomous_risk=5.0,
        coordinated_risk=5.0,
        communication_available=True,
        evidence_source_reachable=True,
        waiting_safe=True,
        shutdown_available=True,
        reviewer_authorized=True,
    )
    scenario = generate_gate_scenario(seed=10, index=3, parameters=parameters)
    outcome = EASDecisionEngine().authorize(
        state=scenario.state,
        action_library=scenario.action_library,
        governance_policy=scenario.governance_policy,
        trace=DecisionTrace(),
        mode=DecisionMode.SHIELD,
        proposal="autonomous_open",
        utility_by_action=scenario.utility_by_action,
        fallback_priority=scenario.fallback_priority,
        decision_id="replay-shielded-proposal",
    )
    assert outcome.selected_action != outcome.proposal
    replay = replay_authorization(outcome.trace_entry)
    assert replay.success, replay.differences
    assert replay.recomputed_entry.proposal == "autonomous_open"


def test_corrupted_trace_is_detected():
    scenario = generate_gate_scenario(seed=10, index=2)
    outcome = EASDecisionEngine().authorize(
        state=scenario.state,
        action_library=scenario.action_library,
        governance_policy=scenario.governance_policy,
        trace=DecisionTrace(),
        mode=DecisionMode.OPTIMIZE,
        utility_by_action=scenario.utility_by_action,
        fallback_priority=scenario.fallback_priority,
        decision_id="replay-corrupt",
    )
    corrupted = replace(outcome.trace_entry, selected_action="not_the_action")
    replay = replay_authorization(corrupted)
    assert not replay.success
    assert "selected_action" in replay.differences
