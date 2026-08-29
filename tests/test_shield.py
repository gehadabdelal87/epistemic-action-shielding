from eas_shield.metrics import decision_record
from eas_shield.scenario_generation import GateScenarioParameters, generate_gate_scenario
from eas_shield.shield import DecisionMode, DecisionStatus, DecisionTrace, EASDecisionEngine
from eas_shield.variants import action_library_for_condition, governance_policy_for_condition


def params(**updates):
    base = dict(
        safe=True,
        robot_knows_safe=True,
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
    base.update(updates)
    return GateScenarioParameters(**base)


def authorize(
    scenario,
    actions=None,
    governance=None,
    *,
    mode=DecisionMode.OPTIMIZE,
    proposal=None,
):
    engine = EASDecisionEngine()
    return engine.authorize(
        state=scenario.state,
        action_library=actions or scenario.action_library,
        governance_policy=governance or scenario.governance_policy,
        trace=DecisionTrace(),
        mode=mode,
        proposal=proposal,
        utility_by_action=scenario.utility_by_action,
        fallback_priority=scenario.fallback_priority,
        decision_id="test",
    )


def test_eas_selects_admissible_open():
    scenario = generate_gate_scenario(seed=1, index=1, parameters=params())
    outcome = authorize(scenario)
    assert outcome.selected_action == "autonomous_open"
    assert outcome.selected_action in outcome.admissible


def test_eas_uses_fallback_when_robot_lacks_knowledge():
    scenario = generate_gate_scenario(
        seed=1, index=2, parameters=params(robot_knows_safe=False)
    )
    outcome = authorize(scenario)
    assert outcome.selected_action in scenario.fallback_priority
    assert "autonomous_open" in outcome.epistemically_blocked


def test_external_policy_proposal_is_authorized_when_admissible():
    scenario = generate_gate_scenario(seed=2, index=1, parameters=params())
    outcome = authorize(
        scenario,
        mode=DecisionMode.SHIELD,
        proposal="autonomous_open",
    )
    assert outcome.proposal == "autonomous_open"
    assert outcome.selected_action == "autonomous_open"
    assert outcome.status is DecisionStatus.PROPOSAL_AUTHORIZED


def test_external_policy_proposal_is_intercepted_when_knowledge_is_missing():
    scenario = generate_gate_scenario(
        seed=2,
        index=2,
        parameters=params(robot_knows_safe=False),
    )
    outcome = authorize(
        scenario,
        mode=DecisionMode.SHIELD,
        proposal="autonomous_open",
    )
    assert outcome.proposal == "autonomous_open"
    assert "autonomous_open" in outcome.epistemically_blocked
    assert outcome.selected_action != "autonomous_open"
    assert outcome.selected_action in scenario.fallback_priority


def test_no_gate_can_violate_canonical_epistemic_precondition():
    scenario = generate_gate_scenario(
        seed=1, index=3, parameters=params(robot_knows_safe=False)
    )
    actions = action_library_for_condition(scenario.action_library, "no_gate")
    outcome = authorize(scenario, actions=actions)
    record = decision_record(
        scenario_id=scenario.scenario_id,
        seed=1,
        condition="no_gate",
        outcome=outcome,
        canonical_actions=scenario.action_library,
        canonical_governance=scenario.governance_policy,
        utility_by_action=scenario.utility_by_action,
        authorization_ns=1,
    )
    assert outcome.selected_action == "autonomous_open"
    assert record.epistemic_violation


def test_no_governance_can_violate_canonical_policy():
    scenario = generate_gate_scenario(
        seed=1,
        index=4,
        parameters=params(provenance_known=False),
    )
    governance = governance_policy_for_condition(
        scenario.governance_policy, "no_governance"
    )
    outcome = authorize(scenario, governance=governance)
    record = decision_record(
        scenario_id=scenario.scenario_id,
        seed=1,
        condition="no_governance",
        outcome=outcome,
        canonical_actions=scenario.action_library,
        canonical_governance=scenario.governance_policy,
        utility_by_action=scenario.utility_by_action,
        authorization_ns=1,
    )
    assert outcome.selected_action == "autonomous_open"
    assert record.governance_violation


def test_no_admissible_fallback_returns_halt():
    scenario = generate_gate_scenario(
        seed=1,
        index=5,
        parameters=params(
            robot_knows_safe=False,
            communication_available=False,
            evidence_source_reachable=False,
            waiting_safe=False,
            shutdown_available=False,
        ),
    )
    outcome = authorize(scenario)
    assert outcome.selected_action is None
    assert outcome.status.value == "no_admissible_action"
