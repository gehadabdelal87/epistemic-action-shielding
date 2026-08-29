from eas_shield.environment import DQNGateEnv
from eas_shield.formulas import Atom, Knows
from eas_shield.scenario_generation import generate_gate_scenario
from eas_shield.shield import DecisionMode, DecisionTrace, EASDecisionEngine


def test_randomized_architectural_invariants():
    engine = EASDecisionEngine()
    for index in range(100):
        scenario = generate_gate_scenario(seed=2026, index=index)
        model = scenario.state.model
        checker = model.checker()
        for agent in model.agents:
            if checker.satisfies(scenario.state.world, Knows(agent, Atom("safe"))):
                assert checker.satisfies(scenario.state.world, Atom("safe"))
        outcome = engine.authorize(
            state=scenario.state,
            action_library=scenario.action_library,
            governance_policy=scenario.governance_policy,
            trace=DecisionTrace(),
            mode=DecisionMode.OPTIMIZE,
            utility_by_action=scenario.utility_by_action,
            fallback_priority=scenario.fallback_priority,
            decision_id=f"property-{index}",
        )
        assert set(outcome.admissible).issubset(outcome.environmentally_permitted)
        assert set(outcome.environmentally_permitted).issubset(outcome.epistemically_permitted)
        if outcome.selected_action is not None:
            assert outcome.selected_action in outcome.admissible


def test_noisy_dqn_sensor_never_counts_as_knowledge_before_verification():
    for seed in range(100):
        env = DQNGateEnv(
            seed,
            sensor_accuracy=0.80,
            reported_sensor_accuracy=0.95,
            verification_probability=0.60,
        )
        env.reset()
        state = env.to_pointed_state(confidence_threshold=0.80)
        checker = state.model.checker()

        # Even a .95 scalar confidence does not collapse the two-world
        # information cell before a verification event occurs.
        assert not checker.satisfies(state.world, Knows("robot", Atom("safe")))


def test_verified_dqn_evidence_preserves_factivity_across_random_seeds():
    for seed in range(100):
        env = DQNGateEnv(seed, verification_probability=1.0)
        env.reset()
        env.step("inspect")
        state = env.to_pointed_state()
        checker = state.model.checker()
        knows_safe = checker.satisfies(state.world, Knows("robot", Atom("safe")))
        if knows_safe:
            assert checker.satisfies(state.world, Atom("safe"))
