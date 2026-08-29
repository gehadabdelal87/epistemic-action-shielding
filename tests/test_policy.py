from eas_shield.environment import DQNGateEnv, GateControlEnv
from eas_shield.formulas import Atom, Knows
from eas_shield.policy import DQNConfig, DQNPolicy, train_dqn, train_q_learning


def test_dqn_training_smoke_and_checkpoint(tmp_path):
    config = DQNConfig(
        hidden_sizes=(16, 16),
        batch_size=16,
        replay_capacity=512,
        min_replay_size=32,
        target_update_interval=25,
        epsilon_decay_steps=200,
    )
    policy, summary = train_dqn(
        lambda seed: DQNGateEnv(seed),
        DQNGateEnv.ACTIONS,
        episodes=120,
        seed=4,
        config=config,
    )
    env = DQNGateEnv(100)
    observation = env.reset()
    action = policy.act(observation)
    values = policy.q_values(observation)

    assert action in DQNGateEnv.ACTIONS
    assert set(values) == set(DQNGateEnv.ACTIONS)
    assert summary.environment_steps > 0
    assert summary.gradient_steps > 0

    checkpoint = tmp_path / "dqn.pt"
    policy.save(checkpoint)
    restored = DQNPolicy.load(checkpoint)
    assert restored.act(observation) == action
    assert restored.q_values(observation) == values


def test_dqn_observation_confidence_does_not_create_knowledge():
    env = DQNGateEnv(
        7,
        safe_probability=1.0,
        sensor_accuracy=1.0,
        reported_sensor_accuracy=0.95,
        verification_probability=1.0,
    )
    env.reset()
    initial = env.to_pointed_state(confidence_threshold=0.80)
    checker = initial.model.checker()

    assert initial.metadata["confidence"] == 0.95
    assert checker.satisfies(initial.world, Atom("safe"))
    assert not checker.satisfies(initial.world, Knows("robot", Atom("safe")))

    env.step("inspect")
    verified = env.to_pointed_state(confidence_threshold=0.80)
    assert verified.metadata["inspection_result"] == "verified_safe"
    assert verified.model.satisfies(
        verified.world, Knows("robot", Atom("safe"))
    )


def test_tabular_q_learning_remains_available_for_legacy_runs():
    policy = train_q_learning(
        lambda seed: GateControlEnv(seed),
        GateControlEnv.ACTIONS,
        episodes=50,
        seed=4,
    )
    env = GateControlEnv(100)
    observation = env.reset()
    assert policy.act(observation) in GateControlEnv.ACTIONS
