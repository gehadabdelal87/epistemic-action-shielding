"""Upstream proposal policies, tabular Q-learning, and DQN integration."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol, Sequence

from .actions import ActionLibrary, ActionSchema
from .model import PointedState


class ProposalPolicy(Protocol):
    def propose(
        self,
        state: PointedState,
        action_library: ActionLibrary,
        history: Sequence[Mapping[str, Any]],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class RuleProposalPolicy:
    """Transparent confidence/value/risk proposal generator.

    This policy is deliberately rule based. It remains useful as an interface
    test because its proposals can be inspected independently of EAS.
    """

    confidence_key: str = "confidence"
    confidence_threshold: float = 0.65
    predicted_value_key: str = "predicted_value"
    max_risk: float = 10.0
    default_action: str = "request_review"

    def propose(
        self,
        state: PointedState,
        action_library: ActionLibrary,
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        candidates = []
        confidence = float(state.metadata.get(self.confidence_key, 0.0))
        predicted = state.metadata.get(self.predicted_value_key, {})
        if not isinstance(predicted, Mapping):
            predicted = {}
        for action in action_library.ordinary:
            if action.risk > self.max_risk:
                continue
            value = float(predicted.get(action.name, -action.cost))
            if action.name.endswith("open") and confidence < self.confidence_threshold:
                continue
            candidates.append((value, -action.risk, action.name))
        if candidates:
            return max(candidates)[2]
        if self.default_action in action_library.by_name:
            return self.default_action
        return action_library.fallbacks[0].name if action_library.fallbacks else action_library.actions[0].name


@dataclass(frozen=True, slots=True)
class UtilityPolicy:
    utility: Mapping[str, float]

    def choose(self, actions: Iterable[ActionSchema]) -> str:
        actions = tuple(actions)
        if not actions:
            raise ValueError("Cannot choose from an empty action set.")
        return max(
            actions,
            key=lambda action: (
                float(self.utility.get(action.name, -action.cost)),
                -action.risk,
                action.name,
            ),
        ).name


StateEncoder = Callable[[Any], Hashable]


@dataclass(slots=True)
class TabularQPolicy:
    """Legacy tabular policy retained for backward-compatible experiments."""

    actions: tuple[str, ...]
    q_values: dict[str, dict[str, float]] = field(default_factory=dict)
    epsilon: float = 0.0
    seed: int = 0

    def _key(self, observation: Hashable) -> str:
        return json.dumps(observation, sort_keys=True, default=str)

    def values(self, observation: Hashable) -> dict[str, float]:
        key = self._key(observation)
        table = self.q_values.setdefault(key, {})
        for action in self.actions:
            table.setdefault(action, 0.0)
        return table

    def act(self, observation: Hashable, *, explore: bool | None = None) -> str:
        key_bytes = self._key(observation).encode("utf-8")
        seed_offset = int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], "big")
        rng = random.Random(self.seed + seed_offset)
        use_exploration = self.epsilon > 0 if explore is None else explore
        if use_exploration and rng.random() < self.epsilon:
            return rng.choice(self.actions)
        values = self.values(observation)
        best = max(values.values())
        return min(action for action, value in values.items() if value == best)

    def update(
        self,
        observation: Hashable,
        action: str,
        reward: float,
        next_observation: Hashable,
        terminal: bool,
        *,
        alpha: float,
        gamma: float,
    ) -> None:
        current = self.values(observation)
        next_values = self.values(next_observation)
        target = reward if terminal else reward + gamma * max(next_values.values())
        current[action] += alpha * (target - current[action])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "actions": list(self.actions),
                    "q_values": self.q_values,
                    "epsilon": self.epsilon,
                    "seed": self.seed,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularQPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            actions=tuple(data["actions"]),
            q_values={
                str(key): {str(action): float(value) for action, value in row.items()}
                for key, row in data.get("q_values", {}).items()
            },
            epsilon=float(data.get("epsilon", 0.0)),
            seed=int(data.get("seed", 0)),
        )


def train_q_learning(
    env_factory: Callable[[int], Any],
    actions: tuple[str, ...],
    *,
    episodes: int = 5_000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
) -> TabularQPolicy:
    """Train the legacy tabular Q-learning baseline."""
    policy = TabularQPolicy(actions=actions, epsilon=epsilon_start, seed=seed)
    rng = random.Random(seed)
    for episode in range(episodes):
        fraction = episode / max(1, episodes - 1)
        policy.epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        env = env_factory(rng.randrange(2**31 - 1))
        observation = env.reset()
        terminal = False
        while not terminal:
            action = policy.act(observation, explore=True)
            next_observation, reward, terminal, _info = env.step(action)
            policy.update(
                observation,
                action,
                reward,
                next_observation,
                terminal,
                alpha=alpha,
                gamma=gamma,
            )
            observation = next_observation
    policy.epsilon = 0.0
    return policy


# ---------------------------------------------------------------------------
# Deep Q-Network policy used by the learned-policy integration experiment.
# Torch is imported lazily so the formal EAS core remains importable without
# the optional experiment dependencies installed.
# ---------------------------------------------------------------------------


def _torch_modules():
    try:
        import numpy as np
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "DQN support requires the experiment dependencies. "
            "Install with `pip install -e '.[experiments]'`."
        ) from exc
    return np, torch, nn


@dataclass(frozen=True, slots=True)
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    replay_capacity: int = 10_000
    min_replay_size: int = 256
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    gradient_clip_norm: float = 10.0
    torch_num_threads: int = 1

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if self.batch_size <= 0 or self.replay_capacity <= 0:
            raise ValueError("batch_size and replay_capacity must be positive")
        if self.min_replay_size < self.batch_size:
            raise ValueError("min_replay_size must be at least batch_size")
        if self.min_replay_size > self.replay_capacity:
            raise ValueError("min_replay_size may not exceed replay_capacity")
        if self.target_update_interval <= 0 or self.epsilon_decay_steps <= 0:
            raise ValueError("update and decay intervals must be positive")
        if self.torch_num_threads <= 0:
            raise ValueError("torch_num_threads must be positive")


class _QNetwork:
    """Small MLP wrapper that avoids importing torch at package import time."""

    def __init__(self, observation_dim: int, action_count: int, hidden_sizes: tuple[int, ...]):
        _np, _torch, nn = _torch_modules()
        layers: list[Any] = []
        previous = observation_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, action_count))
        self.module = nn.Sequential(*layers)


@dataclass(slots=True)
class DQNPolicy:
    """A frozen-or-trainable DQN action proposer.

    EAS never reads or changes the network parameters. During shielded
    evaluation the policy only supplies a proposal and its Q-values.
    """

    actions: tuple[str, ...]
    observation_dim: int
    config: DQNConfig = field(default_factory=DQNConfig)
    seed: int = 0
    _network: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _np, torch, _nn = _torch_modules()
        torch.manual_seed(self.seed)
        self._network = _QNetwork(
            self.observation_dim, len(self.actions), self.config.hidden_sizes
        ).module
        self._network.eval()

    @property
    def network(self) -> Any:
        return self._network

    def q_array(self, observation: Sequence[float]) -> Any:
        np, torch, _nn = _torch_modules()
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected observation shape {(self.observation_dim,)}, got {array.shape}."
            )
        with torch.no_grad():
            tensor = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
            values = self._network(tensor).squeeze(0).cpu().numpy()
        return values

    def q_values(self, observation: Sequence[float]) -> dict[str, float]:
        values = self.q_array(observation)
        return {action: float(values[index]) for index, action in enumerate(self.actions)}

    def act(self, observation: Sequence[float], *, explore: bool = False) -> str:
        if explore:
            raise ValueError(
                "DQNPolicy.act() is deterministic. Exploration is handled by train_dqn()."
            )
        values = self.q_array(observation)
        best_index = int(values.argmax())
        return self.actions[best_index]

    def save(self, path: str | Path) -> None:
        _np, torch, _nn = _torch_modules()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actions": list(self.actions),
                "observation_dim": self.observation_dim,
                "config": asdict(self.config),
                "seed": self.seed,
                "state_dict": self._network.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DQNPolicy":
        _np, torch, _nn = _torch_modules()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config_data = dict(payload["config"])
        config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
        policy = cls(
            actions=tuple(payload["actions"]),
            observation_dim=int(payload["observation_dim"]),
            config=DQNConfig(**config_data),
            seed=int(payload["seed"]),
        )
        policy._network.load_state_dict(payload["state_dict"])
        policy._network.eval()
        return policy


@dataclass(frozen=True, slots=True)
class DQNTrainingSummary:
    seed: int
    episodes: int
    environment_steps: int
    gradient_steps: int
    final_epsilon: float
    mean_return_last_100: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_dqn(
    env_factory: Callable[[int], Any],
    actions: tuple[str, ...],
    *,
    episodes: int = 5_000,
    seed: int = 0,
    config: DQNConfig | None = None,
) -> tuple[DQNPolicy, DQNTrainingSummary]:
    """Train a DQN independently of EAS.

    The environment must expose ``reset()`` and ``step(action)`` using the same
    lightweight interface as the existing research environments. The function
    returns a policy in evaluation mode plus a compact training summary.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    config = config or DQNConfig()
    np, torch, nn = _torch_modules()

    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(config.torch_num_threads)
    torch.manual_seed(seed)
    try:  # deterministic CPU execution when supported
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    seed_rng = random.Random(seed)
    first_env = env_factory(seed_rng.randrange(2**31 - 1))
    first_observation = np.asarray(first_env.reset(), dtype=np.float32)
    if first_observation.ndim != 1:
        raise ValueError("DQN observations must be one-dimensional numeric vectors")

    policy = DQNPolicy(
        actions=actions,
        observation_dim=int(first_observation.shape[0]),
        config=config,
        seed=seed,
    )
    online = policy.network
    target = _QNetwork(
        policy.observation_dim, len(actions), config.hidden_sizes
    ).module
    target.load_state_dict(online.state_dict())
    target.eval()
    online.train()

    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    replay: deque[tuple[Any, int, float, Any, bool]] = deque(
        maxlen=config.replay_capacity
    )
    action_to_index = {action: index for index, action in enumerate(actions)}
    exploration_rng = random.Random(seed + 17_071)

    environment_steps = 0
    gradient_steps = 0
    returns: list[float] = []

    def epsilon_at(step: int) -> float:
        fraction = min(1.0, step / config.epsilon_decay_steps)
        return config.epsilon_start + fraction * (
            config.epsilon_end - config.epsilon_start
        )

    for _episode in range(episodes):
        env = env_factory(seed_rng.randrange(2**31 - 1))
        observation = np.asarray(env.reset(), dtype=np.float32)
        terminal = False
        episode_return = 0.0

        while not terminal:
            epsilon = epsilon_at(environment_steps)
            if exploration_rng.random() < epsilon:
                action_index = exploration_rng.randrange(len(actions))
            else:
                with torch.no_grad():
                    q_values = online(
                        torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                    )
                    action_index = int(q_values.argmax(dim=1).item())
            action = actions[action_index]
            next_observation, reward, terminal, _info = env.step(action)
            next_array = np.asarray(next_observation, dtype=np.float32)
            replay.append(
                (
                    observation.copy(),
                    action_index,
                    float(reward),
                    next_array.copy(),
                    bool(terminal),
                )
            )
            observation = next_array
            episode_return += float(reward)
            environment_steps += 1

            if len(replay) >= config.min_replay_size:
                batch = exploration_rng.sample(list(replay), config.batch_size)
                obs_batch = torch.as_tensor(
                    np.stack([row[0] for row in batch]), dtype=torch.float32
                )
                action_batch = torch.as_tensor(
                    [row[1] for row in batch], dtype=torch.int64
                )
                reward_batch = torch.as_tensor(
                    [row[2] for row in batch], dtype=torch.float32
                )
                next_batch = torch.as_tensor(
                    np.stack([row[3] for row in batch]), dtype=torch.float32
                )
                terminal_batch = torch.as_tensor(
                    [row[4] for row in batch], dtype=torch.float32
                )

                predicted = online(obs_batch).gather(
                    1, action_batch.unsqueeze(1)
                ).squeeze(1)
                with torch.no_grad():
                    next_value = target(next_batch).max(dim=1).values
                    target_value = reward_batch + (
                        1.0 - terminal_batch
                    ) * config.gamma * next_value

                loss = loss_fn(predicted, target_value)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    online.parameters(), config.gradient_clip_norm
                )
                optimizer.step()
                gradient_steps += 1

                if environment_steps % config.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())

        returns.append(episode_return)

    online.eval()
    final_epsilon = epsilon_at(environment_steps)
    tail = returns[-min(100, len(returns)) :]
    summary = DQNTrainingSummary(
        seed=seed,
        episodes=episodes,
        environment_steps=environment_steps,
        gradient_steps=gradient_steps,
        final_epsilon=final_epsilon,
        mean_return_last_100=float(sum(tail) / len(tail)),
    )
    return policy, summary
