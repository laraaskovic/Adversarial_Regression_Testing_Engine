import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .action_space import ActionInstance, ActionSpace
from .environment import BackendEnvironment, Observation


@dataclass
class EpisodeStep:
    step: int
    action: ActionInstance
    observation: Observation
    reward: float

    def to_dict(self) -> Dict:
        data = {
            "step": self.step,
            "action": asdict(self.action),
            "reward": self.reward,
            "status_code": self.observation.status_code,
            "latency_ms": self.observation.latency_ms,
            "state_signature": self.observation.state_signature(),
            "anomaly_markers": self.observation.anomaly_markers(),
            "response_json": self.observation.response_json,
            "state": self.observation.state,
            "log_excerpt": self.observation.log_excerpt,
        }
        return data


@dataclass
class EpisodeLog:
    seed: int
    steps: List[EpisodeStep]
    start_ts: float
    base_url: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "base_url": self.base_url,
                "start_ts": self.start_ts,
                "steps": [step.to_dict() for step in self.steps],
            },
            indent=2,
        )

    def anomalies(self) -> List[EpisodeStep]:
        return [step for step in self.steps if step.observation.anomaly_markers()]


class NoveltyArchive:
    """Tracks previously seen states and anomalies for reward shaping."""

    def __init__(self) -> None:
        self.state_signatures: set[str] = set()
        self.anomaly_signatures: set[str] = set()

    def update(self, obs: Observation) -> None:
        self.state_signatures.add(obs.state_signature())
        markers = obs.anomaly_markers()
        if markers:
            self.anomaly_signatures.add("|".join(sorted(markers)))

    def is_new_state(self, obs: Observation) -> bool:
        return obs.state_signature() not in self.state_signatures

    def is_new_anomaly(self, obs: Observation) -> bool:
        markers = obs.anomaly_markers()
        if not markers:
            return False
        return "|".join(sorted(markers)) not in self.anomaly_signatures


class RewardModel:
    """Reward favors novelty and correctness violations over raw coverage."""

    def __init__(self, novelty_weight: float = 1.0, anomaly_weight: float = 2.0, latency_weight: float = 0.5):
        self.novelty_weight = novelty_weight
        self.anomaly_weight = anomaly_weight
        self.latency_weight = latency_weight

    def score(self, obs: Observation, archive: NoveltyArchive) -> float:
        reward = 0.0
        if archive.is_new_state(obs):
            reward += self.novelty_weight
        if obs.anomaly_markers():
            reward += self.anomaly_weight * len(obs.anomaly_markers())
        if archive.is_new_anomaly(obs):
            reward += self.anomaly_weight
        if obs.latency_ms > 250:
            reward += self.latency_weight
        if obs.status_code >= 400:
            reward += 0.3
        return reward


class EpsilonGreedyPolicy:
    """Simple exploration policy; keeps sophistication low but directed."""

    def __init__(self, epsilon: float = 0.25):
        self.epsilon = epsilon
        self.action_stats: Dict[str, Dict[str, float]] = {}

    def select(self, space: ActionSpace, rng: random.Random) -> ActionInstance:
        templates = space.all_templates()
        if rng.random() < self.epsilon:
            return rng.choice(templates).sample(rng)

        # Choose the highest average reward action; fall back to random when unseen.
        scored = [(name, stats.get("avg", 0.0)) for name, stats in self.action_stats.items()]
        if not scored:
            return rng.choice(templates).sample(rng)
        best_name = max(scored, key=lambda kv: kv[1])[0]
        candidates = [t for t in templates if t.name == best_name]
        if not candidates:
            return rng.choice(templates).sample(rng)
        return candidates[0].sample(rng)

    def observe(self, action_name: str, reward: float) -> None:
        stats = self.action_stats.setdefault(action_name, {"count": 0, "avg": 0.0})
        count = stats["count"]
        stats["count"] = count + 1
        # Running average keeps the policy stable without storing full history.
        stats["avg"] = (stats["avg"] * count + reward) / (count + 1)


class ReplayWriter:
    """Persists episodes that surface anomalies so they can be replayed."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def persist(self, episode: EpisodeLog) -> Path:
        filename = f"episode_seed{episode.seed}_{int(episode.start_ts)}.json"
        path = self.artifact_dir / filename
        path.write_text(episode.to_json(), encoding="utf-8")
        return path


class Explorer:
    """Drives the environment, logs every step, and captures failures."""

    def __init__(
        self,
        env: BackendEnvironment,
        action_space: ActionSpace,
        reward_model: RewardModel,
        policy: Optional[EpsilonGreedyPolicy] = None,
        artifact_dir: Path = Path("artifacts/episodes"),
    ) -> None:
        self.env = env
        self.action_space = action_space
        self.reward_model = reward_model
        self.policy = policy or EpsilonGreedyPolicy()
        self.replay_writer = ReplayWriter(artifact_dir)

    def run_episode(self, steps: int, seed: int) -> EpisodeLog:
        rng = random.Random(seed)
        archive = NoveltyArchive()
        episode = EpisodeLog(seed=seed, steps=[], start_ts=time.time(), base_url=self.env.base_url)

        # Always start from a clean state.
        reset_obs = self.env.reset()
        archive.update(reset_obs)
        self.policy.observe("reset", 0.0)

        for step_idx in range(steps):
            action = self.policy.select(self.action_space, rng)
            obs = self.env.perform(action)
            reward = self.reward_model.score(obs, archive)
            archive.update(obs)
            self.policy.observe(action.name, reward)

            episode.steps.append(EpisodeStep(step=step_idx, action=action, observation=obs, reward=reward))

            if obs.anomaly_markers():
                # Store only interesting runs to keep artifacts small.
                self.replay_writer.persist(episode)

        return episode
