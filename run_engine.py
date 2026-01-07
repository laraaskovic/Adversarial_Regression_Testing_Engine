import argparse
from pathlib import Path

from engine.action_space import ActionSpace
from engine.environment import BackendEnvironment
from engine.health import ensure_backend_available
from engine.explorer import Explorer, RewardModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial regression testing engine")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Target app base URL")
    parser.add_argument("--episodes", type=int, default=3, help="How many episodes to run")
    parser.add_argument("--steps", type=int, default=12, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/episodes"),
        help="Where to store replayable failure episodes",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Target app base URL")
    parser.add_argument(
        "--no-autostart",
        action="store_false",
        dest="autostart",
        help="Disable auto-starting demo_app.py when backend is unreachable",
    )
    parser.set_defaults(autostart=True)
    parser.add_argument("--app-script", default="demo_app.py", help="Path to demo app script for autostart")
    args = parser.parse_args()

    ensure_backend_available(args.base_url, autostart=args.autostart, app_script=args.app_script)
    env = BackendEnvironment(args.base_url)
    action_space = ActionSpace()
    reward_model = RewardModel()
    explorer = Explorer(env, action_space, reward_model, artifact_dir=args.artifacts)

    stored = 0
    for episode_idx in range(args.episodes):
        seed = args.seed + episode_idx
        episode = explorer.run_episode(steps=args.steps, seed=seed)
        anomalies = episode.anomalies()
        if anomalies:
            stored += 1
            print(
                f"[episode {episode_idx}] seed={seed} anomalies={len(anomalies)} "
                f"stored_at={args.artifacts}"
            )
        else:
            print(f"[episode {episode_idx}] seed={seed} no anomalies")

    print(f"Done. Stored {stored} replayable episodes in {args.artifacts}")


if __name__ == "__main__":
    main()
