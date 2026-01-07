import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .action_space import ActionInstance
from .environment import BackendEnvironment


def load_episode(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def replay(path: Path, base_url: Optional[str] = None) -> List[Dict]:
    episode = load_episode(path)
    target_base = base_url or episode.get("base_url") or "http://127.0.0.1:8000"
    env = BackendEnvironment(target_base)
    env.reset()
    results = []
    for step in episode.get("steps", []):
        action_data = step["action"]
        action = ActionInstance(
            name=action_data["name"],
            method=action_data["method"],
            path=action_data["path"],
            json=action_data["json"],
        )
        obs = env.perform(action)
        results.append(
            {
                "step": step["step"],
                "action": action_data,
                "expected_status": step["status_code"],
                "actual_status": obs.status_code,
                "anomaly_markers": obs.anomaly_markers(),
                "state_signature": obs.state_signature(),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically replay a stored episode.")
    parser.add_argument("episode_path", type=Path, help="Path to episode JSON produced by the explorer")
    parser.add_argument("--base-url", default=None, help="Override target app base URL")
    args = parser.parse_args()

    results = replay(args.episode_path, args.base_url)
    for row in results:
        print(
            f"step={row['step']} status={row['actual_status']} "
            f"expected={row['expected_status']} anomalies={row['anomaly_markers']}"
        )


if __name__ == "__main__":
    main()
