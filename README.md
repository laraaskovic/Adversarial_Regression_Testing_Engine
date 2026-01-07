# Adversarial Regression Testing Engine

This project couples a small stateful Flask app with an adversarial testing engine that explores legal HTTP actions to surface regressions, anomalies, and slow paths. The focus is observability and reproducibility, not penetration testing.

## Components
- `demo_app.py`: Inventory-like demo backend with explicit state representation (`/state` and `/diagnostics`), bounded in-memory log, and a deliberately flawed expedite path that can drive inventory negative.
- `engine/`: Testing engine that treats the backend as an environment.
  - `action_space.py`: Whitelisted, parameterized HTTP actions (reset, restock, drain, purchase, toggle mode).
  - `environment.py`: Executes actions, measures latency, and attaches state/log excerpts to each observation.
  - `explorer.py`: Epsilon-greedy search with a novelty/anomaly reward, stores replayable episodes.
  - `replay.py`: Deterministically replays stored episodes to reproduce failures.
- `run_engine.py`: CLI entrypoint to launch exploration runs and persist interesting episodes.

## Quickstart
1) Install deps (Python 3.10+ recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on *nix
pip install -r requirements.txt
```
2) Start the demo app (listens on `http://127.0.0.1:8000`):
```bash
python demo_app.py
```
3) Run the adversarial explorer from another shell (will auto-start the app if it is not reachable at the base URL):
```bash
python run_engine.py --episodes 5 --steps 15 --base-url http://127.0.0.1:8000
```
Replayable episodes are written to `artifacts/episodes/episode_seed*_*.json` whenever anomalies are observed (negative inventory, slow responses, HTTP errors, etc.).

4) Replay a stored failure deterministically:
```bash
python -m engine.replay artifacts/episodes/episode_seed42_*.json --base-url http://127.0.0.1:8000
```
The replay runner resets the app, replays the same action sequence, and reports whether anomalies persist.

## Interactive UI Demo
- Start the app (`python demo_app.py`) and open `http://127.0.0.1:8000` in a browser.
- The page shows live state from `/state`, buttons for mode/restock/purchase, and an action log.
- Click **Run demo** to watch a scripted "model" sequence move a cursor between controls, trigger the intentional oversell path, and surface alerts/slow mode.
- Use the manual controls to reproduce or explore other sequences; the state panel and alerts update after every action.

## Design Notes
- Explicit state: `/state` surfaces mode, inventory, alerts, invariant violations, and recent events so the engine can reason about transitions.
- Observability: Every observation includes status code, latency, state snapshot, and log excerpts; request latency is also emitted via `X-Request-Latency-ms`.
- Novelty-focused reward: Rewards new state signatures and anomalous markers more than raw coverage; epsilon-greedy keeps exploration simple and predictable.
- Deterministic replay: Seeds, actions, responses, and state signatures are stored verbatim in artifacts to support reliable reproduction of failures.
- Safety scope: Action space is fixed to the demo app paths; the engine never discovers or targets external hosts.
