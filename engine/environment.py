import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .action_space import ActionInstance


@dataclass
class Observation:
    action: ActionInstance
    status_code: int
    latency_ms: float
    response_json: Dict[str, Any]
    state: Dict[str, Any]
    log_excerpt: List[Dict[str, Any]]

    def state_signature(self) -> str:
        # Stable signature to feed novelty detection; sorted keys avoids drift.
        summary = {
            "inventory": self.state.get("inventory", {}),
            "mode": self.state.get("mode"),
            "orders_total": self.state.get("orders_total"),
            "alerts": sorted(self.state.get("alerts", [])),
            "invariants": sorted(self.state.get("invariants", [])),
            "state_version": self.state.get("state_version"),
        }
        return json.dumps(summary, sort_keys=True)

    def anomaly_markers(self) -> List[str]:
        markers: List[str] = []
        if self.status_code >= 500:
            markers.append("http_5xx")
        if self.status_code >= 400:
            markers.append("http_4xx")
        if self.state.get("alerts"):
            markers.append("alerts_present")
        for inv in self.state.get("invariants", []):
            if "inventory_negative" in inv:
                markers.append("inventory_negative")
        if self.latency_ms > 250:
            markers.append("slow_response")
        return markers


class BackendEnvironment:
    """Wraps the web app so the explorer can treat it as an RL environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        # All network calls stay inside the provided base_url. The engine does
        # not discover new hosts to avoid drifting into external targets.

    def reset(self) -> Observation:
        action = ActionInstance(name="reset", method="POST", path="/reset", json={})
        return self.perform(action)

    def perform(self, action: ActionInstance) -> Observation:
        url = f"{self.base_url}{action.path}"
        start = time.perf_counter()
        resp = self._session.request(
            method=action.method,
            url=url,
            json=action.json,
            timeout=5,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        response_json: Dict[str, Any] = {}
        try:
            response_json = resp.json()
        except ValueError:
            response_json = {"raw_body": resp.text}

        state = self._fetch_state(fallback=response_json.get("state"))
        return Observation(
            action=action,
            status_code=resp.status_code,
            latency_ms=elapsed_ms,
            response_json=response_json,
            state=state,
            log_excerpt=state.get("recent_events", []),
        )

    def _fetch_state(self, fallback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure every observation carries a state snapshot.

        The server already exposes /state for explicit state representation;
        when that fails we fall back to whatever the action returned.
        """
        try:
            resp = self._session.get(f"{self.base_url}/state", timeout=5)
            data = resp.json()
            return data.get("state", fallback or {})
        except Exception:
            return fallback or {}
