import logging
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List

from flask import Flask, jsonify, request, g


# The demo app is intentionally stateful to surface regression risks that only
# appear after certain sequences of actions (inventory oversell, slow mode).
# Flask keeps things simple while still giving us hooks for explicit state
# representation and observability.
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


class AppState:
    """Holds all mutable state plus a small event log for diagnostics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            self.inventory: Dict[str, int] = {"widgets": 6, "gadgets": 3, "doodads": 2}
            self.mode: str = "normal"  # normal | maintenance | slow
            self.orders: List[Dict[str, Any]] = []
            self.state_version: int = 0
            self.events: List[Dict[str, Any]] = []
            self.outstanding_alerts: List[str] = []
            self._record_event("reset", {"reason": "api"})
            return self.summary()

    def _record_event(self, name: str, detail: Dict[str, Any]) -> None:
        """Bounded in-memory log; also used for replay-friendly diagnostics."""
        event = {
            "name": name,
            "detail": detail,
            "state_version": self.state_version,
            "timestamp": time.time(),
        }
        self.events.append(event)
        if len(self.events) > 200:
            self.events.pop(0)
        logging.info("event=%s detail=%s", name, detail)

    def _bump_version(self) -> None:
        self.state_version += 1

    def summary(self, recent_events: int = 10) -> Dict[str, Any]:
        with self._lock:
            return {
                "inventory": deepcopy(self.inventory),
                "mode": self.mode,
                "orders": deepcopy(self.orders[-5:]),
                "orders_total": len(self.orders),
                "state_version": self.state_version,
                "alerts": list(self.outstanding_alerts),
                "invariants": self._invariant_flags(),
                "recent_events": deepcopy(self.events[-recent_events:]),
            }

    def _invariant_flags(self) -> List[str]:
        flags: List[str] = []
        for item, qty in self.inventory.items():
            if qty < 0:
                flags.append(f"inventory_negative:{item}")
        if self.mode == "maintenance" and self.orders:
            flags.append("orders_in_maintenance")
        if self.mode == "slow":
            flags.append("slow_mode")
        return flags

    def add_inventory(self, item: str, quantity: int) -> Dict[str, Any]:
        with self._lock:
            self.inventory[item] = self.inventory.get(item, 0) + quantity
            self._bump_version()
            self._record_event("restock", {"item": item, "quantity": quantity})
            return self.summary()

    def toggle_mode(self, mode: str) -> Dict[str, Any]:
        if mode not in {"normal", "maintenance", "slow"}:
            raise ValueError("invalid mode")
        with self._lock:
            self.mode = mode
            self._bump_version()
            self._record_event("mode_change", {"mode": mode})
            return self.summary()

    def purchase(self, item: str, quantity: int, expedite: bool) -> Dict[str, Any]:
        """Implements a slightly flawed optimistic path for expedite purchases.

        The expedite branch debits inventory before confirming availability.
        That bug is intentional: it is the kind of regression the adversarial
        engine should discover and then replay deterministically.
        """
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        with self._lock:
            available = self.inventory.get(item, 0)
            if self.mode == "maintenance":
                self._record_event("purchase_rejected", {"item": item, "reason": "maintenance"})
                raise RuntimeError("store in maintenance mode")

            if expedite:
                # Bug: optimistic fast path skips availability validation.
                self.inventory[item] = available - quantity
                status = "accepted_expedited_without_validation"
            else:
                if available < quantity:
                    self._record_event(
                        "purchase_conflict", {"item": item, "requested": quantity, "available": available}
                    )
                    raise RuntimeError("not enough inventory")
                self.inventory[item] = available - quantity
                status = "accepted"

            order = {
                "item": item,
                "quantity": quantity,
                "expedite": expedite,
                "status": status,
                "order_id": len(self.orders) + 1,
            }
            self.orders.append(order)
            self._bump_version()
            if self.inventory[item] < 0:
                self.outstanding_alerts.append(f"oversold:{item}")
            self._record_event("purchase", order)
            return self.summary()


STATE = AppState()


@app.before_request
def _start_timer() -> None:
    g.start_time = time.perf_counter()


@app.after_request
def _log_request(response):
    elapsed_ms = (time.perf_counter() - g.start_time) * 1000
    response.headers["X-Request-Latency-ms"] = f"{elapsed_ms:.2f}"
    logging.info(
        "method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.route("/reset", methods=["POST"])
def reset():
    summary = STATE.reset()
    return jsonify({"state": summary}), 200


@app.route("/inventory", methods=["GET", "POST"])
def inventory():
    if request.method == "GET":
        return jsonify({"state": STATE.summary()})

    payload = request.get_json(force=True, silent=True) or {}
    item = payload.get("item")
    quantity = payload.get("quantity")
    if item is None or quantity is None:
        return jsonify({"error": "item and quantity required"}), 400
    try:
        quantity_int = int(quantity)
    except ValueError:
        return jsonify({"error": "quantity must be integer"}), 400
    if quantity_int == 0:
        return jsonify({"error": "quantity must not be zero"}), 422

    summary = STATE.add_inventory(item, quantity_int)
    return jsonify({"state": summary}), 201


@app.route("/purchase", methods=["POST"])
def purchase():
    payload = request.get_json(force=True, silent=True) or {}
    item = payload.get("item")
    quantity = payload.get("quantity")
    expedite = bool(payload.get("expedite", False))
    if item is None or quantity is None:
        return jsonify({"error": "item and quantity required"}), 400
    try:
        quantity_int = int(quantity)
    except ValueError:
        return jsonify({"error": "quantity must be integer"}), 400

    slow_mode_delay = 0.25 if STATE.mode == "slow" else 0.0
    time.sleep(slow_mode_delay)

    try:
        summary = STATE.purchase(item, quantity_int, expedite)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "state": STATE.summary()}), 409

    return jsonify({"state": summary}), 201


@app.route("/mode", methods=["POST"])
def mode():
    payload = request.get_json(force=True, silent=True) or {}
    mode = payload.get("mode")
    if not mode:
        return jsonify({"error": "mode required"}), 400
    try:
        summary = STATE.toggle_mode(mode)
    except ValueError:
        return jsonify({"error": "mode must be one of normal|maintenance|slow"}), 400
    return jsonify({"state": summary}), 200


@app.route("/state", methods=["GET"])
def state():
    return jsonify({"state": STATE.summary()}), 200


@app.route("/diagnostics", methods=["GET"])
def diagnostics():
    # Provides a structured view the engine can consume for observability.
    return jsonify(
        {
            "state": STATE.summary(),
            "meta": {"app": "demo-inventory", "description": "stateful demo app"},
        }
    )


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    # Use the Flask dev server for simplicity; production setups should swap this
    # out for gunicorn/uwsgi. host=0.0.0.0 keeps it reachable from local tools.
    app.run(host="0.0.0.0", port=8000, debug=False)
