import logging
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List

from flask import Flask, jsonify, request, g, render_template_string


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

logging.info("__name__=%s", __name__)


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


UI_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Adversarial Regression Demo</title>
  <style>
    :root {
      --bg: #0b1224;
      --panel: #0f172a;
      --muted: #94a3b8;
      --bright: #38bdf8;
      --accent: #1d4ed8;
      --danger: #f43f5e;
      --success: #22c55e;
      --border: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.08), transparent 30%),
                  radial-gradient(circle at 80% 0%, rgba(79, 70, 229, 0.12), transparent 32%),
                  var(--bg);
      color: #e2e8f0;
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 22px 48px;
      position: relative;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
    }
    h1 {
      margin: 0 0 6px 0;
      font-size: 24px;
      letter-spacing: 0.4px;
    }
    .blurb {
      margin: 0;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.45;
    }
    .badge {
      background: rgba(56, 189, 248, 0.14);
      color: #e0f2fe;
      border: 1px solid rgba(56, 189, 248, 0.5);
      border-radius: 999px;
      padding: 6px 14px;
      font-weight: 600;
      font-size: 13px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 14px;
      margin-top: 18px;
    }
    .panel {
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px 16px;
      box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
    }
    .panel h2 {
      margin: 0 0 6px;
      font-size: 16px;
    }
    .panel p {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 14px;
    }
    .state {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      min-height: 250px;
      overflow: auto;
      font-family: "JetBrains Mono", "Fira Code", monospace;
      font-size: 13px;
      line-height: 1.4;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .stack {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 8px 0;
    }
    button {
      background: var(--accent);
      color: #e2e8f0;
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 10px 12px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      transition: transform 0.08s ease, box-shadow 0.16s ease;
      box-shadow: 0 10px 28px rgba(29, 78, 216, 0.28);
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      box-shadow: none;
    }
    button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(56, 189, 248, 0.24);
    }
    button.secondary { background: #0ea5e9; }
    button.danger { background: #be123c; box-shadow: 0 10px 22px rgba(190, 18, 60, 0.32); }
    button.ghost {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      box-shadow: none;
    }
    label {
      font-size: 13px;
      color: var(--muted);
      display: block;
      margin-bottom: 4px;
    }
    input, select {
      width: 100%;
      padding: 9px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0b1224;
      color: #e2e8f0;
      font-size: 14px;
    }
    form {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .log {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      min-height: 260px;
      max-height: 280px;
      overflow: auto;
      font-family: "JetBrains Mono", "Fira Code", monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .log-entry {
      margin: 0 0 6px 0;
      color: #cbd5e1;
    }
    .log-entry strong { color: #f8fafc; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid rgba(148, 163, 184, 0.3);
      background: rgba(148, 163, 184, 0.08);
    }
    .pill.alert { border-color: rgba(244, 63, 94, 0.5); color: #fecdd3; background: rgba(244, 63, 94, 0.08); }
    .cursor {
      position: fixed;
      width: 18px;
      height: 18px;
      border: 2px solid var(--bright);
      border-radius: 50%;
      box-shadow: 0 0 12px var(--bright);
      pointer-events: none;
      transition: transform 0.35s ease, opacity 0.2s ease;
      opacity: 0;
      z-index: 50;
      transform: translate(-60px, -60px);
    }
    @media (max-width: 960px) {
      .grid { grid-template-columns: 1fr; }
      header { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div>
        <h1>Adversarial Regression UI</h1>
        <p class="blurb">
          Watch a scripted mini-run of the adversarial tester exercising the demo app.
          Use the controls to poke the API yourself and observe state, alerts, and recent events.
        </p>
      </div>
      <div class="badge">Live demo</div>
    </header>

    <div class="grid">
      <section class="panel">
        <h2>Live state</h2>
        <p>Snapshot from <code>/state</code>. Updates after every action.</p>
        <pre class="state" id="state-view">{ "loading": true }</pre>
        <div class="stack" id="alerts"></div>
      </section>

      <section class="panel">
        <h2>Scripted demo (model)</h2>
        <p>Runs a short sequence that nudges the app into an anomalous state.</p>
        <div class="stack">
          <button id="run-demo">Run demo</button>
          <button class="ghost" id="btn-reset">Reset backend</button>
          <button class="ghost" id="btn-refresh">Refresh state</button>
        </div>
        <div class="stack">
          <button id="btn-mode-normal" class="ghost">Mode: normal</button>
          <button id="btn-mode-maintenance" class="ghost">Mode: maintenance</button>
          <button id="btn-mode-slow" class="ghost">Mode: slow</button>
        </div>
        <div class="log" id="demo-log"></div>
      </section>

      <section class="panel">
        <h2>Manual controls</h2>
        <p>Issue the same actions the engine explores.</p>
        <form id="restock-form">
          <label for="restock-item">Restock inventory</label>
          <div class="stack">
            <input id="restock-item" name="item" placeholder="item (widgets)" required />
            <input id="restock-qty" name="quantity" type="number" placeholder="qty (3)" required />
            <button type="submit" class="secondary">Restock</button>
          </div>
        </form>

        <form id="purchase-form">
          <label for="purchase-item">Purchase</label>
          <div class="stack">
            <input id="purchase-item" name="item" placeholder="item (gadgets)" required />
            <input id="purchase-qty" name="quantity" type="number" placeholder="qty (2)" required />
            <label><input type="checkbox" id="purchase-expedite" /> Expedite</label>
            <button type="submit">Purchase</button>
          </div>
        </form>

        <form id="restock-demo-form">
          <label>Quick actions</label>
          <div class="stack">
            <button type="button" id="btn-restock-widgets" class="secondary">+3 widgets</button>
            <button type="button" id="btn-restock-doodads" class="secondary">+2 doodads</button>
            <button type="button" id="btn-expedite-gadgets" class="danger">Expedite 4 gadgets</button>
          </div>
        </form>
      </section>
    </div>
  </div>

  <div class="cursor" id="demo-cursor"></div>

  <script>
    const stateView = document.getElementById("state-view");
    const alertsView = document.getElementById("alerts");
    const logBox = document.getElementById("demo-log");
    const cursor = document.getElementById("demo-cursor");
    const runButton = document.getElementById("run-demo");

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    const log = (message, tone = "info") => {
      const entry = document.createElement("div");
      entry.className = "log-entry";
      const prefix = tone === "error" ? "✖" : tone === "ok" ? "✔" : "•";
      entry.innerHTML = "<strong>" + prefix + "</strong> " + message;
      logBox.prepend(entry);
      while (logBox.children.length > 60) {
        logBox.removeChild(logBox.lastChild);
      }
    };

    const moveCursor = (el) => {
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const x = rect.left + rect.width / 2 + window.scrollX;
      const y = rect.top + rect.height / 2 + window.scrollY;
      cursor.style.opacity = 1;
      cursor.style.transform = "translate(" + x + "px," + y + "px)";
    };

    const api = async (path, options = {}, tone = "info") => {
      try {
        const res = await fetch(path, {
          headers: { "Content-Type": "application/json" },
          ...options,
        });
        let data = null;
        try {
          data = await res.json();
        } catch (jsonErr) {
          data = null;
        }
        if (!res.ok) {
          const err = data && data.error ? data.error : res.statusText;
          log(path + " → " + err, "error");
          return data;
        }
        log(path + " → ok", tone);
        return data;
      } catch (err) {
        log(path + " failed: " + err, "error");
        return null;
      }
    };

    const renderState = (state) => {
      if (!state) return;
      stateView.textContent = JSON.stringify(state.state, null, 2);
      alertsView.innerHTML = "";
      const alerts = state.state.alerts || [];
      if (!alerts.length) return;
      alerts.forEach((a) => {
        const pill = document.createElement("div");
        pill.className = "pill alert";
        pill.textContent = "alert: " + a;
        alertsView.appendChild(pill);
      });
    };

    const refreshState = async () => {
      const snapshot = await api("/state");
      renderState(snapshot);
    };

    document.getElementById("btn-reset").addEventListener("click", async () => {
      moveCursor(document.getElementById("btn-reset"));
      const state = await api("/reset", { method: "POST" }, "ok");
      renderState(state);
    });

    document.getElementById("btn-refresh").addEventListener("click", refreshState);
    document.getElementById("btn-mode-normal").addEventListener("click", async () => {
      moveCursor(document.getElementById("btn-mode-normal"));
      const state = await api("/mode", { method: "POST", body: JSON.stringify({ mode: "normal" }) }, "ok");
      renderState(state);
    });
    document.getElementById("btn-mode-maintenance").addEventListener("click", async () => {
      moveCursor(document.getElementById("btn-mode-maintenance"));
      const state = await api("/mode", { method: "POST", body: JSON.stringify({ mode: "maintenance" }) }, "ok");
      renderState(state);
    });
    document.getElementById("btn-mode-slow").addEventListener("click", async () => {
      moveCursor(document.getElementById("btn-mode-slow"));
      const state = await api("/mode", { method: "POST", body: JSON.stringify({ mode: "slow" }) }, "ok");
      renderState(state);
    });

    document.getElementById("restock-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const item = document.getElementById("restock-item").value.trim();
      const qty = parseInt(document.getElementById("restock-qty").value, 10);
      const state = await api("/inventory", { method: "POST", body: JSON.stringify({ item, quantity: qty }) }, "ok");
      renderState(state);
    });

    document.getElementById("purchase-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const item = document.getElementById("purchase-item").value.trim();
      const qty = parseInt(document.getElementById("purchase-qty").value, 10);
      const expedite = document.getElementById("purchase-expedite").checked;
      const state = await api(
        "/purchase",
        { method: "POST", body: JSON.stringify({ item, quantity: qty, expedite }) },
        expedite ? "error" : "ok",
      );
      renderState(state);
    });

    document.getElementById("btn-restock-widgets").addEventListener("click", async () => {
      const state = await api("/inventory", { method: "POST", body: JSON.stringify({ item: "widgets", quantity: 3 }) }, "ok");
      renderState(state);
    });
    document.getElementById("btn-restock-doodads").addEventListener("click", async () => {
      const state = await api("/inventory", { method: "POST", body: JSON.stringify({ item: "doodads", quantity: 2 }) }, "ok");
      renderState(state);
    });
    document.getElementById("btn-expedite-gadgets").addEventListener("click", async () => {
      moveCursor(document.getElementById("btn-expedite-gadgets"));
      const state = await api(
        "/purchase",
        { method: "POST", body: JSON.stringify({ item: "gadgets", quantity: 4, expedite: true }) },
        "error",
      );
      renderState(state);
    });

    const scriptedActions = [
      {
        label: "Reset the app",
        el: () => document.getElementById("btn-reset"),
        fn: () => api("/reset", { method: "POST" }, "ok"),
      },
      {
        label: "Switch to slow mode",
        el: () => document.getElementById("btn-mode-slow"),
        fn: () => api("/mode", { method: "POST", body: JSON.stringify({ mode: "slow" }) }, "ok"),
      },
      {
        label: "Expedite 4 gadgets (intentional oversell)",
        el: () => document.getElementById("btn-expedite-gadgets"),
        fn: () => api("/purchase", { method: "POST", body: JSON.stringify({ item: "gadgets", quantity: 4, expedite: true }) }, "error"),
      },
      {
        label: "Toggle maintenance mode",
        el: () => document.getElementById("btn-mode-maintenance"),
        fn: () => api("/mode", { method: "POST", body: JSON.stringify({ mode: "maintenance" }) }, "ok"),
      },
      {
        label: "Attempt purchase in maintenance (should reject)",
        el: () => document.getElementById("purchase-form"),
        fn: () => api("/purchase", { method: "POST", body: JSON.stringify({ item: "widgets", quantity: 1, expedite: false }) }, "error"),
      },
      {
        label: "Restock doodads",
        el: () => document.getElementById("btn-restock-doodads"),
        fn: () => api("/inventory", { method: "POST", body: JSON.stringify({ item: "doodads", quantity: 2 }) }, "ok"),
      },
      {
        label: "Back to normal mode",
        el: () => document.getElementById("btn-mode-normal"),
        fn: () => api("/mode", { method: "POST", body: JSON.stringify({ mode: "normal" }) }, "ok"),
      },
    ];

    const runDemo = async () => {
      runButton.disabled = true;
      log("Starting scripted demo…");
      for (const step of scriptedActions) {
        const el = step.el();
        moveCursor(el);
        log(step.label);
        const state = await step.fn();
        renderState(state);
        await sleep(800);
      }
      log("Demo finished", "ok");
      runButton.disabled = false;
    };

    runButton.addEventListener("click", runDemo);
    refreshState();
    moveCursor(runButton);
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(UI_TEMPLATE)


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    # Use the Flask dev server for simplicity; production setups should swap this
    # out for gunicorn/uwsgi. host=0.0.0.0 keeps it reachable from local tools.
    logging.info("starting demo_app Flask dev server on http://127.0.0.1:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
