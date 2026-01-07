import atexit
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests


def _probe(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/state", timeout=2)
        return resp.ok
    except requests.RequestException:
        return False


def ensure_backend_available(base_url: str, autostart: bool = True, app_script: str = "demo_app.py") -> Optional[subprocess.Popen]:
    """Check connectivity and optionally start the demo app for convenience.

    This keeps the happy path simple when users forget to launch the app; the
    subprocess is terminated on exit to avoid orphan processes.
    """
    if _probe(base_url):
        return None

    if not autostart:
        raise RuntimeError(f"Backend not reachable at {base_url}. Start it via `python {app_script}`.")

    script_path = Path(app_script)
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot autostart: {script_path} not found.")

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Make sure we clean up even on Ctrl+C.
    atexit.register(proc.terminate)

    deadline = time.time() + 8
    while time.time() < deadline:
        if _probe(base_url):
            return proc
        time.sleep(0.3)

    proc.terminate()
    raise RuntimeError(f"Failed to reach backend at {base_url} after autostarting {app_script}.")
