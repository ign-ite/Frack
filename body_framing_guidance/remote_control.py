"""Minimal Flask-based remote control surface for phone-driven actions."""

import threading
from queue import Queue
from typing import Optional, Set

from config import WEB_PANEL_HOST, WEB_PANEL_PORT


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Framing Control Panel</title>
  <style>
    :root {
      --bg: #f3f4f6;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --accent: #0f766e;
      --danger: #b91c1c;
      --border: #d1d5db;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at 20% 10%, #e0f2fe 0%, var(--bg) 45%);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .panel {
      width: min(520px, 100%);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 14px 36px rgba(2, 6, 23, 0.12);
      padding: 20px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.35rem;
    }
    p {
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.4;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    button {
      appearance: none;
      width: 100%;
      border: none;
      border-radius: 12px;
      padding: 14px 10px;
      font-size: 0.97rem;
      font-weight: 600;
      color: #fff;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); opacity: 0.95; }
    .teal { background: var(--accent); }
    .slate { background: #334155; }
    .danger { background: var(--danger); }
    .amber { background: #b45309; }
    .wide { grid-column: 1 / -1; }
    .status {
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.9rem;
      min-height: 1.2em;
    }
  </style>
</head>
<body>
  <main class="panel">
    <h1>Framing Control Panel</h1>
    <p>Use this page from your phone on the same Wi-Fi network to control the live guidance app.</p>
    <div class="grid">
      <button class="teal" onclick="sendAction('start')">Start Assessment</button>
      <button class="slate" onclick="sendAction('stop')">Stop Assessment</button>
      <button class="amber" onclick="sendAction('mute')">Mute / Unmute</button>
      <button class="slate" onclick="sendAction('screenshot')">Save Screenshot</button>
      <button class="teal wide" onclick="sendAction('calibrate')">Calibrate Baseline</button>
    </div>
    <div class="status" id="status"></div>
  </main>

<script>
async function sendAction(action) {
  const status = document.getElementById('status');
  try {
    const response = await fetch('/api/action/' + action, { method: 'POST' });
    const payload = await response.json();
    status.textContent = payload.message || ('Sent: ' + action);
  } catch (error) {
    status.textContent = 'Failed to send action. Check app/server status.';
  }
}
</script>
</body>
</html>
"""


class RemoteControlServer:
    """Serve a small local web UI and publish actions into a queue."""

    _SUPPORTED_ACTIONS: Set[str] = {
        "start",
        "stop",
        "mute",
        "screenshot",
        "calibrate",
    }

    def __init__(
        self,
        command_queue: Queue,
        host: str = WEB_PANEL_HOST,
        port: int = WEB_PANEL_PORT,
    ) -> None:
        self._command_queue = command_queue
        self._host = host
        self._port = port
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._startup_error: Optional[str] = None

    @property
    def startup_error(self) -> Optional[str]:
        return self._startup_error

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Start the HTTP server in a background thread."""
        try:
            from flask import Flask, jsonify
            from werkzeug.serving import make_server
        except Exception as exc:
            self._startup_error = f"Flask server unavailable: {exc}"
            return False

        app = Flask(__name__)

        @app.get("/")
        def index():
            return HTML_TEMPLATE

        @app.post("/api/action/<action>")
        def action(action: str):
            if action not in self._SUPPORTED_ACTIONS:
                return jsonify({"ok": False, "message": f"Unknown action: {action}"}), 400

            self._command_queue.put(("remote", action))
            return jsonify({"ok": True, "message": f"Action sent: {action}"})

        try:
            self._server = make_server(self._host, self._port, app, threaded=True)
        except Exception as exc:
            self._startup_error = f"Failed to bind {self.url}: {exc}"
            return False

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._running = True
        return True

    def stop(self) -> None:
        """Stop the server cleanly."""
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
        self._running = False
