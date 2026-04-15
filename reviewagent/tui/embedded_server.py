"""
Start FastAPI (uvicorn) in a background thread so one command can run server and TUI together.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import httpx


def _client_base(host: str, port: int) -> str:
    h = "127.0.0.1" if host in ("0.0.0.0", "::", "::0") else host
    return f"http://{h}:{port}"


def ensure_local_server(
    host: str = "127.0.0.1",
    port: int = 18080,
    *,
    wait_seconds: float = 15.0,
) -> str:
    """
    Reuse an existing healthy service on the port, or start uvicorn in a thread and poll /health.
    Returns the API base URL the TUI should use.
    """
    base = _client_base(host, port)

    try:
        r = httpx.get(f"{base}/health", timeout=1.0)
        if r.status_code == 200:
            return base
    except httpx.HTTPError:
        pass

    from reviewagent.uvicorn_support import prepare_uvicorn_event_loop, win_preflight_tcp_bind

    prepare_uvicorn_event_loop()
    win_preflight_tcp_bind(host, port)

    import uvicorn

    from reviewagent.api.server import create_app

    app = create_app()
    _uv_log = (os.environ.get("UVICORN_LOG_LEVEL") or "warning").lower()
    config = uvicorn.Config(app, host=host, port=port, log_level=_uv_log)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=0.4)
            if r.status_code == 200:
                return base
        except httpx.HTTPError:
            time.sleep(0.05)

    raise RuntimeError(
        f"Embedded API did not become ready within {wait_seconds}s: {base}\n"
        "Try another port or start manually: content-review server"
    )


__all__ = ["ensure_local_server", "_client_base"]
