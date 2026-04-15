"""Uvicorn / asyncio tweaks per host OS."""

from __future__ import annotations

import sys


def win_preflight_tcp_bind(host: str, port: int) -> None:
    """
    Fail fast with a clear hint when bind will not work on Windows even though netstat shows
    the port as \"free\" (common with Hyper-V / Docker / WSL excluded port ranges).
    Only runs for common literal bind addresses used by this CLI.
    """
    if sys.platform != "win32":
        return

    literal_ok = host in ("127.0.0.1", "0.0.0.0", "::") or (":" in host)
    if not literal_ok:
        return

    import socket

    if host == "::":
        fam, addr = socket.AF_INET6, ("::", port)
    elif ":" in host:
        fam, addr = socket.AF_INET6, (host, port)
    else:
        fam, addr = socket.AF_INET, (host, port)

    sock = socket.socket(fam, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(addr)
    except OSError as exc:
        print(
            f"Cannot bind TCP {host!r} port {port}: {exc}\n",
            "On Windows this often happens when the port sits in an OS \"excluded\" range "
            "(Hyper-V, Docker Desktop, WSL2) even though no process appears to listen on it.\n"
            "Check reserved ranges:\n"
            "  netsh interface ipv4 show excludedportrange protocol=tcp\n"
            "Then pick a port outside those ranges, e.g.:\n"
            "  content-review server --port 28080\n",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc
    finally:
        sock.close()


def prepare_uvicorn_event_loop() -> None:
    """
    On Windows the default Proactor event loop can cause accepted TCP connections to be reset
    when used with some ASGI stacks. Uvicorn upstream recommends the selector policy here.
    Call once per process before the first asyncio loop used by uvicorn is created.
    """
    if sys.platform != "win32":
        return
    import asyncio

    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass
