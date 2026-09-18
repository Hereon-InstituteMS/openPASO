"""JSON-backed session persistence for the WebUI.

A session captures: settings (model, mode, MCP toggles), the chat
history (events list), and the working-directory path the agent last
used. Sessions land in ``data/webui_sessions/<id>.json``.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import config


def _path(sid: str) -> Path:
    if not sid or "/" in sid or ".." in sid:
        raise ValueError(f"unsafe session id: {sid!r}")
    return config.SESSION_DIR / f"{sid}.json"


def new_session(model: str | None = None,
                mode: str | None = None,
                mcp_servers: list[str] | None = None) -> dict:
    # Distinguish "unspecified" (None → fall back to defaults) from
    # "explicit empty" (mcp_servers=[] → really disable all MCPs).
    if mcp_servers is None:
        mcp_servers = [sid for sid, s in config.MCP_SERVERS.items()
                       if s["default_on"]]
    return {
        "id": uuid.uuid4().hex[:12],
        "created_at": time.time(),
        "model": model or config.default_model(),
        "mode": mode or config.DEFAULT_MODE,
        "mcp_servers": list(mcp_servers),
        "events": [],
        "workdir": "",
        "tokens_in": 0,
        "tokens_out": 0,
    }


def save(session: dict) -> Path:
    path = _path(session["id"])
    path.write_text(json.dumps(session, indent=2, default=str))
    return path


def load(sid: str) -> dict:
    return json.loads(_path(sid).read_text())


def list_sessions() -> list[dict]:
    out = []
    for p in sorted(config.SESSION_DIR.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text())
            out.append({"id": d.get("id"), "created_at": d.get("created_at"),
                        "model": d.get("model"),
                        "n_events": len(d.get("events", []))})
        except Exception:
            continue
    return out


def delete(sid: str) -> bool:
    p = _path(sid)
    if p.exists():
        p.unlink()
        return True
    return False
