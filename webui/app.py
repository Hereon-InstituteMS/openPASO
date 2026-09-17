"""FastAPI app for the openPASO WebUI.

Start with::

    .venv-lg/bin/uvicorn webui.app:app --reload --port 8080

Endpoints:

* ``GET  /`` — single-page UI (HTML)
* ``GET  /api/models`` / ``/api/mcp_servers`` / ``/api/modes``
* ``GET  /api/files?rel=…`` — sandbox directory listing
* ``GET  /api/file?rel=…`` — file text content
* ``GET  /api/viz?rel=…`` — visualization payload
* ``GET  /sandbox-file/{path:path}`` — raw file bytes for vtk.js etc.
* ``GET  /api/sessions`` / ``POST /api/sessions`` / ``DELETE``
* ``GET  /api/extract_params?rel=…`` — heuristic slider extraction
* ``WS   /ws/{session_id}`` — streamed run channel

The WebSocket protocol is symmetric JSON:
``{"type": "...", ...}`` either direction. See :mod:`webui.runner` for
outbound event types and :func:`_handle_inbound` for the inbound set.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, files, sessions, viz
from .runner import (ApprovalGate, _session_workdir,
                     open_agent_for_session, stream_turn)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("openpaso.webui")

app = FastAPI(title="openPASO WebUI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()


# ───────────────────────────────────────────────────────────────────
# Config endpoints
# ───────────────────────────────────────────────────────────────────
@app.get("/api/models")
async def get_models():
    out = []
    for k, m in config.MODELS.items():
        # The mock is a fake server that answers one canned turn and runs no
        # solver. It stays for the test suite and is not offered to a person:
        # a fabricated run used to be indistinguishable from real work.
        if k == "mock":
            continue
        out.append({"id": k, "label": m["label"], "port": m["port"]})
    from . import claude_code
    if claude_code.available():
        out.append({"id": config.CLAUDE_CODE_ID,
                    "label": "Claude Code (uses your subscription)", "port": None})
    have_key = bool(config.openrouter_key())
    for k, label in config.OPENROUTER_MODELS.items():
        out.append({"id": k, "label": label, "port": None,
                    "needs_key": not have_key})
    return {"models": out, "default": config.default_model(),
            "openrouter_key": have_key}


@app.get("/api/sessions/{sid}/manifest")
async def get_manifest(sid: str):
    """Everything needed to check or reproduce one run, in one file.

    A result that cannot be traced back to what produced it is not a result. The
    interface shows a summary; this is the record behind it, with the full
    untruncated event log and a hash for every artefact the run wrote."""
    import hashlib
    try:
        state = sessions.load(sid)
    except Exception:
        return JSONResponse({"error": f"no such run: {sid}"}, status_code=404)

    events = state.get("events") or []
    # A run that raised is a failed run, whatever came after it. The server
    # emits "done" immediately following "error", so taking the last terminal
    # event reported a crash as completed. Older sessions carry no outcome
    # field at all, which is why the presence of an error decides it.
    outcome = "unknown"
    for e in events:
        if e.get("type") == "error":
            outcome = e.get("outcome") or "failed"
        elif e.get("type") == "done":
            stated = e.get("outcome")
            if stated:
                outcome = stated
            elif outcome == "unknown":
                outcome = "completed"

    work = config.SANDBOX_ROOT / f"webui_{sid}"
    artefacts = []
    if work.is_dir():
        for f in sorted(work.rglob("*")):
            if not f.is_file():
                continue
            data = f.read_bytes()
            artefacts.append({
                "path": str(f.relative_to(work)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "modified": time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.localtime(f.stat().st_mtime)),
            })

    try:
        import sys
        src = str(config.REPO / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from core import registry
        registry.load_all_backends()
        solvers = {r["display_name"]: r["version"] for r in registry.list_backends()
                   if r["status"] == "available"}
    except Exception:
        solvers = {}

    prompt = next((e.get("text") for e in events if e.get("type") == "user_msg"), None)
    return {
        "run": sid,
        "outcome": outcome,
        "prompt": prompt,
        "model": state.get("model"),
        "mode": state.get("mode"),
        "mcp_servers": state.get("mcp_servers"),
        "tokens": {"in": state.get("tokens_in"), "out": state.get("tokens_out")},
        "solver_versions": solvers,
        "working_directory": str(work),
        "artefacts": artefacts,
        "events": events,
        "openpaso_commit": _commit(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _commit() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(config.REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


@app.get("/api/solvers")
async def get_solvers():
    """What is actually installed, from the same registry `discover` reads.

    The first screen used to assert nine solvers from a hardcoded array. On a
    machine with none installed it still said nine. If this check cannot run,
    it says so rather than guessing."""
    try:
        import sys
        src = str(config.REPO / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from core import registry
        registry.load_all_backends()
        rows = registry.list_backends()
    except Exception as exc:
        log.warning("solver check failed: %s", exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "solvers": []}
    return {"ok": True, "solvers": [
        {"name": r["display_name"], "status": r["status"],
         "version": r["version"], "physics": r["physics_count"]}
        for r in rows
    ]}


@app.get("/api/mcp_servers")
async def get_mcp():
    return {"servers": [
        {"id": k, "label": v["label"], "default_on": v["default_on"]}
        for k, v in config.MCP_SERVERS.items()
    ]}


@app.get("/api/modes")
async def get_modes():
    return {"modes": list(config.MODES), "default": config.DEFAULT_MODE}


# ───────────────────────────────────────────────────────────────────
# Files & viz
# ───────────────────────────────────────────────────────────────────
@app.get("/api/files")
async def api_files(rel: str = ""):
    try:
        return files.list_dir(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/api/file")
async def api_file(rel: str):
    try:
        return files.read_text(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.post("/api/file")
async def api_file_save(body: dict):
    """Write ``content`` to ``rel`` inside the sandbox. Used by the
    in-browser editor for input files / scripts."""
    rel = body.get("rel")
    content = body.get("content")
    if not rel or content is None:
        raise HTTPException(400, "rel and content required")
    try:
        p = files._safe(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if p.is_dir():
        raise HTTPException(400, "path is a directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"ok": True, "path": str(p), "bytes": len(content)}


@app.get("/api/viz")
async def api_viz(rel: str):
    try:
        return viz.visualize(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/api/extract_params")
async def api_extract_params(rel: str):
    info = files.read_text(rel)
    if "text" not in info:
        return {"params": []}
    return {"params": viz.extract_params(info["text"])}


@app.get("/sandbox-file/{rel:path}")
async def sandbox_file(rel: str):
    try:
        p = files._safe(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not p.is_file():
        raise HTTPException(404, "not a file")
    return FileResponse(p)


# ───────────────────────────────────────────────────────────────────
# Sessions
# ───────────────────────────────────────────────────────────────────
@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": sessions.list_sessions()}


@app.post("/api/sessions")
async def new_session(body: dict | None = None):
    body = body or {}
    s = sessions.new_session(
        model=body.get("model"),
        mode=body.get("mode"),
        mcp_servers=body.get("mcp_servers"))
    sessions.save(s)
    return s


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    try:
        return sessions.load(sid)
    except FileNotFoundError:
        raise HTTPException(404, "no such session")


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    return {"deleted": sessions.delete(sid)}


# ───────────────────────────────────────────────────────────────────
# WebSocket — interactive runs
# ───────────────────────────────────────────────────────────────────
class WSSession:
    def __init__(self, ws: WebSocket, sid: str):
        self.ws = ws
        self.sid = sid
        self.state = sessions.load(sid)
        self.gate = ApprovalGate()
        self.agent = None
        self.agent_context = None
        self.workdir = _session_workdir(sid)
        # The active agent turn runs as a background task so we can keep
        # processing approve/reject messages from the WS while the gated
        # tool call is waiting. Without this, plan-mode deadlocks.
        self.turn_task: asyncio.Task | None = None

    def mode(self) -> str:
        return self.state.get("mode", config.DEFAULT_MODE)

    async def emit(self, event: dict):
        # Strip any state-snapshot fields before persisting so we never
        # build a self-referencing tree (state.events ⊃ event ⊃ session
        # ⊃ state). The wire payload still includes the snapshot.
        wire = event
        # A status is transient chrome: it drives the header line and is gone.
        # Persisting it meant every reload replayed 'connected' and each
        # 'thinking...' as a permanent bubble in the scrollback.
        if event.get("type") != "status":
            persist = {k: v for k, v in event.items() if k != "session"}
            self.state["events"].append(persist)
            # The record used to be written only when the turn ended, so a
            # crashed process lost every event and the manifest for that run
            # was empty. Checkpointing costs a small write and means the log
            # survives whatever happens to the run.
            if len(self.state["events"]) % 10 == 0:
                try:
                    sessions.save(self.state)
                except Exception:
                    log.warning("could not checkpoint session %s", self.state["id"])
        if event.get("type") == "token_count":
            self.state["tokens_in"] = (self.state.get("tokens_in", 0)
                                       + (event.get("input") or 0))
            self.state["tokens_out"] = (self.state.get("tokens_out", 0)
                                        + (event.get("output") or 0))
        try:
            await self.ws.send_text(json.dumps(wire, default=str))
        except Exception:
            pass

    async def ensure_agent(self):
        if self.agent is None:
            self.agent_context = open_agent_for_session(
                model=self.state.get("model", config.DEFAULT_MODEL),
                mcp_on="openpaso" in self.state.get("mcp_servers", []),
                workdir=self.workdir,
                emitter=self.emit,
                get_mode=self.mode,
                gate=self.gate,
            )
            self.agent = await self.agent_context.__aenter__()
        return self.agent

    async def close_agent(self):
        turn, self.turn_task = self.turn_task, None
        if (turn is not None and not turn.done()
                and turn is not asyncio.current_task()):
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn
        context, self.agent_context = self.agent_context, None
        self.agent = None
        if context is not None:
            await context.__aexit__(None, None, None)


@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    try:
        ses = WSSession(ws, sid)
    except FileNotFoundError:
        await ws.send_text(json.dumps({"type": "error",
                                       "message": f"no such session: {sid}"}))
        await ws.close()
        return
    try:
        # Send a session snapshot WITHOUT the events array so the wire
        # payload stays bounded; the frontend already has the
        # historical events via GET /api/sessions/{sid}.
        snap = {k: v for k, v in ses.state.items() if k != "events"}
        await ses.emit({"type": "status", "message": "connected",
                        "session": snap})
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ses.emit({"type": "error", "message": "bad json"})
                continue
            await _handle_inbound(ses, msg)
    except WebSocketDisconnect:
        log.info("client disconnected: %s", sid)
    except Exception as e:
        log.exception("ws error")
        try:
            await ws.send_text(json.dumps({"type": "error",
                                           "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ses.close_agent()
        except Exception:
            log.exception("could not close agent session")
        try:
            sessions.save(ses.state)
        except Exception:
            pass


async def _handle_inbound(ses: WSSession, msg: dict):
    t = msg.get("type")
    if t == "prompt":
        text = msg.get("text", "").strip()
        if not text:
            return
        # Refuse to start a second turn while one is in flight so the
        # event stream stays interpretable.
        if ses.turn_task is not None and not ses.turn_task.done():
            await ses.emit({"type": "error",
                            "message": "previous turn is still running"})
            return
        await ses.emit({"type": "user_msg", "text": text})

        use_claude_code = ses.state.get("model") == config.CLAUDE_CODE_ID
        if not use_claude_code:
            await ses.ensure_agent()

        async def _run():
            """One turn. It ends in exactly one terminal state, and the user is
            told which.

            This used to be `except Exception: pass` around the whole body, with
            the `done` emit inside the try. Any failure was therefore swallowed
            without a log, no `done` was sent, and the browser pulsed "working"
            forever on a dead process. A silent failure that looks like a slow
            success is the worst of the three possible outcomes."""
            outcome = "completed"
            try:
                if use_claude_code:
                    from . import claude_code
                    from .runner import _session_workdir
                    await ses.emit({"type": "status", "message": "thinking…"})
                    await claude_code.stream_turn(
                        text,
                        workdir=_session_workdir(ses.state["id"]),
                        servers=list(ses.state.get("mcp_servers") or []),
                        model=None, mode=ses.mode, emit=ses.emit)
                else:
                    await stream_turn(agent=ses.agent, user_text=text,
                                      emitter=ses.emit, emit_done=False)
            except asyncio.CancelledError:
                outcome = "interrupted"
                await ses.emit({"type": "error", "outcome": outcome,
                                "message": "Run stopped."})
                raise
            except Exception as exc:
                outcome = "failed"
                log.exception("turn failed for session %s", ses.state["id"])
                await ses.emit({"type": "error", "outcome": outcome,
                                "message": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()[-4000:]})
            finally:
                if outcome != "interrupted":
                    await ses.emit({"type": "done", "outcome": outcome})
                sessions.save(ses.state)

        ses.turn_task = asyncio.create_task(_run())
    elif t == "stop":
        # A run can take a quarter of an hour. Someone who started the wrong
        # thing must be able to change their mind.
        task = ses.turn_task
        if task is not None and not task.done():
            task.cancel()
        else:
            await ses.emit({"type": "error", "outcome": "failed",
                            "message": "There is no run to stop."})

    elif t == "approve":
        ses.gate.resolve(msg["call_id"], True)
    elif t == "reject":
        ses.gate.resolve(msg["call_id"], False, msg.get("reason", ""))
    elif t == "set_mode":
        ses.state["mode"] = msg.get("mode", ses.state["mode"])
        sessions.save(ses.state)
        await ses.emit({"type": "status", "message": "",
                        "session": {k: v for k, v in ses.state.items()
                                    if k != "events"}})
    elif t == "set_model":
        ses.state["model"] = msg["model"]
        await ses.close_agent()
        sessions.save(ses.state)
        await ses.emit({"type": "status", "message": "",
                        "session": {k: v for k, v in ses.state.items()
                                    if k != "events"}})
    elif t == "set_mcp":
        ses.state["mcp_servers"] = msg.get("servers", [])
        await ses.close_agent()
        sessions.save(ses.state)
        await ses.emit({"type": "status",
                        "message": "MCP servers updated; agent will "
                        "rebuild on next prompt"})
    elif t == "restart":
        ses.state["events"] = []
        ses.state["tokens_in"] = 0
        ses.state["tokens_out"] = 0
        await ses.close_agent()
        sessions.save(ses.state)
        await ses.emit({"type": "status", "message": "session restarted"})
    else:
        await ses.emit({"type": "error",
                        "message": f"unknown inbound type: {t}"})
