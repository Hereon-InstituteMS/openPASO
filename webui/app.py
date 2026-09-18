"""FastAPI app for the openPASO WebUI.

Start with::

    .venv-lg/bin/uvicorn webui.app:app --reload --port 8080

Endpoints:

* ``GET  /`` — single-page UI (HTML)
* ``GET  /api/models`` / ``/api/mcp_servers`` / ``/api/modes``
* ``GET  /api/file?rel=…`` — file text content (read only)
* ``GET  /api/viz?rel=…`` — visualization payload
* ``GET  /sandbox-file/{path:path}`` — raw file bytes for vtk.js etc.
* ``GET  /api/sessions`` / ``POST /api/sessions`` / ``DELETE``
* ``WS   /ws/{session_id}`` — streamed run channel

The WebSocket protocol is symmetric JSON:
``{"type": "...", ...}`` either direction. See :mod:`webui.runner` for
outbound event types and :func:`_handle_inbound` for the inbound set.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import logging
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import catalog, config, files, runs, sessions, viz
from .outcome import fold as outcome_fold
from .privacy import scrub, scrub_text
from .runner import _session_workdir

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("openpaso.webui")

app = FastAPI(title="openPASO WebUI", version="0.1.0")
# The interface is served by this app, so cross-origin access is only ever
# wanted from the Vite dev server. A wildcard let any page a researcher had
# open read this sandbox and, while POST /api/file existed, write to it.
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173",
                                  "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

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
    """Models grouped by where they run, each with whether it works right now,
    what it costs and where the data goes. The mock is never offered."""
    out = await catalog.models()
    # what this person's own runs with each model have cost: a measured range,
    # not an estimate, and only where there is one
    costs: dict[str, list[float]] = {}
    for p in config.SESSION_DIR.glob("*.json"):
        row = _summary(p)
        if row and row.get("prompt") and row.get("cost_usd") is not None and not row.get("running"):
            costs.setdefault(row["model"], []).append(float(row["cost_usd"]))
    for g in out["groups"]:
        for m in g["models"]:
            c = costs.get(m["id"])
            if c:
                m.update(past_runs=len(c), past_cost_low=min(c), past_cost_high=max(c))
    return out


@app.get("/api/config")
async def get_config():
    return {"modes": [{"id": m, **config.MODE_INFO[m]} for m in config.MODES],
            "default_mode": config.DEFAULT_MODE,
            "docs_url": config.DOCS_URL,
            "max_running": config.MAX_RUNNING}


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
    outcome = outcome_fold(events)

    work = config.SANDBOX_ROOT / f"webui_{sid}"
    artefacts = []
    if work.is_dir():
        root = work.resolve()
        for f in sorted(work.rglob("*")):
            # a link the run made can point anywhere; hashing what it points at
            # would put a file from outside the run into its record
            if f.is_symlink() or not f.is_file():
                continue
            if not f.resolve().is_relative_to(root):
                continue
            digest, size = hashlib.sha256(), 0
            with f.open("rb") as fh:                 # a run may write gigabytes
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            artefacts.append({
                "path": str(f.relative_to(work)),
                "bytes": size,
                "sha256": digest.hexdigest(),
                "modified": time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.localtime(f.stat().st_mtime)),
            })

    check = await catalog.solvers()
    solvers = {r["name"]: r["version"] for r in check.get("solvers", [])
               if r["status"] == "available"}

    prompt = next((e.get("text") for e in events if e.get("type") == "user_msg"), None)
    return scrub({
        "run": sid,
        "outcome": outcome,
        "prompt": prompt,
        "model": state.get("model"),
        "mode": state.get("mode"),
        "mcp_servers": state.get("mcp_servers"),
        "tokens": {"in": state.get("tokens_in"), "out": state.get("tokens_out")},
        "solver_versions": solvers,
        "working_directory": f"eval_interactive/webui_{sid}",
        "artefacts": artefacts,
        "events": events,
        "openpaso_commit": _commit(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def _commit() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(config.REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


@app.get("/api/solvers")
async def get_solvers(refresh: bool = False):
    """What is installed, asked of openPASO's own Python in a subprocess.

    It used to run in the web server's Python, which is not the one openPASO
    uses (it lacked scikit-fem, so the page said 8 of 9 while runs had all nine),
    and inside the event loop, freezing every other request while it ran."""
    return await catalog.solvers(refresh=refresh)


@app.get("/api/mcp_servers")
async def get_mcp():
    return {"servers": [
        {"id": k, "label": v["label"], "default_on": v["default_on"]}
        for k, v in config.MCP_SERVERS.items()
    ]}


# ───────────────────────────────────────────────────────────────────
# Files & viz
# ───────────────────────────────────────────────────────────────────
_RUN_REL = re.compile(r"^webui_[0-9a-f]{6,32}/")


def _run_path(rel: str) -> str:
    """Files are served from run folders only. The sandbox also holds other
    working directories (evaluation campaigns, scratch) that are nobody's run."""
    rel = (rel or "").lstrip("/")
    if not _RUN_REL.match(rel) or ".." in Path(rel).parts:
        raise HTTPException(403, "Only files inside a run's folder can be opened here.")
    return rel


@app.get("/api/file")
async def api_file(rel: str):
    try:
        return scrub(files.read_text(_run_path(rel)))
    except PermissionError:
        raise HTTPException(403, "That path is outside the run folders.")


# There is no POST /api/file: it once wrote anywhere under the sandbox with no
# authentication. Files reach a run only through its upload endpoint.


@app.get("/api/viz")
async def api_viz(rel: str):
    try:
        return scrub(viz.visualize(_run_path(rel)))
    except PermissionError:
        raise HTTPException(403, "That path is outside the run folders.")


# Text a run wrote can hold the home directory it worked in, and this is the
# path a Download link uses. Everything else the browser is shown is scrubbed;
# without this the one route that hands over whole files was the exception.
_SCRUB_SUFFIXES = {".txt", ".log", ".out", ".err", ".md", ".json", ".yaml", ".yml",
                   ".xml", ".csv", ".tsv", ".py", ".sh", ".dat", ".inp", ".i",
                   ".feb", ".xdmf", ".pvd", ".geo", ".cfg", ".ini", ".toml"}
_SCRUB_MAX = 32 * 1024 * 1024


@app.get("/sandbox-file/{rel:path}")
async def sandbox_file(rel: str):
    try:
        p = files._safe(_run_path(rel))
    except PermissionError:
        raise HTTPException(403, "That path is outside the run folders.")
    if not p.is_file():
        raise HTTPException(404, "not a file")
    if p.suffix.lower() in _SCRUB_SUFFIXES and p.stat().st_size <= _SCRUB_MAX:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return FileResponse(p)
        return Response(scrub_text(text), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'inline; filename="{p.name}"'})
    return FileResponse(p)


# ───────────────────────────────────────────────────────────────────
# Sessions
# ───────────────────────────────────────────────────────────────────
_SUMMARY_CACHE: dict[str, tuple[float, dict]] = {}


def _summary(path: Path) -> dict | None:
    sid = path.stem
    live = runs.live(sid)
    if live is not None:
        st, running, outcome = live.state, live.running, live.outcome()
        waiting = live.waiting
    else:
        mtime = path.stat().st_mtime
        hit = _SUMMARY_CACHE.get(sid)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            st = json.loads(path.read_text())
        except Exception:
            return None
        running, waiting, outcome = False, False, outcome_fold(st.get("events") or [])
        if outcome == "running":
            outcome = runs.UNFINISHED
    events = st.get("events") or []
    prompt = next((e.get("text") for e in events if e.get("type") == "user_msg"), None)
    label, kind = catalog.model_label(st.get("model", ""))
    out = {
        "id": st.get("id"), "created_at": st.get("created_at"),
        "updated_at": path.stat().st_mtime if path.exists() else st.get("created_at"),
        "model": st.get("model"), "model_label": label, "model_kind": kind,
        "model_detail": st.get("model_detail") or st.get("claude_model"),
        "mode": st.get("mode"), "prompt": prompt, "outcome": outcome,
        "running": running,
        "waiting": waiting,
        "steps": sum(1 for e in events if e.get("type") == "tool_call_pending"),
        "cost_usd": st.get("cost_usd"),
    }
    if live is None:
        _SUMMARY_CACHE[sid] = (path.stat().st_mtime, out)
    return out


@app.get("/api/sessions")
async def list_sessions(all: bool = False):
    """Runs a person has actually started, newest first. Records with no prompt
    (a page that was opened and left) and test-model runs are not work, so they
    are left out unless asked for."""
    rows = []
    paths = sorted(config.SESSION_DIR.glob("*.json"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    for path in paths:
        row = await asyncio.to_thread(_summary, path)
        if not row:
            continue
        if not all and (not row["prompt"] or row["model_kind"] == "test"):
            continue
        rows.append(row)
    return {"sessions": rows, "running": runs.running_count()}


@app.post("/api/sessions")
async def new_session(body: dict | None = None):
    body = body or {}
    model = body.get("model")
    mode = body.get("mode") or config.DEFAULT_MODE
    known = (set(config.OPENROUTER_MODELS) | {config.CLAUDE_CODE_ID}
             | {k for k in config.MODELS if k != "mock" or body.get("test")})
    if model not in known:
        raise HTTPException(400, f"unknown model: {model}")
    if mode not in config.MODES:
        raise HTTPException(400, f"unknown mode: {mode}")
    if runs.running_count() >= config.MAX_RUNNING:
        raise HTTPException(409, f"{config.MAX_RUNNING} runs are already working on this machine. "
                                 "Your prompt was not sent. Wait for one to finish or stop one, then press Run again.")
    if model == config.CLAUDE_CODE_ID and mode == "plan":
        raise HTTPException(400, "Claude Code cannot stop to ask before each step. "
                                 "Choose 'Run without asking', or a different model.")
    s = sessions.new_session(model=model, mode=mode,
                             mcp_servers=body.get("mcp_servers"))
    sessions.save(s)
    return s


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    try:
        run = runs.get(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "no such run")
    return scrub({**run.snapshot(), "events": run.state["events"]})


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    live = runs.live(sid)
    if live is not None and live.running:
        raise HTTPException(409, "This run is still working. Stop it first.")
    if live is not None:
        await live.close_agent()
        runs.RUNS.pop(sid, None)
    import shutil
    folder = config.SANDBOX_ROOT / f"webui_{sid}"
    try:
        deleted = sessions.delete(sid)
    except ValueError:
        raise HTTPException(400, "bad run id")
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    _SUMMARY_CACHE.pop(sid, None)
    return {"deleted": deleted}


@app.get("/api/sessions/{sid}/files")
async def run_files(sid: str, sub: str = ""):
    """This run's own folder only. The file panel used to open at the root of
    the sandbox and list every run's directory side by side."""
    try:
        runs.get(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "no such run")
    base = f"webui_{sid}/work"
    rel = f"{base}/{sub}".rstrip("/") if sub else base
    try:
        listing = files.list_dir(rel)
    except PermissionError:
        raise HTTPException(403, "That path is outside this run's folder.")
    if not (listing.get("rel") or "").startswith(base):
        raise HTTPException(403, "outside this run")
    for e in listing.get("entries", []):
        e.pop("abs_path", None)
        e["sub"] = e["rel_path"][len(base) + 1:]
    listing.pop("path", None)
    listing["sub"] = sub
    return listing


@app.post("/api/sessions/{sid}/upload")
async def upload(sid: str, files_in: list[UploadFile] = File(..., alias="files")):
    """Put a person's own geometry, mesh or input into a run's folder.

    Bound to one run and to a list of file types a simulation can use. There was
    no way to bring your own geometry in at all, so the tool could only solve
    problems it could generate from a sentence."""
    try:
        run = runs.get(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "no such run")
    import re as _re
    dest = run.workdir / "uploads"
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for up in files_in:
        name = _re.sub(r"[^A-Za-z0-9._-]", "_", Path(up.filename or "file").name)[:120]
        if Path(name).suffix.lower() not in runs.UPLOAD_SUFFIXES:
            raise HTTPException(415, f"{name}: this file type is not accepted. Accepted: "
                                     + " ".join(sorted(runs.UPLOAD_SUFFIXES)))
        target = dest / name
        size = 0
        with target.open("wb") as fh:
            while chunk := await up.read(1 << 20):
                size += len(chunk)
                if size > runs.UPLOAD_MAX_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, f"{name} is larger than 500 MB.")
                fh.write(chunk)
        saved.append({"name": name, "bytes": size})
        await run.emit({"type": "file_uploaded", "name": name, "bytes": size})
    run.save()
    return {"saved": saved}


# ───────────────────────────────────────────────────────────────────
# WebSocket: a tab subscribes to a run; the run does not belong to the tab
# ───────────────────────────────────────────────────────────────────
@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    try:
        run = runs.get(sid)
    except (FileNotFoundError, ValueError):
        await ws.send_text(json.dumps({"type": "error", "message": f"no such run: {sid}"}))
        await ws.close()
        return
    run.subscribers.add(ws)
    try:
        # The live record, not a copy read from disk: a second tab used to load
        # its own stale copy and write it back over the running one on close.
        await ws.send_text(json.dumps(scrub({"type": "hello", "session": run.snapshot(),
                                             "events": run.state["events"]}), default=str))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await _handle_inbound(run, msg, ws)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws error on run %s", sid)
    finally:
        run.subscribers.discard(ws)
        # Leaving a page no longer stops the work. An idle run with nobody
        # watching gives back its openPASO server process until it is needed.
        if not run.running and not run.subscribers:
            await run.close_agent()


async def _tell(ws: WebSocket, message: str):
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps({"type": "notice", "message": message}))


async def _handle_inbound(run: "runs.Run", msg: dict, ws: WebSocket):
    t = msg.get("type")
    if t == "prompt":
        if not run.running and runs.running_count() >= config.MAX_RUNNING:
            await _tell(ws, f"{config.MAX_RUNNING} runs are already working on this machine. "
                            "Wait for one to finish or stop one, then send this again.")
            return
        await run.prompt(msg.get("text", ""), msg.get("attachments"))
    elif t == "steer":
        if run.running:
            await run.steer(msg.get("text", "").strip())
        else:
            await run.prompt(msg.get("text", ""))
    elif t == "stop":
        ended = await run.stop()
        if ended < 0:
            await _tell(ws, "Nothing is running in this run.")
    elif t == "end_step":
        why = await run.end_step(msg.get("call_id", ""))
        if why:
            await _tell(ws, why)
    elif t == "approve":
        run.gate.resolve(msg.get("call_id", ""), True)
    elif t == "reject":
        run.gate.resolve(msg.get("call_id", ""), False, msg.get("reason", ""))
    elif t == "set_mode":
        mode = msg.get("mode")
        if mode not in config.MODES:
            return
        if mode == "plan" and run.state.get("model") == config.CLAUDE_CODE_ID:
            await _tell(ws, "Claude Code cannot stop to ask before each step.")
            return
        run.state["mode"] = mode
        run.save()
        await run.emit({"type": "mode_changed", "mode": mode})
        await run.push_snapshot()
    elif t == "set_model":
        if any(e.get("type") == "user_msg" for e in run.state["events"]):
            # what produced a result must stay what the record says produced it
            await _tell(ws, "The model of a run cannot change after it has started. Start a new run.")
            return
        run.state["model"] = msg.get("model")
        await run.close_agent()
        run.save()
        await run.push_snapshot()
