"""Run a turn through the Claude Code CLI instead of an API model.

Most people who would try openPASO already have Claude Code and do not have an
API key, so this is the path that needs no key and no GPU: the CLI is driven
headless, openPASO is handed to it as an MCP server, and its streamed JSON is
translated into the same events the rest of the WebUI already speaks.

Nothing here knows about the agent loop. Claude Code runs its own.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import AsyncIterator, Callable

from . import config

BINARY = "claude"


def available() -> bool:
    return shutil.which(BINARY) is not None


def _mcp_config(servers: list[str]) -> dict:
    """The --mcp-config payload, built from the same entries the WebUI shows."""
    out: dict[str, dict] = {}
    for sid in servers:
        spec = config.MCP_SERVERS.get(sid)
        if not spec:
            continue
        env = dict(spec.get("env_extra") or {})
        out[sid] = {
            "command": spec["command"],
            "args": list(spec.get("args") or []),
            "env": env,
        }
    return {"mcpServers": out}


# The work an agent has to do here is write an input deck and run a solver, so
# it needs the file and shell tools as well as openPASO's own. Naming them
# explicitly is the point: the run is allowed to do these things and nothing
# else, rather than being handed a blanket bypass.
WORK_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch"]


async def stream_turn(
    task: str,
    *,
    workdir: Path,
    servers: list[str],
    model: str | None = None,
    mode: str = "accept",
    emit: Callable[[dict], "asyncio.Future | None"],
    state: dict | None = None,
) -> str:
    """Run one task and emit WebUI events. Returns the final text."""
    if not available():
        raise RuntimeError(
            "Claude Code is not on PATH. Install it, or pick an API model instead.")
    if mode == "plan":
        raise RuntimeError(
            "Claude Code runs headless here, so it cannot stop and ask you to "
            "approve each step. Choose \"Run without asking\" for it, or pick "
            "another model to keep \"Ask before each step\".")

    cfg = _mcp_config(servers)
    tmp = Path(tempfile.mkdtemp(prefix="openpaso-cc-"))
    cfg_path = tmp / "mcp.json"
    cfg_path.write_text(json.dumps(cfg))

    allowed = list(WORK_TOOLS) + [f"mcp__{sid}" for sid in cfg["mcpServers"]]
    cmd = [
        BINARY, "--print", task,
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "acceptEdits",
        "--allowedTools", *allowed,
        "--mcp-config", str(cfg_path),
    ]
    if model:
        cmd += ["--model", model]
    # A follow-up continues the same Claude conversation instead of starting blind.
    if state and state.get("claude_session_id"):
        cmd += ["--resume", state["claude_session_id"]]

    workdir.mkdir(parents=True, exist_ok=True)
    root = str(workdir.resolve())
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=root,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        # one stream-json line can hold a whole file Claude read; the default
        # 64 KB line limit crashed runs with "chunk is longer than limit"
        limit=64 * 1024 * 1024,
        # its own process group, and the run's marker, so Stop can end it and
        # everything it launched (it was left running before)
        env={**os.environ, "PYVISTA_OFF_SCREEN": "true", "OPENPASO_CELL_WORKDIR": root},
        start_new_session=True,
    )
    try:
        return await _consume(proc, emit, state)
    except asyncio.CancelledError:
        _kill(proc)
        raise
    finally:
        # one of these per turn, left behind on a long-lived server
        shutil.rmtree(tmp, ignore_errors=True)


def _kill(proc) -> None:
    import signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return


async def _consume(proc, emit, state) -> str:

    final = ""
    calls: dict[str, str] = {}
    assert proc.stdout is not None
    async for line in _lines(proc.stdout):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = msg.get("type")
        if kind == "system" and msg.get("subtype") == "init":
            # which Claude model actually ran: the record used to say only
            # "claude-code", so nobody could tell what produced a number
            if state is not None:
                state["claude_session_id"] = msg.get("session_id") or state.get("claude_session_id")
                if msg.get("model"):
                    state["claude_model"] = msg["model"]
            if msg.get("model"):
                await _send(emit, {"type": "model_info", "model": msg["model"]})
        elif kind == "assistant":
            for block in (msg.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and block.get("text", "").strip():
                    await _send(emit, {"type": "agent_msg", "text": block["text"]})
                elif block.get("type") == "tool_use":
                    cid = block.get("id") or uuid.uuid4().hex[:8]
                    # Strip the MCP prefix so the step reads as the tool's name.
                    name = str(block.get("name", "")).split("__")[-1]
                    calls[cid] = name
                    await _send(emit, {"type": "tool_call_pending", "call_id": cid,
                                       "tool": name, "args": block.get("input") or {}})
                    await _send(emit, {"type": "tool_call_executing",
                                       "call_id": cid, "tool": name})
        elif kind == "user":
            for block in (msg.get("message") or {}).get("content") or []:
                if block.get("type") != "tool_result":
                    continue
                cid = block.get("tool_use_id", "")
                body = block.get("content")
                if isinstance(body, list):
                    body = " ".join(b.get("text", "") for b in body
                                    if isinstance(b, dict))
                await _send(emit, {"type": "tool_result", "call_id": cid,
                                   "tool": calls.get(cid, ""),
                                   "result": str(body)[:8000]})
        elif kind == "result":
            final = msg.get("result") or final
            if state is not None and msg.get("session_id"):
                state["claude_session_id"] = msg["session_id"]
            cost = msg.get("total_cost_usd")
            if cost is not None:
                await _send(emit, {"type": "cost", "usd": cost})
            usage = msg.get("usage") or {}
            if usage:
                await _send(emit, {"type": "token_count",
                                   "input": usage.get("input_tokens", 0),
                                   "output": usage.get("output_tokens", 0)})

    err = (await proc.stderr.read()).decode("utf-8", "replace") if proc.stderr else ""
    await proc.wait()
    if proc.returncode != 0 and not final:
        raise RuntimeError(f"Claude Code exited {proc.returncode}: {err.strip()[:400]}")
    return final


async def _lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    while True:
        raw = await stream.readline()
        if not raw:
            return
        text = raw.decode("utf-8", "replace").strip()
        if text:
            yield text


async def _send(emit, event: dict) -> None:
    res = emit(event)
    if asyncio.isfuture(res) or asyncio.iscoroutine(res):
        await res
