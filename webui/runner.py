"""Agent runner that streams structured events to a single WebSocket.

Wraps the existing ``langgraph_eval.agent`` factories and surfaces every
intermediate step the user wants to see in the UI:

* user_msg / agent_msg / agent_chunk
* tool_call_pending  → in plan mode, blocks waiting for an approve/reject
                       event from the client before executing
* tool_call_executing / tool_result
* subagent_spawned / subagent_returned (via a callback we wire into
  spawn_subagent so its children are visible)
* token_count after each LLM call
* error / done

The runner is mode-aware: plan mode gates every tool call; accept mode
auto-approves; autonomous is identical to accept but the UI is told not
to interrupt.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from . import config

REPO = config.REPO
sys.path.insert(0, str(REPO / "langgraph_eval"))


# ───────────────────────────────────────────────────────────────────
# Mock LLM (no GPU) — drives a deterministic plan→spawn_critic→done
# chain so the UI is usable without vLLM. Real models go through
# ChatOpenAI as configured in langgraph_eval/agent.py.
#
# Implementation reuses the langgraph_eval/_mock_openai_server.py
# fake OpenAI server (proven by the langgraph_eval smoke tests). We
# start one thread per app process on first mock use; subsequent
# sessions all point at the same loopback endpoint.
# ───────────────────────────────────────────────────────────────────
_MOCK_PORT = 18234
_MOCK_STARTED = False


def _ensure_mock_server():
    global _MOCK_STARTED
    if _MOCK_STARTED:
        return
    import _mock_openai_server as mock_srv  # from langgraph_eval/
    mock_srv.start(port=_MOCK_PORT)
    _MOCK_STARTED = True


def _mock_chat_model():
    _ensure_mock_server()
    from langchain_openai import ChatOpenAI
    # disable_streaming forces astream_events to go through the
    # non-streaming agenerate path, so the mock server's plain JSON
    # response works (no SSE needed).
    return ChatOpenAI(
        base_url=f"http://127.0.0.1:{_MOCK_PORT}/v1",
        api_key="not-used", model="mock-qwen",
        temperature=0.0, timeout=30,
        disable_streaming=True,
    )


# ───────────────────────────────────────────────────────────────────
# Per-session workdir
# ───────────────────────────────────────────────────────────────────
def _session_workdir(session_id: str) -> Path:
    p = config.SANDBOX_ROOT / f"webui_{session_id}" / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ───────────────────────────────────────────────────────────────────
# Approval gate (plan mode)
# ───────────────────────────────────────────────────────────────────
class ApprovalGate:
    """A future-per-pending-call gate. The UI signals approve/reject via
    ``resolve`` from a separate code path."""

    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}

    def open(self, call_id: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[call_id] = fut
        return fut

    def resolve(self, call_id: str, approved: bool, reason: str = ""):
        fut = self._pending.pop(call_id, None)
        if fut and not fut.done():
            fut.set_result({"approved": approved, "reason": reason})


# ───────────────────────────────────────────────────────────────────
# Tool wrapping for mode gating + event emission
# ───────────────────────────────────────────────────────────────────
def _wrap_tool(tool, *, emitter, get_mode, gate, agent_label="agent"):
    """Return a copy of ``tool`` whose invoke emits events and (when in
    plan mode) waits for explicit approval before running.

    The original tool's func is preserved; we only intercept invocation.
    """
    from langchain_core.tools import BaseTool

    class Gated(BaseTool):
        name: str = tool.name
        description: str = tool.description
        args_schema: Any = getattr(tool, "args_schema", None)

        async def _arun(self, *args, **kwargs):
            call_id = f"tc_{uuid.uuid4().hex[:8]}"
            await emitter({"type": "tool_call_pending",
                           "call_id": call_id, "tool": tool.name,
                           "args": kwargs, "agent": agent_label})
            mode = get_mode()
            if mode == "plan":
                fut = gate.open(call_id)
                decision = await fut
                if not decision["approved"]:
                    await emitter({"type": "tool_call_rejected",
                                   "call_id": call_id,
                                   "reason": decision.get("reason", "")})
                    return f"[rejected by user: {decision.get('reason','')}]"
            await emitter({"type": "tool_call_executing",
                           "call_id": call_id, "tool": tool.name})
            try:
                if hasattr(tool, "ainvoke"):
                    result = await tool.ainvoke(kwargs)
                else:
                    result = tool.invoke(kwargs)
            except Exception as e:
                await emitter({"type": "tool_error",
                               "call_id": call_id,
                               "error": f"{type(e).__name__}: {e}"})
                raise
            await emitter({"type": "tool_result",
                           "call_id": call_id, "tool": tool.name,
                           "result": str(result)[:8000]})
            return result

        def _run(self, *args, **kwargs):
            # asyncio.get_event_loop() is deprecated and, from Python 3.12, raises
            # "There is no current event loop" once anything in the process has
            # already run asyncio.run() and closed the loop behind itself. The
            # failure therefore depends on what ran before, which is the worst
            # kind. asyncio.run() makes and closes its own loop, and refuses just
            # the same as the old code did if one is already running.
            return asyncio.run(self._arun(*args, **kwargs))

    return Gated()


# ───────────────────────────────────────────────────────────────────
# Agent factories with WebUI hooks
# ───────────────────────────────────────────────────────────────────
def build_agent_for_session(*, model: str, mcp_on: bool,
                            workdir: Path,
                            emitter,
                            get_mode,
                            gate: ApprovalGate,
                            _mcp_tools=None):
    """Build a LangGraph ReAct agent with all WebUI hooks wired in.

    * ``model`` is a key from ``config.MODELS``. ``mock`` skips vLLM.
    * ``mcp_on`` attaches openPASO via langchain-mcp-adapters when True.
    * ``emitter`` is an async function ``(event_dict) -> None`` used to
      stream events back over the WebSocket.
    * ``get_mode`` is a callable returning the current mode string.
    * ``gate`` is the per-session ApprovalGate.
    """
    import agent as la

    # ── LLM
    if model == "mock":
        llm = _mock_chat_model()
    elif model in config.OPENROUTER_MODELS:
        key = config.openrouter_key()
        if not key:
            raise ValueError(
                "No OpenRouter key. Copy .env.example to .env and paste your "
                "key after OPENROUTER_API_KEY=, then pick this model again.")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=config.OPENROUTER_URL, api_key=key, model=model,
            temperature=0.2, timeout=600,
        )
    else:
        if model not in config.MODELS:
            raise ValueError(f"unknown model: {model}")
        port = config.MODELS[model]["port"]
        from langchain_openai import ChatOpenAI
        # disable_streaming forces a single JSON POST to /v1/chat/
        # completions instead of an SSE stream. The user's local
        # transformers_openai_server.py returns plain JSON, which
        # langchain-openai's streaming parser refuses ("No generations
        # found in stream"); vLLM works in either mode. So we always
        # disable streaming for predictability.
        llm = ChatOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-used", model=model,
            temperature=0.2, timeout=900,
            disable_streaming=True,
        )

    # Stateful MCP tools must be supplied by open_agent_for_session(), whose
    # context stays alive until the WebSocket session closes.
    if mcp_on and _mcp_tools is None:
        raise RuntimeError(
            "MCP-enabled WebUI agents must be built with "
            "open_agent_for_session() so server state persists across calls")
    mcp_tools = list(_mcp_tools or [])

    # ── Host tools (bash/read/write/web_search/spawn_subagent)
    host = []
    host.append(la._bash_tool_for(workdir))
    host.extend(la._read_write_tools_for(workdir))
    host.append(la.web_search)

    # spawn_subagent: build the sub-agent INLINE so it uses the parent's
    # active model (mock vs vLLM). Using langgraph_eval's factory
    # hard-codes a vLLM endpoint by size, which breaks the mock path.
    from langgraph.prebuilt import create_react_agent

    def _sub_llm():
        if model == "mock":
            return _mock_chat_model()
        from langchain_openai import ChatOpenAI
        port = config.MODELS[model]["port"]
        return ChatOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-used", model=model,
            temperature=0.3, timeout=600,
            disable_streaming=True,
        )

    _SUB_PROMPTS = {
        "critic": ("You are a ruthlessly critical reviewer. Challenge "
                   "every parameter choice, check units, look for sign "
                   "errors, verify BCs and validate against literature "
                   "via web_search if available. Respond with one of: "
                   "APPROVED: <reason> | REJECTED: <issue and fix>."),
        "verifier": ("You are an independent verifier. Re-derive the "
                     "requested quantity from first principles or by "
                     "an alternative method, then compare numerically."),
        "researcher": ("You are a research assistant. Look up "
                       "authoritative sources for the requested "
                       "information and summarise."),
    }

    async def spawn_subagent_emitting(role: str, task: str,
                                      context: str = "") -> str:
        sa_id = f"sa_{uuid.uuid4().hex[:8]}"
        await emitter({"type": "subagent_spawned", "sa_id": sa_id,
                       "role": role, "task": task, "context": context})
        sub_tools = [t for t in (mcp_tools + host)
                     if t.name != "spawn_subagent"]
        sys = _SUB_PROMPTS.get(role, _SUB_PROMPTS["researcher"])
        sub_agent = create_react_agent(_sub_llm(), tools=sub_tools,
                                       prompt=sys)
        msg = f"Task: {task}\n\nContext provided by parent:\n{context}"
        try:
            # ainvoke avoids the inner asyncio.run() that the sync .invoke
            # would require, and keeps us on the caller's event loop.
            out = await sub_agent.ainvoke(
                {"messages": [("user", msg)]},
                config={"recursion_limit": 40})
            res = out["messages"][-1].content
        except Exception as e:
            res = f"[sub-agent error: {type(e).__name__}: {e}]"
        await emitter({"type": "subagent_returned",
                       "sa_id": sa_id, "result": str(res)[:6000]})
        return res

    from langchain_core.tools import StructuredTool
    spawn_wrapped = StructuredTool.from_function(
        coroutine=spawn_subagent_emitting,
        name="spawn_subagent",
        description=("Spawn a sub-agent. role∈{critic, verifier, "
                     "researcher}. task = what it should do. context = "
                     "facts to pass in. Returns its final message."),
    )
    host.append(spawn_wrapped)

    # ── Gate every tool through the wrapper for mode gating
    gated = [_wrap_tool(t, emitter=emitter, get_mode=get_mode, gate=gate,
                        agent_label="main") for t in mcp_tools + host]

    from langgraph.prebuilt import create_react_agent
    # Use the EXACT same system prompts as the langgraph_eval driver, including
    # the MANDATORY CRITIC paragraph. Softening them in the WebUI would change
    # the agent's behaviour relative to the paper claim, and any failure mode
    # the strict prompt causes on small models is a real finding, not a bug to
    # paper over.
    #
    # The openPASO arm's text became a function when the server's own
    # instructions became its source, so it is built per call rather than read
    # from a constant. Calling the private name is deliberate: the guarantee
    # above is that these are the same bytes the driver uses, and a local copy
    # would quietly stop being that.
    prompt = (la._mcp_system_prompt() if mcp_on else la.BARE_SYSTEM)
    return create_react_agent(llm, tools=gated, prompt=prompt)


@asynccontextmanager
async def open_agent_for_session(**kwargs):
    """Yield one WebUI agent and keep its MCP process alive across turns."""
    import agent as la

    workdir = kwargs["workdir"]
    try:
        if kwargs.get("mcp_on"):
            async with la.openpaso_mcp_tools_session(workdir) as mcp_tools:
                yield build_agent_for_session(
                    **kwargs, _mcp_tools=mcp_tools)
        else:
            yield build_agent_for_session(**kwargs)
    finally:
        la.cleanup_sandbox_scratch(workdir)


# ───────────────────────────────────────────────────────────────────
# Streamed turn
# ───────────────────────────────────────────────────────────────────
async def stream_turn(*, agent, user_text: str, emitter, emit_done: bool = True):
    """Run one user turn. Streams chunks/events via ``emitter`` and
    returns the final message text. Emits a 'thinking' status as soon
    as we start so the user sees activity even before the first model
    response, and an 'error' event with a clear message on any failure
    (rather than dying silently)."""
    final_text = ""
    inputs = {"messages": [("user", user_text)]}
    await emitter({"type": "status", "message": "thinking…"})
    try:
        # LangGraph's default ceiling is 25 steps, which a simulation task
        # passes while it is still reading documentation: a real run died at it
        # having written its solver but never run it. openPASO's own rule is
        # that there is one budget and it is the clock, so the step ceiling sits
        # well above anything the wall time can reach.
        async for event in agent.astream_events(
                inputs, version="v2", config={"recursion_limit": 400}):
            kind = event.get("event")
            name = event.get("name")
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and getattr(chunk, "content", None):
                    await emitter({"type": "agent_chunk",
                                   "text": chunk.content})
            elif kind == "on_chat_model_end":
                gen = event["data"].get("output")
                if gen is not None and hasattr(gen, "usage_metadata"):
                    um = gen.usage_metadata or {}
                    await emitter({"type": "token_count",
                                   "input": um.get("input_tokens"),
                                   "output": um.get("output_tokens")})
                if gen is not None and getattr(gen, "content", ""):
                    final_text = gen.content
                    # Non-streaming models (disable_streaming=True) won't
                    # emit on_chat_model_stream chunks, so surface the
                    # full content here as an agent_msg.
                    await emitter({"type": "agent_msg",
                                   "text": gen.content})
            elif kind == "on_chain_end" and name == "LangGraph":
                msgs = event["data"].get("output", {}).get("messages") or []
                if msgs:
                    last = msgs[-1]
                    final_text = (getattr(last, "content", "")
                                  or final_text)
    except Exception:
        # The caller decides the terminal state and reports it. Emitting a
        # "done" here as well is what let a crashed run read as "finished".
        raise
    if emit_done:
        await emitter({"type": "done", "final_text": final_text,
                       "outcome": "completed"})
    return final_text
