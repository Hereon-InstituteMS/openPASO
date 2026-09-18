"""Smoke tests for the WebUI.

Run from repo root: .venv-lg/bin/pytest webui/tests -v

These tests use FastAPI's TestClient (HTTP) and websockets for the
streamed channel. No vLLM, no GPU; the runner uses the mock LLM so
the end-to-end spawn_subagent chain is exercised in <1 s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pytest
from fastapi.testclient import TestClient

from webui import app as webui_app
from webui import config, files, sessions


@pytest.fixture
def client():
    return TestClient(webui_app.app)


# ───────────────────────────────────────────────────────────────────
# Config endpoints
# ───────────────────────────────────────────────────────────────────
def _all_models(client):
    return [m for g in client.get("/api/models").json()["groups"] for m in g["models"]]


def test_models_say_where_they_run_and_whether_they_work(client):
    """The picker used to be one flat list: local models with no server behind
    them looked exactly like hosted ones that worked."""
    r = client.get("/api/models").json()
    kinds = {g["kind"] for g in r["groups"]}
    assert {"openrouter", "local"} <= kinds
    for g in r["groups"]:
        assert g["title"] and g["note"], "every group must say where data goes"
        for m in g["models"]:
            assert isinstance(m["available"], bool) and m["status"]
    assert any(m["id"].startswith("qwen2.5-") for m in _all_models(client))


def test_a_fresh_install_has_no_default_model_rather_than_a_server_it_lacks(monkeypatch):
    from webui import claude_code
    monkeypatch.setattr(claude_code, "available", lambda: False)
    monkeypatch.setattr(config, "openrouter_key", lambda: None)
    assert config.default_model() is None


def test_the_fake_model_is_never_offered_to_a_person(client):
    """The mock answers one canned turn and runs no solver. It was once the
    default, and a fabricated run looked exactly like real work."""
    assert "mock" not in [m["id"] for m in _all_models(client)]
    r = client.post("/api/sessions", json={"model": "mock", "mode": "accept"})
    assert r.status_code == 400, "a person must not be able to create a fake-model run"


def test_the_solver_count_is_measured_not_asserted(client):
    """The first screen used to state nine solvers from a hardcoded array.

    It must come from the registry, and when the check cannot run it must say
    so rather than guess."""
    r = client.get("/api/solvers").json()
    assert "ok" in r and "solvers" in r
    if r["ok"]:
        assert all("status" in s and "name" in s for s in r["solvers"])


def test_mcp_servers(client):
    r = client.get("/api/mcp_servers").json()
    assert any(s["id"] == "openpaso" for s in r["servers"])


def test_modes_are_only_the_ones_that_do_something(client):
    """"autonomous" was offered and read by nothing: it behaved exactly like accept."""
    r = client.get("/api/config").json()
    assert [m["id"] for m in r["modes"]] == ["plan", "accept"]
    assert all(m["label"] and m["detail"] for m in r["modes"])


def test_claude_code_cannot_be_started_in_ask_before_each_step(client):
    """It runs headless and cannot stop to ask; it used to accept the mode and
    silently ignore it."""
    r = client.post("/api/sessions", json={"model": "claude-code", "mode": "plan"})
    assert r.status_code == 400


def test_files_are_served_from_run_folders_only(client):
    for rel in ("../../etc/passwd", "campaign/cell/out.txt", "webui_abc123/../../x", ""):
        for url in ("/api/file", "/api/viz"):
            assert client.get(url, params={"rel": rel}).status_code == 403, (url, rel)
    assert client.get("/sandbox-file/campaign/cell/out.txt").status_code == 403
    # the old sandbox-wide listing and the parameter extractor are gone
    assert client.get("/api/files").status_code in (404, 405)
    assert client.post("/api/extract_params", json={"source": "N = 3"}).status_code in (404, 405)


def test_private_paths_never_reach_the_browser():
    from webui.privacy import scrub
    home = str(Path.home())
    user = Path.home().name
    removable = f"/media/{user}/disk"          # a mounted drive of the same person
    out = scrub({"a": f"{home}/x/run.py", "b": [f"/home/{user}/y"], "c": f"{removable}/z"})
    text = json.dumps(out)
    assert home not in text and removable not in text
    assert "~/x/run.py" in text


def test_classify():
    p = config.SANDBOX_ROOT / "x" / "out.vtu"
    assert files.classify(p) == "vtk"
    p2 = config.SANDBOX_ROOT / "x" / "data.csv"
    assert files.classify(p2) == "table"


# ───────────────────────────────────────────────────────────────────
# Sessions round-trip
# ───────────────────────────────────────────────────────────────────
def test_session_lifecycle(client):
    n = client.post("/api/sessions", json={"model": "mock", "test": True}).json()
    sid = n["id"]
    got = client.get(f"/api/sessions/{sid}").json()
    assert got["id"] == sid and got["model"] == "mock"
    # a run nobody prompted, and a fake-model run, are not listed as work
    listed = client.get("/api/sessions").json()["sessions"]
    assert not any(s["id"] == sid for s in listed)
    assert any(s["id"] == sid for s in client.get("/api/sessions?all=true").json()["sessions"])
    assert client.delete(f"/api/sessions/{sid}").json()["deleted"]


def test_a_run_over_the_limit_is_refused_before_it_is_created(client, monkeypatch):
    from webui import runs
    monkeypatch.setattr(runs, "running_count", lambda: config.MAX_RUNNING)
    before = len(client.get("/api/sessions?all=true").json()["sessions"])
    r = client.post("/api/sessions", json={"model": "mock", "test": True})
    assert r.status_code == 409 and "not sent" in r.json()["detail"]
    assert len(client.get("/api/sessions?all=true").json()["sessions"]) == before


def test_upload_is_bound_to_a_run_and_to_simulation_file_types(client):
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        ok = client.post(f"/api/sessions/{sid}/upload",
                         files={"files": ("my part.msh", b"$MeshFormat\n", "application/octet-stream")})
        assert ok.status_code == 200 and ok.json()["saved"][0]["name"] == "my_part.msh"
        bad = client.post(f"/api/sessions/{sid}/upload",
                          files={"files": ("run.sh", b"rm -rf /", "text/plain")})
        assert bad.status_code == 415
        listing = client.get(f"/api/sessions/{sid}/files", params={"sub": "uploads"}).json()
        assert [e["name"] for e in listing["entries"]] == ["my_part.msh"]
        assert "/home/" not in json.dumps(listing), "absolute paths must not reach the browser"
        assert client.get(f"/api/sessions/{sid}/files", params={"sub": "../../"}).status_code in (403, 404)
    finally:
        client.delete(f"/api/sessions/{sid}")


def _solver_result(status, trustworthy=None):
    body = {"status": status}
    if trustworthy is not None:
        body["trustworthy_result"] = trustworthy
    return "[{'type': 'text', 'text': '" + json.dumps(body) + "'}]"


def test_a_solver_result_counts_only_when_openpaso_verified_it():
    from webui.outcome import classify_solver_result as c
    assert c(_solver_result("completed", True)) == "verified"
    assert c(_solver_result("completed", False)) == "unverified"
    assert c(_solver_result("completed")) == "unverified"
    assert c(_solver_result("completed_with_warnings", True)) == "unverified"
    assert c(_solver_result("failed")) == "failed"
    assert c("[{'type': 'text', 'text': 'Unknown solver: abaqus'}]") == "failed"
    assert c("") == "failed"


def test_outcome_is_judged_per_turn_and_needs_a_verified_solver_result():
    from webui.outcome import fold
    ev = lambda *xs: [x if isinstance(x, dict) else {"type": x} for x in xs]
    tool = lambda raw: {"type": "tool_result", "tool": "run_simulation", "result": raw}
    solved = tool(_solver_result("completed", True))
    shaky = tool(_solver_result("completed", False))
    unknown = tool("[{'type': 'text', 'text': 'Unknown solver: abaqus'}]")
    assert fold(ev("turn_start", "user_msg", "done")) == "no_result"
    assert fold(ev("turn_start", "user_msg", solved, "done")) == "completed"
    assert fold(ev("turn_start", "user_msg", shaky, "done")) == "unverified"
    assert fold(ev("turn_start", "user_msg", unknown, "done")) == "no_result"
    # an old record's "done: completed" is not taken on trust
    assert fold(ev("turn_start", "user_msg", {"type": "done", "outcome": "completed"})) == "no_result"
    assert fold(ev("turn_start", "user_msg", {"type": "error", "outcome": "failed"}, "done")) == "failed"
    # an error without an outcome of its own is a failure
    assert fold(ev("turn_start", "user_msg", solved, {"type": "error"}, "done")) == "failed"
    # a follow-up still working is running, whatever the turn before did
    assert fold(ev("turn_start", "user_msg", solved, "done", "turn_start", "user_msg")) == "running"
    # a follow-up that ran no solver has no result even if an earlier turn did
    assert fold(ev("turn_start", "user_msg", solved, "done", "turn_start", "user_msg", "done")) == "no_result"


def test_stop_ends_processes_started_in_the_run_folder():
    """Stop used to cancel the waiting coroutine and leave the command and its
    solver running while the screen said the run had stopped."""
    import subprocess, tempfile, time
    from webui import proctree
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.Popen(["bash", "-c", "sleep 120 & wait"], cwd=d, start_new_session=True)
        time.sleep(0.5)
        assert proc.pid in proctree.run_processes(Path(d))
        assert proctree.end_run_processes(Path(d)) >= 1
        proc.wait(timeout=10)
        assert proctree.run_processes(Path(d)) == []


def test_a_correction_is_handed_to_the_model_with_the_next_tool_result():
    from langchain_core.tools import StructuredTool
    from webui.runner import ApprovalGate, _wrap_tool
    pending = [{"id": "st_1", "text": "Use a finer mesh."}]
    def take():
        out = list(pending); pending.clear(); return out
    async def emitter(_e):
        return None
    tool = StructuredTool.from_function(func=lambda x: f"ran {x}", name="t", description="d")
    wrapped = _wrap_tool(tool, emitter=emitter, get_mode=lambda: "accept",
                         gate=ApprovalGate(), take_steers=take)
    result = asyncio.run(wrapped.ainvoke({"x": "1"}))
    assert "ran 1" in result and "Use a finer mesh." in result and "MESSAGE FROM THE USER" in result
    assert pending == []


def test_after_a_stop_the_model_is_given_the_steps_it_really_took():
    """Re-seeding used to hand back prompts and messages only, so after a Stop
    the model said nothing had been computed while the page showed the solves."""
    from webui.runs import _history
    events = [
        {"type": "turn_start"}, {"type": "user_msg", "text": "Solve the cantilever."},
        {"type": "agent_msg", "text": "I will run three meshes."},
        {"type": "tool_call_pending", "call_id": "a", "tool": "run_bash", "agent": "main",
         "args": {"command": "python cantilever.py 40"}},
        {"type": "tool_result", "call_id": "a", "tool": "run_bash", "result": "tip deflection -1.8e-3 relative error"},
        {"type": "tool_call_pending", "call_id": "b", "tool": "knowledge", "agent": "main", "args": {"query": "x"}},
        {"type": "tool_call_rejected", "call_id": "b"},
        {"type": "error", "outcome": "interrupted", "message": "Run stopped."},
        {"type": "done", "outcome": "interrupted"},
    ]
    h = _history(events)
    assert h[0] == ("user", "Solve the cantilever.")
    text = h[1][1]
    assert h[1][0] == "assistant"
    assert "python cantilever.py 40" in text and "-1.8e-3" in text
    assert "skipped this step" in text and "stopped the run" in text


def test_one_hung_step_can_be_ended_and_the_run_carries_on():
    """A correction waits for the current step; a step that hangs used to leave
    Stop, which ends the whole run, as the only way out."""
    import asyncio as aio, subprocess, time
    from langchain_core.tools import StructuredTool
    from webui import proctree
    from webui.runner import ApprovalGate, StepControl, _wrap_tool
    with tempfile.TemporaryDirectory() as d:
        def hang(x: str) -> str:
            r = subprocess.run(["bash", "-c", "sleep 300"], cwd=d, capture_output=True, text=True)
            return f"exit {r.returncode}"
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", "-m", "server"], cwd=d)
        try:
            steps = StepControl()
            events = []
            async def emitter(e):
                events.append(e)
            tool = StructuredTool.from_function(func=hang, name="run_bash", description="d")
            wrapped = _wrap_tool(tool, emitter=emitter, get_mode=lambda: "accept", gate=ApprovalGate(),
                                 steps=steps, take_steers=lambda: [{"id": "s", "text": "Skip the MPI test."}])
            async def main():
                call = aio.create_task(wrapped.ainvoke({"x": "1"}))
                for _ in range(50):
                    await aio.sleep(0.1)
                    if steps.running():
                        break
                await aio.sleep(0.5)
                cid = steps.running()[0]
                steps.ended.add(cid)
                await aio.to_thread(proctree.end_step_processes, Path(d))
                return await aio.wait_for(call, timeout=15)
            t0 = time.monotonic()
            result = aio.run(main())
            assert time.monotonic() - t0 < 15
            assert "The user ended this step" in result and "Skip the MPI test." in result
            assert server.poll() is None, "the openPASO server of the run must survive"
            assert steps.tasks == {} and steps.ended == set()
        finally:
            server.kill()


# ───────────────────────────────────────────────────────────────────
# Runner end-to-end with mock LLM.
#
# We exercise the agent flow directly via runner.build_agent_for_session
# + runner.stream_turn rather than through TestClient's WebSocket. The
# TestClient's anyio-driven WS loop deadlocks against asyncio.to_thread
# inside the gated spawn_subagent tool; the runner itself works fine
# under plain asyncio, which is what the live server uses.
# ───────────────────────────────────────────────────────────────────
import asyncio
import tempfile
from contextlib import asynccontextmanager

from webui.runner import (ApprovalGate, build_agent_for_session,
                          open_agent_for_session, stream_turn)


def _run_turn(*, mode, prompt, approve_first=False):
    seen, events = [], []

    async def main():
        async def emitter(e):
            seen.append(e["type"])
            events.append(e)

        gate = ApprovalGate()

        def get_mode():
            return mode

        with tempfile.TemporaryDirectory() as d:
            agent = build_agent_for_session(
                model="mock", mcp_on=False, workdir=Path(d),
                emitter=emitter, get_mode=get_mode, gate=gate)

            turn = asyncio.create_task(stream_turn(
                agent=agent, user_text=prompt, emitter=emitter, emit_done=True))

            if approve_first:
                # Wait until the first pending call appears, then approve.
                while turn.done() is False:
                    pending = next((e for e in events
                                    if e["type"] == "tool_call_pending"), None)
                    if pending:
                        gate.resolve(pending["call_id"], True)
                        break
                    await asyncio.sleep(0.05)
            return await turn

    final = asyncio.run(main())
    return final, seen, events


def test_runner_mock_run_accept():
    final, seen, _ = _run_turn(
        mode="accept", prompt="Plan a Poisson MMS demo. Use the critic.")
    assert "subagent_spawned" in seen, (
        f"expected spawn_subagent in {set(seen)}")
    assert "subagent_returned" in seen
    assert "done" in seen
    assert "critic approved" in final.lower()


def test_runner_mock_plan_mode_pending_then_approve():
    final, seen, events = _run_turn(
        mode="plan", prompt="Begin.", approve_first=True)
    pending = [e for e in events if e["type"] == "tool_call_pending"]
    assert pending, "no tool_call_pending event in plan mode"
    # After approval the call must have executed (tool_result) and the
    # turn must have terminated (done).
    assert any(e["type"] == "tool_result" for e in events)
    assert "done" in seen, f"missing done in {set(seen)}"


def test_webui_keeps_mcp_context_open_for_agent_lifetime(monkeypatch):
    import agent as langgraph_agent

    lifecycle = []

    @asynccontextmanager
    async def fake_mcp_session(_workdir, **kwargs):
        # the product gets every tool and no sandbox jail; the campaign's
        # defaults (a fixed subset, bubblewrap) are for measurements only
        assert kwargs == {"surface": "all", "isolate": False}, kwargs
        lifecycle.append("opened")
        try:
            yield []
        finally:
            lifecycle.append("closed")

    monkeypatch.setattr(
        langgraph_agent, "openpaso_mcp_tools_session", fake_mcp_session)

    async def main():
        async def emitter(_event):
            return None

        with tempfile.TemporaryDirectory() as directory:
            async with open_agent_for_session(
                    model="mock", mcp_on=True, workdir=Path(directory),
                    emitter=emitter, get_mode=lambda: "accept",
                    gate=ApprovalGate()) as built:
                assert built is not None
                assert lifecycle == ["opened"]
            assert lifecycle == ["opened", "closed"]

    asyncio.run(main())


def test_a_run_does_not_belong_to_the_browser_tab():
    """Closing the socket used to cancel the run. The run now lives on the
    server; a mock turn completes with nobody subscribed."""
    from webui import runs
    s = sessions.new_session(model="mock", mode="accept", mcp_servers=[])
    sessions.save(s)
    try:
        async def main():
            run = runs.get(s["id"])
            assert not run.subscribers
            await run.prompt("Plan a Poisson demo. Use the critic.")
            await asyncio.wait_for(run.turn_task, timeout=60)
            await asyncio.sleep(0.2)
            types = [e["type"] for e in run.state["events"]]
            assert types[0] == "turn_start" and "user_msg" in types
            done = [e for e in run.state["events"] if e["type"] == "done"]
            assert len(done) == 1 and done[0]["outcome"] == "no_result", done
            # the tool connection is opened and closed by one task, whichever task asks
            closer = asyncio.create_task(run.close_agent())
            await closer
            assert run.agent is None and run._keeper is None
        asyncio.run(main())
    finally:
        runs.RUNS.pop(s["id"], None)
        sessions.delete(s["id"])
