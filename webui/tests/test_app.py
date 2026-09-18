"""Smoke tests for the WebUI.

Run from repo root: .venv-lg/bin/pytest webui/tests -v

These tests use FastAPI's TestClient (HTTP) and websockets for the
streamed channel. No vLLM, no GPU; the runner uses the mock LLM so
the end-to-end spawn_subagent chain is exercised in <1 s.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pytest
from fastapi.testclient import TestClient

from webui import app as webui_app
from webui import catalog, config, files, runs, sessions


@pytest.fixture(autouse=True)
def allow_the_test_model(monkeypatch):
    """The fake model is refused unless the server was started for testing
    (OPENPASO_TEST_MODEL). These tests are that case, and say so."""
    monkeypatch.setattr(config, "ALLOW_TEST_MODEL", True)


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


def test_the_fake_model_needs_the_server_to_be_started_for_testing(client, monkeypatch):
    """Asking for it over the API is not enough: this API has no authentication,
    and the fake model fabricates answers."""
    monkeypatch.setattr(config, "ALLOW_TEST_MODEL", False)
    r = client.post("/api/sessions", json={"model": "mock", "test": True})
    assert r.status_code == 400 and "mock" in r.json()["detail"]


def test_an_approval_that_arrives_first_is_not_lost():
    """The step is announced before the gate opens, so a fast client can answer
    before there is anything to answer, and the run waited for ever."""
    import asyncio as aio
    from webui.runner import ApprovalGate

    async def main():
        gate = ApprovalGate()
        gate.resolve("tc_1", True)            # the answer, before the question
        decision = await aio.wait_for(gate.open("tc_1"), timeout=1)
        assert decision["approved"] is True

    aio.run(main())


def test_switching_to_run_without_asking_releases_a_waiting_step():
    import asyncio as aio
    from webui.runner import ApprovalGate

    async def main():
        gate = ApprovalGate()
        fut = gate.open("tc_9")
        assert gate.open_all(True) == 1
        assert (await aio.wait_for(fut, timeout=1))["approved"] is True

    aio.run(main())


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


def test_stop_pressed_straight_after_send_still_stops_the_run():
    """The flag that says a turn is over was cleared inside the turn's own task,
    so a Stop in the same breath as Send was told nothing was running."""
    import asyncio as aio
    from webui import runs as runs_mod
    run = runs_mod.Run.__new__(runs_mod.Run)
    run.state = {"events": [], "mode": "accept"}
    run._turn_ended = True
    run.turn_task = None            # `running` is read from this
    run.steers = []
    run.push_snapshot = lambda: aio.sleep(0)
    run.emit = lambda e: aio.sleep(0)

    async def never_ending(*a, **k):
        await aio.sleep(30)

    run._turn = never_ending

    async def main():
        await runs_mod.Run.prompt(run, "do something")
        # what stop() looks at, before the turn task has had any time to run
        assert run._turn_ended is False
        assert run.turn_task is not None and not run.turn_task.done()
        run.turn_task.cancel()

    aio.run(main())


def test_an_xdmf_file_is_not_read_as_hdf5(tmp_path):
    """h5py cannot read XDMF, and the failure was swallowed: the file was
    labelled HDF5 with no contents, and the .h5 holding the numbers never shown."""
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "out.xdmf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('<Xdmf><Domain><Grid><Geometry><DataItem Format="HDF">mesh.h5:/points'
                 '</DataItem></Geometry><Attribute><DataItem Format="HDF">field.h5:/u'
                 '</DataItem></Attribute></Grid></Domain></Xdmf>')
    try:
        out = viz._hdf(p)
        assert out["kind"] == "xdmf"
        assert out["data_files"] == ["mesh.h5", "field.h5"]
        assert "<Xdmf>" in out["text"]
    finally:
        p.unlink()


def test_a_run_cannot_be_started_on_a_model_that_cannot_run(client, monkeypatch):
    async def nothing_available():
        return {"groups": [{"kind": "openrouter", "title": "t", "note": "", "models": [
            {"id": "deepseek/deepseek-v4.1-flash", "label": "DeepSeek v4.1 Flash",
             "kind": "openrouter", "available": False, "status": "no OpenRouter key"}]}],
            "default": None}
    monkeypatch.setattr(catalog, "models", nothing_available)
    r = client.post("/api/sessions", json={"model": "deepseek/deepseek-v4.1-flash", "mode": "accept"})
    assert r.status_code == 409 and "no OpenRouter key" in r.json()["detail"]
    assert "not sent" in r.json()["detail"]


def test_a_long_result_keeps_the_verdict_at_its_end():
    """openPASO stamps its verification verdict at the END of a report. Cutting
    a long result to its first 8000 characters threw that away, and a verified
    run then showed as unverified."""
    from webui.outcome import RESULT_LIMIT, classify_solver_result, shorten
    body = json.dumps({"status": "completed", "log": "x" * 40000, "trustworthy_result": True})
    raw = "[{'type': 'text', 'text': '" + body + "'}]"
    assert len(raw) > RESULT_LIMIT
    assert classify_solver_result(shorten(raw)) == "verified"
    assert len(shorten(raw)) <= RESULT_LIMIT + 200


def test_claude_code_writes_its_results_into_the_run(tmp_path):
    """Its openPASO server took the shared install directories, so a Claude Code
    run's output landed outside the run and two runs could collide."""
    from webui import claude_code
    cfg = claude_code._mcp_config(["openpaso"], tmp_path)
    env = cfg["mcpServers"]["openpaso"]["env"]
    for key in ("OPENPASO_CELL_WORKDIR", "OPENPASO_OUTPUT_DIR", "OPENPASO_COUPLING_DIR",
                "OPENPASO_MESH_DIR", "OPENPASO_BENCHMARK_DIR"):
        assert env[key].startswith(str(tmp_path.resolve())), (key, env[key])


def test_every_text_file_is_scrubbed_however_it_is_named(client):
    """Deciding by suffix let a solver's own formats through unscrubbed."""
    work = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    work.mkdir(parents=True, exist_ok=True)
    deck = work / "case.4c"
    deck.write_text(f"MESHFILE: {Path.home()}/meshes/part.msh\n")
    binary = work / "field.bin"
    binary.write_bytes(b"\x00\x01\x02" + str(Path.home()).encode())
    try:
        text = client.get("/sandbox-file/webui_abc123def/work/case.4c")
        assert str(Path.home()) not in text.text and "~/meshes/part.msh" in text.text
        # a binary file is served as it is; scrubbing it would corrupt it
        assert client.get("/sandbox-file/webui_abc123def/work/field.bin").status_code == 200
    finally:
        deck.unlink(); binary.unlink()


def test_a_download_is_scrubbed_like_everything_else(client, tmp_path, monkeypatch):
    """The download route hands over whole files; it used to be the one way a
    home path could reach the browser."""
    run = config.SANDBOX_ROOT / "webui_abc123def"
    (run / "work").mkdir(parents=True, exist_ok=True)
    log = run / "work" / "solver.log"
    log.write_text(f"reading mesh from {Path.home()}/runs/mesh.msh\n")
    try:
        r = client.get("/sandbox-file/webui_abc123def/work/solver.log")
        assert r.status_code == 200
        assert str(Path.home()) not in r.text and "~/runs/mesh.msh" in r.text
    finally:
        log.unlink()


def test_stop_leaves_a_process_that_was_already_in_the_folder(tmp_path):
    """A terminal someone opened in the run folder is not the run's work."""
    import subprocess, time
    from webui import proctree
    proc = subprocess.Popen(["bash", "-c", "sleep 60 & wait"], cwd=tmp_path, start_new_session=True)
    try:
        time.sleep(0.5)
        started_later = time.time() + 5
        assert proc.pid in proctree.run_processes(tmp_path)
        assert proc.pid not in proctree.run_processes(tmp_path, since=started_later)
        assert proctree.run_processes(tmp_path, since=time.time() - 60)
    finally:
        proc.kill(); proc.wait(timeout=10)


def test_a_table_preview_reads_only_what_it_shows(tmp_path):
    """A solver log can be hundreds of megabytes; the preview used to read all
    of it to show the first 999 rows."""
    import time
    from webui import viz
    big = config.SANDBOX_ROOT / "webui_abc123def" / "big.csv"
    big.parent.mkdir(parents=True, exist_ok=True)
    with big.open("w") as f:
        f.write("x,y\n")
        for i in range(400_000):          # about 5 MB
            f.write(f"{i},{i * 2}\n")
    try:
        t0 = time.monotonic()
        out = viz._csv(big)
        took = time.monotonic() - t0
        assert out["kind"] == "table" and out["truncated"] is True
        assert len(out["rows"]) == 999, len(out["rows"])
        assert took < 0.25, f"read the whole file: {took:.2f}s"
    finally:
        big.unlink()


def test_a_tab_separated_table_is_read_with_tabs(tmp_path):
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "t.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x\ty\n1\t2\n")
    try:
        out = viz._csv(p)
        assert out["header"] == ["x", "y"] and out["rows"] == [["1", "2"]]
    finally:
        p.unlink()


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


def test_the_first_prompt_reaches_the_server_without_the_tab(client, monkeypatch):
    """It used to be held in the browser until the socket said hello: closing
    the tab in that moment lost the message and left a run nobody could see."""
    sent = {}

    async def record(self, text, attachments=None):
        sent["run"], sent["text"], sent["files"] = self.sid, text, attachments

    monkeypatch.setattr(runs.Run, "prompt", record)
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        assert client.post(f"/api/sessions/{sid}/prompt", json={"text": " "}).status_code == 400
        assert client.post("/api/sessions/nosuchrun00/prompt", json={"text": "hi"}).status_code == 404
        r = client.post(f"/api/sessions/{sid}/prompt",
                        json={"text": "hello there", "attachments": ["part.msh"]})
        assert r.status_code == 200 and r.json()["sent"]
        assert sent == {"run": sid, "text": "hello there", "files": ["part.msh"]}
    finally:
        runs.RUNS.pop(sid, None)
        client.delete(f"/api/sessions/{sid}")


def test_the_prompt_is_on_disk_as_soon_as_it_is_sent(client):
    """Only every tenth event was written, so a restart during the first turn
    left a record with no prompt — and a run with no prompt is hidden."""
    import asyncio as aio
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        run = runs.get(sid)
        aio.run(run.emit({"type": "turn_start"}))
        aio.run(run.emit({"type": "user_msg", "text": "solve the plate"}))
        saved = json.loads((config.SESSION_DIR / f"{sid}.json").read_text())
        assert [e["type"] for e in saved["events"]] == ["turn_start", "user_msg"]
        assert saved["events"][1]["text"] == "solve the plate"
    finally:
        runs.RUNS.pop(sid, None)
        client.delete(f"/api/sessions/{sid}")


def test_deleting_refuses_a_made_up_id_before_touching_anything(client):
    for bad in ("../webui_other", "..%2Fx", "not-hex-id", ""):
        assert client.delete(f"/api/sessions/{bad}").status_code in (400, 404, 405)


def test_ending_one_step_leaves_a_step_that_started_earlier(tmp_path):
    """Both carry the run's marker; only the one being ended may be ended."""
    import subprocess, time
    from webui import proctree
    env = {**os.environ, "OPENPASO_CELL_WORKDIR": str(tmp_path.resolve())}
    earlier = subprocess.Popen(["bash", "-c", "sleep 60 & wait"], cwd="/tmp", env=env, start_new_session=True)
    time.sleep(0.6)
    step_started = time.time()
    time.sleep(0.6)
    later = subprocess.Popen(["bash", "-c", "sleep 60 & wait"], cwd="/tmp", env=env, start_new_session=True)
    try:
        time.sleep(0.6)
        claimed = proctree.run_processes(tmp_path, step_started, since_covers_marker=True)
        assert later.pid in claimed, claimed
        assert earlier.pid not in claimed, claimed
    finally:
        for p in (earlier, later):
            p.kill(); p.wait(timeout=10)


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


def _coupling_result(**fields):
    return "[{'type': 'text', 'text': '" + json.dumps(fields) + "'}]"


def test_the_legacy_coupled_solve_report_is_read_as_a_run_that_happened():
    """coupled_solve answers in prose, not JSON: a convergence report and
    openPASO's verification note. Read as JSON it became "failed", and a
    coupling that really ran was reported as having computed nothing."""
    from webui.outcome import classify_solver_result as c
    ran = ("Coupling converged in 12 iterations (residual 4.1e-07)\n\n"
           "[openPASO verification: NOT VERIFIED — openPASO's independent critic has not "
           "reviewed this setup...]")
    reviewed = ("Coupling converged in 9 iterations\n\n"
                "[openPASO verification: LEGACY coupled_solve — critic-reviewed. Trust is "
                "governed by the convergence report above...]")
    assert c(ran) == "unverified"
    assert c(reviewed) == "unverified", "prose carries no machine-readable verdict"
    assert c("Backend not found: fenicsx or dealii") == "failed"
    assert c("Unknown problem: thermoelastic. Available: ['fsi']") == "failed"


def test_stop_counts_the_processes_that_really_ended(monkeypatch):
    from webui import proctree
    alive = {41, 42}
    monkeypatch.setattr(proctree, "_alive", lambda pid: pid in alive)
    monkeypatch.setattr(proctree.os, "kill", lambda pid, sig: alive.discard(pid) if pid == 41 else None)
    # 41 dies, 42 will not: the count is what ended, not what was asked to end
    assert proctree._end([41, 42], grace=0.2) == 1


def test_a_correction_sent_late_becomes_a_message_the_record_keeps():
    """It is run as a follow-up turn. Without a user message in the log, the
    history a restarted run is given lost it, though the transcript showed it."""
    from webui.runs import _history
    events = [
        {"type": "turn_start"}, {"type": "user_msg", "text": "Solve it."},
        {"type": "done", "outcome": "no_result"},
        {"type": "turn_start"}, {"type": "user_msg", "text": "Use a finer mesh."},
        {"type": "agent_msg", "text": "Refining."}, {"type": "done", "outcome": "no_result"},
    ]
    h = _history(events)
    assert ("user", "Use a finer mesh.") in h


def test_a_coupling_reports_its_verdict_in_its_own_shape():
    """couple and couple_precice carry the verification gate's verdict with no
    `status` field at all. Reading only the run shape called every verified
    coupling a failure, so a real coupled result could never be finished."""
    from webui.outcome import SOLVER_TOOLS, classify_solver_result as c
    assert c(_coupling_result(converged=True, iterations=7, trustworthy_result=True)) == "verified"
    assert c(_coupling_result(converged=True, trustworthy_result=False)) == "unverified"
    assert c(_coupling_result(converged=False, error="coupling driver failed")) == "failed"
    # couple_levels answers one level at a time
    nested = _coupling_result(all_levels_converged=True, levels_run=2,
                              levels=[{"converged": True, "trustworthy_result": False},
                                      {"converged": True, "trustworthy_result": True}])
    assert c(nested) == "verified"
    assert {"couple", "couple_levels", "couple_precice", "verify_mesh_independence"} <= SOLVER_TOOLS


def test_the_run_list_is_scrubbed_like_every_other_answer(client):
    """A prompt can name the folder someone worked in."""
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        run = runs.get(sid)
        run.state["events"].append({"type": "user_msg", "seq": 1,
                                    "text": f"use the mesh in {Path.home()}/meshes/part.msh"})
        run.save()
        body = client.get("/api/sessions?all=true").text
        assert str(Path.home()) not in body and "~/meshes/part.msh" in body
    finally:
        client.delete(f"/api/sessions/{sid}")


def test_a_correction_reaches_a_critic_while_it_works_and_the_main_agent_after():
    """A critic can run for many minutes. A correction queued until it finished
    arrived after the thing it was meant to prevent."""
    from webui.runs import Run
    run = Run.__new__(Run)
    run.steers = [{"id": "st_1", "text": "Stop running MPI tests."}]
    run.emit = lambda e: asyncio.sleep(0)
    run.push_snapshot = lambda: asyncio.sleep(0)
    first = run._take_steers("sa_1111")
    assert [s["text"] for s in first] == ["Stop running MPI tests."]
    assert run._take_steers("sa_1111") == [], "the same sub-agent is not told twice"
    # a second critic in the same run is a second worker and hears it too
    assert [s["text"] for s in run._take_steers("sa_2222")] == ["Stop running MPI tests."]
    assert [s["text"] for s in run._take_steers("main")] == ["Stop running MPI tests."]
    assert run.steers == []


def test_a_large_field_file_is_described_without_being_parsed(tmp_path):
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "field.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    frames = [[[0.123456] * 100 for _ in range(100)] for _ in range(60)]
    p.write_text(json.dumps({"kind": "field_series", "field": "vorticity", "unit": "1/s",
                             "nx": 100, "ny": 100, "vmin": -1.0, "vmax": 1.0,
                             "times": [i * 0.1 for i in range(40)], "frames": frames}))
    try:
        assert p.stat().st_size > 4 * 1024 * 1024, p.stat().st_size
        out = viz._json(p)
        assert out["kind"] == "field_series"
        assert out["field"] == "vorticity" and out["nx"] == 100 and out["n_frames"] == 40
        assert out["vmin"] == -1.0 and out["vmax"] == 1.0
    finally:
        p.unlink()


def test_a_file_name_with_a_question_mark_still_downloads(client):
    run = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    run.mkdir(parents=True, exist_ok=True)
    odd = run / "sweep?a=1.txt"
    odd.write_text("value 3\n")
    try:
        from urllib.parse import quote
        url = "/sandbox-file/" + "/".join(quote(s, safe="") for s in
                                          ("webui_abc123def", "work", "sweep?a=1.txt"))
        r = client.get(url)
        assert r.status_code == 200 and "value 3" in r.text
    finally:
        odd.unlink()


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
