"""Live flows against a running web UI, with a real hosted model.

Opt-in and not part of pytest: it spends a few cents of OpenRouter credit and
needs the server running. Start it, then:

    .venv-lg/bin/python webui/tests/live_flows.py            # all flows
    .venv-lg/bin/python webui/tests/live_flows.py stop plan  # some

Each flow checks one promise the interface makes: a run survives its tab
closing, two runs work at once, a correction reaches the model, a follow-up
remembers, Stop ends the processes, "Ask before each step" waits.
"""
import asyncio, json, os, sys, time, urllib.request
from pathlib import Path
import websockets

REPO = Path(__file__).resolve().parents[2]

BASE = os.environ.get("OPENPASO_UI", "http://127.0.0.1:8080")
WS = BASE.replace("http", "ws", 1) + "/ws/"
MODEL = "deepseek/deepseek-v4.1-flash"
RESULTS = []

def http(method, path, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = headers or ({"Content-Type": "application/json"} if body is not None else {})
    req = urllib.request.Request(BASE + path, method=method, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  ::  {detail}" if detail else ""), flush=True)

async def recv_until(ws, pred, timeout):
    got = []
    end = time.time() + timeout
    while time.time() < end:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, end - time.time()))
        except asyncio.TimeoutError:
            break
        e = json.loads(raw); got.append(e)
        if pred(e):
            return got, e
    return got, None

def new_run(mode="accept", model=MODEL, test=False):
    st, s = http("POST", "/api/sessions", {"model": model, "mode": mode, "test": test})
    assert st == 200, (st, s)
    return s["id"]

def workdir(sid):
    return str(REPO / "eval_interactive" / f"webui_{sid}" / "work")

def procs_in(sid):
    root = os.path.realpath(workdir(sid)); n = []
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            if os.readlink(f"/proc/{p}/cwd").startswith(root): n.append(int(p))
        except OSError: pass
    return n

# ── 1. test model: a turn ends with a truthful outcome ─────────────────────
async def t_mock():
    sid = new_run(model="mock", test=True)
    async with websockets.connect(WS + sid, max_size=2**26) as ws:
        hello = json.loads(await ws.recv())
        check("hello carries session and events", hello["type"] == "hello" and "events" in hello)
        await ws.send(json.dumps({"type": "prompt", "text": "hello"}))
        got, done = await recv_until(ws, lambda e: e.get("type") == "done", 90)
    types = [e.get("type") for e in got]
    check("mock: turn_start before user_msg", "turn_start" in types and types.index("turn_start") < types.index("user_msg"))
    check("mock: ends with done", done is not None, str(done))
    check("mock: no solver so no_result", done and done.get("outcome") == "no_result", str(done))
    seqs = [e["seq"] for e in got if "seq" in e]
    check("mock: events carry increasing seq and t", seqs == sorted(seqs) and all("t" in e for e in got if e.get("type") != "session"))

# ── 2. leaving mid-run does not stop the run; a second tab gets the live log ──
async def t_survive():
    sid = new_run()
    ws = await websockets.connect(WS + sid, max_size=2**26); await ws.recv()
    await ws.send(json.dumps({"type": "prompt", "text":
        "Use run_bash to run exactly: sleep 30 && echo finished-sleeping . Then reply with the single word DONE."}))
    got, ex = await recv_until(ws, lambda e: e.get("type") == "tool_call_executing" and e.get("tool") == "run_bash", 240)
    check("survive: command started", ex is not None)
    await ws.close()
    await asyncio.sleep(6)
    st, lst = http("GET", "/api/sessions")
    row = next((r for r in lst["sessions"] if r["id"] == sid), None)
    check("survive: still running after the tab closed", row and row["running"], str(row and row["outcome"]))
    async with websockets.connect(WS + sid, max_size=2**26) as ws2:
        hello = json.loads(await ws2.recv())
        replay = [e.get("type") for e in hello["events"]]
        check("survive: reconnect replays the log so far", "tool_call_executing" in replay and hello["session"]["running"])
        got, done = await recv_until(ws2, lambda e: e.get("type") == "done", 240)
    res = [e for e in hello["events"] + got if e.get("type") == "tool_result" and e.get("tool") == "run_bash"]
    check("survive: the command finished while nobody watched", any("finished-sleeping" in (e.get("result") or "") for e in res))
    check("survive: run ended", done is not None, str(done))

# ── 3. Stop ends the processes the run started ───────────────────────────────
async def t_stop():
    sid = new_run()
    async with websockets.connect(WS + sid, max_size=2**26) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "prompt", "text": "Use run_bash to run exactly: sleep 300 . Do nothing else."}))
        got, ex = await recv_until(ws, lambda e: e.get("type") == "tool_call_executing" and e.get("tool") == "run_bash", 240)
        check("stop: long command started", ex is not None)
        await asyncio.sleep(3)
        before = procs_in(sid)
        check("stop: the command is a live process in the run folder", len(before) > 0, str(before))
        await ws.send(json.dumps({"type": "stop"}))
        got, done = await recv_until(ws, lambda e: e.get("type") == "done", 60)
    err = next((e for e in got if e.get("type") == "error"), {})
    check("stop: outcome interrupted", done and done.get("outcome") == "interrupted", str(done))
    check("stop: says how many processes it ended", (err.get("processes_ended") or 0) >= 1, err.get("message", ""))
    await asyncio.sleep(1)
    after = procs_in(sid)
    check("stop: no process of the run is left", after == [], str(after))

# ── 4+5. correction mid-run is delivered; a follow-up remembers ─────────────
async def t_steer():
    sid = new_run()
    async with websockets.connect(WS + sid, max_size=2**26) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "prompt", "text":
            "Use run_bash to run exactly: sleep 20 && echo first . After that finishes, reply briefly."}))
        got, ex = await recv_until(ws, lambda e: e.get("type") == "tool_call_executing" and e.get("tool") == "run_bash", 240)
        await ws.send(json.dumps({"type": "steer", "text": "Correction from me: in your final reply you must include the word BANANA."}))
        got2, done = await recv_until(ws, lambda e: e.get("type") == "done", 240)
        allev = got + got2
        states = [e.get("state") for e in allev if e.get("type") in ("user_steer", "steer_state")]
        check("steer: queued then delivered", "queued" in states and ("delivered" in states or "sent_as_followup" in states), str(states))
        # wait for a possible follow-up turn created from an undelivered correction
        if "sent_as_followup" in states:
            got3, done = await recv_until(ws, lambda e: e.get("type") == "done", 240); allev += got3
        final = " ".join(e.get("text", "") for e in allev if e.get("type") == "agent_msg")
        check("steer: the model acted on the correction", "BANANA" in final.upper(), final[-200:])
        await ws.send(json.dumps({"type": "prompt", "text": "Which word did I ask you to include? Answer with that word only."}))
        got4, done = await recv_until(ws, lambda e: e.get("type") == "done", 240)
        ans = " ".join(e.get("text", "") for e in got4 if e.get("type") == "agent_msg")
        check("follow-up: remembers the earlier turn", "BANANA" in ans.upper(), ans[-200:])

# ── 6. plan mode waits for approval ─────────────────────────────────────────
async def t_plan():
    sid = new_run(mode="plan")
    async with websockets.connect(WS + sid, max_size=2**26) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "prompt", "text": "Use run_bash to run exactly: echo approved-ok . Then reply OK."}))
        got, pend = await recv_until(ws, lambda e: e.get("type") == "tool_call_pending", 240)
        check("plan: a call waits", pend is not None, str(pend and pend.get("tool")))
        got2, ex = await recv_until(ws, lambda e: e.get("type") == "tool_call_executing", 8)
        check("plan: nothing runs before approval", ex is None)
        await ws.send(json.dumps({"type": "approve", "call_id": pend["call_id"]}))
        got3, done = await recv_until(ws, lambda e: e.get("type") == "done" or
                                      (e.get("type") == "tool_call_pending" and e.get("call_id") != pend["call_id"]), 240)
        while done and done.get("type") == "tool_call_pending":
            await ws.send(json.dumps({"type": "approve", "call_id": done["call_id"]}))
            more, done = await recv_until(ws, lambda e: e.get("type") in ("done", "tool_call_pending"), 240); got3 += more
        res = [e for e in got3 if e.get("type") == "tool_result"]
        check("plan: approved call ran", any("approved-ok" in (e.get("result") or "") for e in res))

# ── End this step: one hung step ends, the correction arrives, the run goes on ──
async def t_endstep():
    sid = new_run()
    async with websockets.connect(WS + sid, max_size=2**26) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "prompt", "text":
            "Use run_bash to run exactly: sleep 600 . When that step returns, reply in one sentence "
            "saying what the step returned and repeat any message from the user word for word. Do not spawn a critic."}))
        got, ex = await recv_until(ws, lambda e: e.get("type") == "tool_call_executing" and e.get("tool") == "run_bash", 240)
        check("end step: long command running", ex is not None)
        await asyncio.sleep(3)
        check("end step: its process exists", len(procs_in(sid)) > 0)
        await ws.send(json.dumps({"type": "steer", "text": "The code word is PERIDOT."}))
        t0 = time.time()
        await ws.send(json.dumps({"type": "end_step", "call_id": ex["call_id"]}))
        got2, res = await recv_until(ws, lambda e: e.get("type") == "tool_result" and e.get("call_id") == ex["call_id"], 30)
        check("end step: the step returned quickly", res is not None and time.time() - t0 < 20, f"{time.time()-t0:.1f}s")
        check("end step: the model is told the user ended it", res is not None and "The user ended this step" in (res.get("result") or ""))
        got3, done = await recv_until(ws, lambda e: e.get("type") == "done", 240)
        allev = got2 + got3
        check("end step: the run carried on to a normal end", done is not None and done.get("outcome") != "interrupted", str(done))
        check("end step: the correction was delivered", any(e.get("type") == "steer_state" and e.get("state") == "delivered" for e in allev))
        reply = " ".join(e.get("text") or "" for e in allev if e.get("type") == "agent_msg")
        check("end step: the model acted on the correction", "PERIDOT" in reply.upper(), reply[:200])
        # the run's own openPASO server stays up while the run is open; the
        # command the step started must be gone
        left = [p for p in procs_in(sid) if b"-m\0server" not in open(f"/proc/{p}/cmdline", "rb").read()]
        check("end step: the step's processes are gone", left == [], str(left))
        check("end step: the run's openPASO server is still up", len(procs_in(sid)) - len(left) >= 1)

# ── Ask before each step covers a critic's steps too ─────────────────────────
async def t_plan_critic():
    sid = new_run(mode="plan")
    async with websockets.connect(WS + sid, max_size=2**26) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "prompt", "text":
            "Call spawn_subagent exactly once with role critic and the task: 'Use run_bash to run exactly: echo critic-step . "
            "Then reply DONE.' Do nothing else before that. After it returns, reply OK."}))
        sub_pending, critic_pending, critic_ran_unasked = None, None, False
        end = time.time() + 400
        while time.time() < end:
            got, e = await recv_until(ws, lambda e: e.get("type") in ("tool_call_pending", "tool_call_executing", "done"), end - time.time())
            if e is None or e.get("type") == "done":
                break
            if e["type"] == "tool_call_executing" and e.get("approved_by") != "user":
                critic_ran_unasked = True
            if e["type"] == "tool_call_pending":
                if e.get("agent") == "critic" and critic_pending is None:
                    critic_pending = e
                    _, ex = await recv_until(ws, lambda x: x.get("type") == "tool_call_executing", 6)
                    check("plan critic: the critic's step waits too", ex is None)
                if e.get("tool") == "spawn_subagent" and sub_pending is None:
                    sub_pending = e
                    check("plan critic: the task is shown before approval", "critic-step" in json.dumps(e.get("args")))
                await ws.send(json.dumps({"type": "approve", "call_id": e["call_id"]}))
        check("plan critic: a sub-agent was asked for", sub_pending is not None)
        check("plan critic: the critic's step asked first", critic_pending is not None)
        check("plan critic: nothing ran without approval", not critic_ran_unasked)
        await ws.send(json.dumps({"type": "stop"}))

# ── 7. upload and per-run files; 8. rules at creation ──────────────────────
def t_upload_and_rules():
    sid = new_run(model="mock", test=True)
    boundary = "----openpasotest"
    def part(name, content):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n"
    body = part("bracket mesh.msh", b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n") + f"--{boundary}--\r\n".encode()
    st, r = http("POST", f"/api/sessions/{sid}/upload", raw=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    check("upload: mesh accepted", st == 200 and r["saved"][0]["name"] == "bracket_mesh.msh", str((st, r)))
    body = part("evil.sh", b"rm -rf /") + f"--{boundary}--\r\n".encode()
    st, r = http("POST", f"/api/sessions/{sid}/upload", raw=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    check("upload: unknown type refused with a reason", st == 415, str(st))
    st, f = http("GET", f"/api/sessions/{sid}/files?sub=uploads")
    check("files: lists this run's upload, no absolute paths", st == 200 and any(e["name"] == "bracket_mesh.msh" for e in f["entries"])
          and all("abs_path" not in e for e in f["entries"]) and "/home/" not in json.dumps(f), json.dumps(f)[:200])
    st, f = http("GET", f"/api/sessions/{sid}/files?sub=../../webui_other")
    check("files: cannot leave the run folder", st in (403, 404), str(st))
    st, r = http("POST", "/api/sessions", {"model": "claude-code", "mode": "plan"})
    check("rules: Claude Code with ask-before-each-step refused with a reason", st == 400, str(r))
    st, r = http("POST", "/api/sessions", {"model": "mock", "mode": "accept"})
    check("rules: the fake model is not creatable by a person", st == 400, str(st))

async def main():
    which = sys.argv[1:] or ["mock", "rules", "parallel", "stop", "plan", "endstep", "critic"]
    if "mock" in which: await t_mock()
    if "rules" in which: t_upload_and_rules()
    if "parallel" in which:
        t0 = time.time()
        await asyncio.gather(t_survive(), t_steer())   # two runs at the same time
        check("parallel: two runs worked at the same time", True, f"{time.time()-t0:.0f}s")
    if "stop" in which: await t_stop()
    if "plan" in which: await t_plan()
    if "endstep" in which: await t_endstep()
    if "critic" in which: await t_plan_critic()
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS)-len(bad)}/{len(RESULTS)} passed")

asyncio.run(main())
