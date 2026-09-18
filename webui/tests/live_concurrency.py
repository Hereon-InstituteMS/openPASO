"""Several runs opening openPASO at once, while another run's tabs come and go.

Opt-in (real model, a few cents), server running:
    .venv-lg/bin/python webui/tests/live_concurrency.py

Guards the bug where closing one run's tool-server connection from a different
task broke a second run's connection as it was being opened.
"""
import asyncio, json, os, urllib.request
import websockets

BASE = os.environ.get("OPENPASO_UI", "http://127.0.0.1:8080")
WS = BASE.replace("http", "ws", 1) + "/ws/"
MODEL = "deepseek/deepseek-v4.1-flash"

def new_run():
    req = urllib.request.Request(BASE + "/api/sessions", method="POST",
        data=json.dumps({"model": MODEL, "mode": "accept"}).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["id"]

async def one(i):
    sid = new_run()
    async with websockets.connect(WS + sid, max_size=2**26) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "prompt", "text": f"Use run_bash to run exactly: echo run-{i}-ok . Then reply OK."}))
        errors, outcome = [], None
        while True:
            e = json.loads(await asyncio.wait_for(ws.recv(), 300))
            if e.get("type") == "error": errors.append(e.get("message"))
            if e.get("type") == "done": outcome = e.get("outcome"); break
        return sid, outcome, errors

async def churn(sid, stop):
    n = 0
    while not stop.is_set():
        async with websockets.connect(WS + sid, max_size=2**26) as ws:
            await ws.recv()
        n += 1
        await asyncio.sleep(0.2)
    return n

async def main():
    warm = new_run()
    async with websockets.connect(WS + warm, max_size=2**26) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "prompt", "text": "Use run_bash to run exactly: echo warm . Then reply OK."}))
        while json.loads(await ws.recv()).get("type") != "done":
            pass
    stop = asyncio.Event()
    churner = asyncio.create_task(churn(warm, stop))
    results = await asyncio.gather(*[one(i) for i in range(3)])
    stop.set(); n = await churner
    bad = 0
    for sid, outcome, errors in results:
        ok = outcome == "no_result" and not errors
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"{sid} outcome={outcome} errors={errors}")
    print(f"tab connect/disconnect cycles on another run meanwhile: {n}")
    for sid in [warm] + [r[0] for r in results]:
        urllib.request.urlopen(urllib.request.Request(f"{BASE}/api/sessions/{sid}", method="DELETE"), timeout=30)
    print("ALL PASS" if not bad else f"{bad} FAILED")

asyncio.run(main())
