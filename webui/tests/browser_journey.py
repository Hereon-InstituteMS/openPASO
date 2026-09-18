"""A person's journey through the web UI, driven in a real browser.

Opt-in (real model, a few cents) with the server running:

    .venv/bin/python webui/tests/browser_journey.py

Screenshots land in $OPENPASO_SHOTS (default /tmp/openpaso_shots); look at
them. A passing script is not a readable interface.
"""
import asyncio, os, re, sys, tempfile, time
from pathlib import Path
from playwright.async_api import async_playwright
B = os.environ.get("OPENPASO_UI", "http://127.0.0.1:8080")
S = os.environ.get("OPENPASO_SHOTS", "/tmp/openpaso_shots") + "/"
Path(S).mkdir(parents=True, exist_ok=True)
NOTES = Path(tempfile.gettempdir()) / "openpaso_notes.csv"
NOTES.write_text("x,u\n0,0\n0.5,0.07\n1,0\n")
LOG = []
def ok(name, cond, detail=""):
    LOG.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f"  ::  {detail}" if detail else ""), flush=True)

async def shot(pg, name):
    await pg.screenshot(path=S + name + ".png")

async def wait_text(pg, text, timeout):
    try:
        await pg.get_by_text(text, exact=False).first.wait_for(timeout=timeout * 1000); return True
    except Exception:
        return False

async def main():
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1920, "height": 1080})
        pg = await ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:300]))

        # ── home and pickers ────────────────────────────────────────────
        await pg.goto(B + "/", wait_until="networkidle"); await pg.wait_for_timeout(1500)
        model_btn = pg.get_by_role("button", name=re.compile("^Model"))
        ok("home: a model is preselected", "DeepSeek" in (await model_btn.inner_text()) or "Qwen" in (await model_btn.inner_text()) or "Claude" in (await model_btn.inner_text()), await model_btn.inner_text())
        await model_btn.click(); await pg.wait_for_timeout(400); await shot(pg, "b01_model_picker")
        ok("model picker: groups say where models run", await pg.get_by_text("Hosted on OpenRouter", exact=True).count() == 1 and await pg.get_by_text("On this machine", exact=True).count() == 1)
        box = await pg.get_by_text("Hosted on OpenRouter", exact=True).bounding_box()
        ok("model picker: fully on screen", box is not None and box["y"] >= 0, str(box))
        ok("model picker: unavailable local model says why", await pg.get_by_text(re.compile("not running: start its model server")).count() >= 1)
        await pg.keyboard.press("Escape"); await pg.wait_for_timeout(200)
        await pg.get_by_role("button", name=re.compile("^Steps")).click(); await pg.wait_for_timeout(300); await shot(pg, "b02_steps_picker")
        ok("steps picker: both modes explained", await pg.get_by_text("Ask before each step").count() >= 1 and await pg.get_by_text(re.compile("stops before every step")).count() == 1)
        await pg.keyboard.press("Escape")

        # ── a draft survives a visit to another page ─────────────────────
        await pg.get_by_test_id("prompt").fill("draft that must survive")
        await pg.get_by_role("link", name=re.compile("^Solvers")).first.click(); await pg.wait_for_timeout(800)
        await pg.go_back(); await pg.wait_for_timeout(800)
        ok("draft: kept after Solvers and Back", await pg.get_by_test_id("prompt").input_value() == "draft that must survive")
        ok("nav: Solvers not marked current on the start page",
           await pg.locator('[aria-current="page"]').filter(has_text="Solvers").count() == 0)
        await pg.get_by_test_id("prompt").fill("")

        # ── start a real run with an attachment ─────────────────────────
        prompt = ("Use run_bash to run exactly: sleep 25 && ls uploads . Then tell me in one sentence "
                  "which file I uploaded. Do not spawn a critic.")
        await pg.fill("#prompt", prompt)
        async with pg.expect_file_chooser() as fc:
            await pg.get_by_role("button", name="Attach files").click()
        await (await fc.value).set_files(str(NOTES))
        ok("attach: file shown before sending", await pg.get_by_text("notes.csv").count() >= 1)
        await shot(pg, "b03_ready_to_run")
        await pg.get_by_test_id("send").click()
        await pg.wait_for_url(re.compile(r"\?run="), timeout=30000)
        run_url = pg.url; rid = run_url.split("run=")[1]
        ok("run: navigated to the run's own address", "?run=" in run_url, run_url)
        ok("run: prompt shown as You", await wait_text(pg, "which file I uploaded", 30))
        ok("run: attachment shown with the prompt", await wait_text(pg, "uploads/openpaso_notes.csv", 30))
        # the command itself running, in the transcript (not the title in the runs panel)
        await pg.locator("text=/^running \\d/").first.wait_for(timeout=240000)
        ok("run: first step is running", True)
        await shot(pg, "b04_running")

        # ── correction while it works ───────────────────────────────────
        await pg.fill("#prompt", "Also say how many bytes the file has.")
        await pg.get_by_test_id("send").click()
        ok("correction: shown in the transcript", await wait_text(pg, "correction sent while it worked", 20))
        await shot(pg, "b05_correction")

        # ── back and forward do not kill the run ────────────────────────
        await pg.go_back(); await pg.wait_for_timeout(2500)
        ok("back: home page shown", await pg.get_by_role("heading", name="Describe the physics.").count() == 1)
        rail = pg.get_by_role("link", name=re.compile("which file I uploaded"))
        ok("back: run still listed as working in the rail", "Working" in (await rail.first.inner_text()), await rail.first.inner_text())
        await shot(pg, "b06_back_home_rail")
        await pg.go_forward(); await pg.wait_for_timeout(2500)
        ok("forward: back on the same run", rid in pg.url)

        # ── wait for the end ─────────────────────────────────────────────
        t0 = time.time()
        ended = False
        while time.time() - t0 < 1500:
            status = await pg.get_by_role("status").first.inner_text()
            if not re.search(r"Working|Waiting for you", status):
                ended = True; break
            await pg.wait_for_timeout(3000)
        status = await pg.get_by_role("status").first.inner_text()
        ok("end: run reached a final state", ended, status.replace("\n", " "))
        await pg.wait_for_timeout(1500); await shot(pg, "b07_finished")
        body = await pg.locator("main").inner_text()
        ok("end: a closing line says how the turn ended", bool(re.search(r"Finished in|Ended after|without a result|Failed", body)))
        ok("end: correction was delivered or sent as follow-up", bool(re.search(r"Delivered to openPASO|sent as a follow-up", body)), re.findall(r"(Waiting: openPASO reads it[^\n]*|Delivered to openPASO|sent as a follow-up|Not delivered[^\n]*)", body)[:3])
        ok("privacy: no home directory shown", not re.search(r"/home/[a-z]", body))

        await pg.get_by_role("radio", name="hidden").click(); await pg.wait_for_timeout(500)
        await shot(pg, "b08_reasoning_hidden")
        await pg.get_by_role("radio", name="shown").click()

        await pg.get_by_role("button", name="Files", exact=True).click(); await pg.wait_for_timeout(1500)
        await shot(pg, "b09_files")
        ok("files: drawer lists the run folder with uploads", await pg.get_by_role("button", name=re.compile("uploads")).count() >= 1)
        await pg.keyboard.press("Escape"); await pg.wait_for_timeout(300)

        # ── ask before each step ────────────────────────────────────────
        await pg.get_by_role("button", name=re.compile("New run")).click(); await pg.wait_for_timeout(1200)
        await pg.get_by_role("button", name=re.compile("^Steps")).click()
        await pg.get_by_role("button", name=re.compile("^Ask before each step")).click()
        await pg.fill("#prompt", "Use run_bash to print the Python version with: python3 --version . Then reply with the version.")
        await pg.get_by_test_id("send").click()
        await pg.wait_for_url(re.compile(r"\?run="), timeout=30000)
        has = await wait_text(pg, "Run this step", 240)
        ok("plan: the step waits with Run / Skip buttons", has)
        await shot(pg, "b10_plan_waiting")
        if has:
            status = await pg.get_by_role("status").first.inner_text()
            ok("plan: state says Waiting for you", "Waiting for you" in status, status.replace("\n", " "))
            n = 0
            while await pg.get_by_role("button", name="Run this step").count() and n < 6:
                await pg.get_by_role("button", name="Run this step").first.click(); n += 1
                await pg.wait_for_timeout(6000)
            ok("plan: approved step ran", await wait_text(pg, "Python 3", 180))

        # ── stop ────────────────────────────────────────────────────────
        await pg.get_by_role("button", name=re.compile("New run")).click(); await pg.wait_for_timeout(1200)
        await pg.get_by_role("button", name=re.compile("^Steps")).click()
        await pg.get_by_role("button", name=re.compile("^Run without asking")).click()
        await pg.fill("#prompt", "Use run_bash to run exactly: sleep 240 . Do nothing else.")
        await pg.get_by_test_id("send").click()
        await pg.wait_for_url(re.compile(r"\?run="), timeout=30000)
        await pg.locator("text=/^running \\d/").first.wait_for(timeout=240000)
        ok("stop: long command running", True)
        await pg.get_by_role("button", name="Stop", exact=True).click()
        ok("stop: asks to confirm", await wait_text(pg, "Stop this run and end everything it started?", 5))
        await shot(pg, "b11_stop_confirm")
        await pg.get_by_role("button", name="Stop", exact=True).click()
        ok("stop: run says it ended the process", await wait_text(pg, "process it had started was ended", 60) or await wait_text(pg, "processes it had started were ended", 5))
        await pg.wait_for_timeout(1500); await shot(pg, "b12_stopped")

        # ── the runs panel can be hidden ──────────────────────────────────
        before = (await pg.locator("main").bounding_box())["width"]
        await pg.get_by_role("button", name="Hide runs").click(); await pg.wait_for_timeout(500)
        after = (await pg.locator("main").bounding_box())["width"]
        ok("panel: hiding it gives the page the space", after > before + 200, f"{before:.0f} -> {after:.0f}")
        ok("panel: New run still reachable when hidden", await pg.get_by_role("button", name=re.compile("New run")).first.is_visible())
        await shot(pg, "b13_panel_hidden")
        await pg.reload(); await pg.wait_for_timeout(1500)
        ok("panel: stays hidden after reload", await pg.get_by_role("button", name="Show runs").count() == 1)
        await pg.get_by_role("button", name="Show runs").click(); await pg.wait_for_timeout(400)

        # ── delete runs from this machine ─────────────────────────────────
        pg.on("dialog", lambda d: asyncio.ensure_future(d.accept()))
        repo = Path(__file__).resolve().parents[2]
        mine = await pg.evaluate("""() => fetch('/api/sessions').then(r => r.json()).then(d => d.sessions
            .filter(s => /python3 --version|sleep 240|which file I uploaded/.test(s.prompt || '') && !s.running).map(s => s.id))""")
        ok("delete: test runs present", len(mine) >= 2, str(mine))
        if len(mine) >= 2:
            one, rest = mine[0], mine[1:]
            row = pg.locator(f'a[href="/?run={one}"]').locator("xpath=..")
            await row.hover()
            await row.get_by_role("button", name=re.compile("^Delete")).click()
            ask = pg.get_by_role("alertdialog", name="Delete runs")
            ok("delete one: asks on the page first", await ask.count() == 1 and "cannot be undone" in await ask.inner_text())
            await ask.get_by_role("button", name="Delete", exact=True).click()
            await pg.wait_for_timeout(2500)
            gone = await pg.locator(f'a[href="/?run={one}"]').count() == 0
            ok("delete one: removed from the list", gone)
            ok("delete one: folder removed from disk", not (repo / "eval_interactive" / f"webui_{one}").exists())
            await pg.get_by_role("button", name="Select", exact=True).click()
            for rid in rest:
                await pg.locator(f'a[href="/?run={rid}"]').locator("xpath=..").get_by_role("checkbox").check()
            await shot(pg, "b14_select_to_delete")
            await pg.get_by_role("button", name="Delete", exact=True).first.click()
            await pg.get_by_role("alertdialog", name="Delete runs").get_by_role("button", name="Delete", exact=True).click()
            await pg.wait_for_timeout(3000)
            ok("delete several: all removed from the list", all([await pg.locator(f'a[href="/?run={r}"]').count() == 0 for r in rest]))
            ok("delete several: folders removed from disk", not any((repo / "eval_interactive" / f"webui_{r}").exists() for r in rest))
            await shot(pg, "b15_after_delete")

        ok("no page errors", not errs, "; ".join(errs[:3]))
        await b.close()
    print(f"\n{sum(1 for _, c in LOG if c)}/{len(LOG)} passed")
    return 1 if any(not c for _, c in LOG) else 0

sys.exit(asyncio.run(main()))
