"""What can be selected, and whether it works: models and solvers.

Both lists used to say less than the truth. The model picker mixed local models
with no server running, Claude Code on the person's own login, and pay-per-token
OpenRouter models in one flat list with no way to tell them apart or to see
which would fail. The solver count came from the web server's own Python, which
is not the Python openPASO runs in (it lacked scikit-fem, so the page said 8 of
9 while the agent could use all nine), and the check ran inside the server's
event loop, freezing every other request for six seconds.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

from . import config

# ── models ───────────────────────────────────────────────────────────────
_PRICES: dict[str, tuple[float, float]] = {}
_PRICES_AT = 0.0


async def openrouter_prices(have_key: bool = True) -> dict[str, tuple[float, float]]:
    """USD per million tokens (input, output), read from OpenRouter itself and
    cached for an hour. Never invented: if the list cannot be fetched, no price
    is shown."""
    global _PRICES, _PRICES_AT
    if _PRICES and time.time() - _PRICES_AT < 3600:
        return _PRICES
    if not have_key:
        # without a key those models cannot be used, so their prices are of no
        # help — and asking for them would put a network call in the way of a
        # page (and of a test suite) that needs none
        return _PRICES
    try:
        import httpx
        async with httpx.AsyncClient(timeout=6) as c:
            r = await c.get(f"{config.OPENROUTER_URL}/models")
            r.raise_for_status()
            out = {}
            for m in r.json().get("data", []):
                pr = m.get("pricing") or {}
                try:
                    out[m["id"]] = (float(pr.get("prompt", 0)) * 1e6,
                                    float(pr.get("completion", 0)) * 1e6)
                except (TypeError, ValueError):
                    continue
            _PRICES, _PRICES_AT = out, time.time()
    except Exception:
        pass
    return _PRICES


def cost_of(model: str, tokens_in: int, tokens_out: int) -> float | None:
    p = _PRICES.get(model)
    if not p:
        return None
    return (tokens_in * p[0] + tokens_out * p[1]) / 1e6


async def _local_models(port: int) -> list[str] | None:
    """The model names a local server offers, or None when nothing serves here.

    The runner sends the id from the picker as the model name, so a server that
    answers on the port but offers something else fails on the first step. A
    port that responds is not the model being there."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=1.5) as c:
            r = await c.get(f"http://127.0.0.1:{port}/v1/models")
            if r.status_code != 200:
                return None
            data = r.json().get("data")
            if not isinstance(data, list):
                return None
            return [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")]
    except Exception:
        return None


def _claude_model_setting() -> str | None:
    try:
        return json.loads((Path.home() / ".claude" / "settings.json").read_text()).get("model")
    except Exception:
        return None


async def models() -> dict:
    groups = []
    key, key_source = config.openrouter_key_source()
    prices = await openrouter_prices(bool(key))
    hosted = []
    for mid, label in config.OPENROUTER_MODELS.items():
        pr = prices.get(mid)
        hosted.append({
            "id": mid, "label": label, "kind": "openrouter",
            "available": bool(key),
            "status": ("ready" if key else "no OpenRouter key"),
            "price_in": pr[0] if pr else None, "price_out": pr[1] if pr else None,
        })
    groups.append({
        "kind": "openrouter", "title": "Hosted on OpenRouter",
        "note": ("Billed per token to your OpenRouter key. Prompts and tool output go to "
                 "OpenRouter and to the model's provider."
                 + ("" if key else " No key found: set OPENROUTER_API_KEY, or put the line "
                    "OPENROUTER_API_KEY=... in a .env file in the openPASO folder, then reload.")),
        "key_source": key_source,
        "models": hosted,
    })

    from . import claude_code
    if claude_code.available():
        setting = _claude_model_setting()
        login = claude_code.signed_in()
        where = ("model from your Claude settings: " + setting) if setting else "model from your Claude settings"
        status = where if login else (where + " — no sign-in found on this machine; "
                                      "if the first step fails, run claude once in a terminal")
        groups.append({
            "kind": "claude-code", "title": "Claude Code",
            "note": ("Runs the claude command on this machine with your own Claude login. "
                     "Prompts and the files it reads go to Anthropic. It cannot stop to ask "
                     "for approval, so it only runs without asking, and its file and shell "
                     "tools can reach anything this account can, not only the run's folder."),
            "models": [{
                "id": config.CLAUDE_CODE_ID, "label": "Claude Code",
                "kind": "claude-code", "available": True,
                "status": status,
                "plan_mode": False,
            }],
        })

    local = []
    rows = [(k, m) for k, m in config.MODELS.items() if k != "mock"]
    served = await asyncio.gather(*[_local_models(m["port"]) for _, m in rows])
    for (mid, m), names in zip(rows, served):
        if names is None:
            status, ok = f"not running: start its model server on port {m['port']}", False
        elif mid in names:
            status, ok = f"running on port {m['port']}", True
        else:
            offered = ", ".join(names[:3]) or "nothing"
            status, ok = (f"port {m['port']} serves {offered}, not {mid}"), False
        local.append({"id": mid, "label": m["label"], "kind": "local",
                      "available": ok, "status": status})
    groups.append({
        "kind": "local", "title": "On this machine",
        "note": "Served by a local model server. Prompts stay on this machine; web search still goes out.",
        "models": local,
    })

    return {"groups": groups, "default": _default(groups)}


def _default(groups) -> str | None:
    for g in groups:
        for m in g["models"]:
            if m["available"]:
                return m["id"]
    return None


def model_label(mid: str) -> tuple[str, str]:
    """(label, kind) for a model id, without network access."""
    if mid in config.OPENROUTER_MODELS:
        return config.OPENROUTER_MODELS[mid], "openrouter"
    if mid == config.CLAUDE_CODE_ID:
        return "Claude Code", "claude-code"
    if mid in config.MODELS:
        if mid == "mock":
            return "Test model (fake, runs no solver)", "test"
        return config.MODELS[mid]["label"], "local"
    return mid, "unknown"


# ── solvers ──────────────────────────────────────────────────────────────
_SOLVERS: dict | None = None
_SOLVERS_AT = 0.0
_SOLVER_LOCK = asyncio.Lock()

# Also asked: whether the mesh generator is importable in the environment the
# openPASO server itself runs in. A solver can be installed and reachable while
# generate_mesh cannot work, and a person only found out after paying for a run.
_PROBE = r"""
import json, importlib.util
from core import registry
registry.load_all_backends()
print("@@JSON@@" + json.dumps({
    "backends": registry.list_backends(),
    "mesher": importlib.util.find_spec("gmsh") is not None,
}, default=str))
"""


def _clean_version(v) -> str | None:
    if not v:
        return None
    v = str(v).strip()
    low = v.lower()
    # FEBio reported "FATAL ERROR: Invalid command line option '--version'." as
    # its version; a failed probe is not a version
    if any(w in low for w in ("error", "failed", "invalid", "not found", "commands distilled")):
        return None
    return v[:60]


def _version_in(message) -> str | None:
    """The availability check often names the version it found ("scikit-fem
    12.0.1 at ...") even when the separate version probe fails."""
    m = re.search(r"(?<![\w.])v?(\d+\.\d+(?:\.\d+)?(?:[-.][A-Za-z0-9]+)?)(?![\w.])", str(message or ""))
    return m.group(1) if m else None


async def solvers(refresh: bool = False) -> dict:
    global _SOLVERS, _SOLVERS_AT
    async with _SOLVER_LOCK:
        if _SOLVERS is not None and not refresh and time.time() - _SOLVERS_AT < 600:
            return _SOLVERS
        spec = config.MCP_SERVERS["openpaso"]
        env = {**os.environ, **spec.get("env_extra", {})}
        try:
            proc = await asyncio.create_subprocess_exec(
                spec["command"], "-c", _PROBE, cwd=spec["cwd"], env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()                      # else it keeps running and holds its pipes
                await proc.wait()
                raise RuntimeError("the solver check took longer than 120 s")
            text = out.decode("utf-8", "replace")
            if "@@JSON@@" not in text:
                raise RuntimeError((err.decode("utf-8", "replace").strip().splitlines() or ["no output"])[-1])
            probe = json.loads(text.split("@@JSON@@", 1)[1].strip().splitlines()[0])
            rows = probe["backends"] if isinstance(probe, dict) else probe
            result = {"ok": True, "checked_at": time.time(),
                      "mesher": bool(probe.get("mesher")) if isinstance(probe, dict) else None,
                      # which interpreter answered: whether a mesh generator is
                      # importable is a fact about this one, not about the machine
                      "python": spec["command"],
                      "solvers": [{
                "name": r.get("display_name"), "status": r.get("status"),
                "version": _clean_version(r.get("version")) or _version_in(r.get("message")),
                "physics": r.get("physics_count"),
            } for r in rows]}
        except Exception as exc:
            result = {"ok": False, "checked_at": time.time(),
                      "error": f"{type(exc).__name__}: {exc}", "solvers": []}
        _SOLVERS, _SOLVERS_AT = result, time.time()
        return result
