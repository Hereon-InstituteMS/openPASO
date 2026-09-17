"""Static configuration for the WebUI.

Only constants live here. Anything you'd actually edit at runtime (active
model, mode, MCP toggles) lives in a Session object on the server side.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SANDBOX_ROOT = REPO / "eval_interactive"
SESSION_DIR = REPO / "data" / "webui_sessions"
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

def _weights(name: str) -> str | None:
    """Local checkpoint folder, under OPENPASO_MODEL_DIR when the person set one."""
    root = os.environ.get("OPENPASO_MODEL_DIR")
    return str(Path(root).expanduser() / name) if root else None


# Each entry: id used in the API → human label, serving endpoint port,
# on-disk weights (under OPENPASO_MODEL_DIR). Pointing at the *-Instruct checkpoints — the base
# Qwen2.5 weights that were on disk before are next-token completion
# models and CAN'T do tool calling, so the agent loop never closes.
MODELS = {
    "qwen2.5-7b":  {"label": "Qwen 2.5 7B Instruct",  "port": 8000,
                    "weights": _weights("qwen2.5-7b-instruct")},
    "qwen2.5-14b": {"label": "Qwen 2.5 14B Instruct", "port": 8001,
                    "weights": _weights("qwen2.5-14b-instruct")},
    "qwen2.5-32b": {"label": "Qwen 2.5 32B Instruct", "port": 8002,
                    "weights": _weights("qwen2.5-32b-instruct")},
    "mock":        {"label": "Mock LLM (no GPU)", "port": None,
                    "weights": None},
}

# Models reached over an API instead of a local GPU. Most people have a key and
# no GPU, so without these the interface was unusable to them even though
# run_agent.py could already do it. The key is read from the same places
# run_agent.py reads it, so one .env serves both.
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Claude Code, driven headless. No key and no GPU: it uses the subscription
# the person already has, which for most people is the easiest path there is.
CLAUDE_CODE_ID = "claude-code"

OPENROUTER_MODELS = {
    "deepseek/deepseek-v4.1-flash": "DeepSeek v4.1 Flash",
    "qwen/qwen3.5-27b":             "Qwen 3.5 27B",
    "qwen/qwen3.5-122b-a10b":       "Qwen 3.5 122B",
}


def openrouter_key() -> str | None:
    """The key from the environment, or from a .env next to the repo."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for candidate in (REPO / ".env",):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    return None

# MCP servers selectable in the UI. The openPASO server is the main one;
# additional rows are placeholders for future plug-ins.
MCP_SERVERS = {
    "openpaso": {
        "label": "openPASO — Open Agentic Simulation System",
        "command": str(REPO / ".venv/bin/python"),
        "args": ["-m", "server"],
        "cwd": str(REPO / "src"),
        "env_extra": {
            "PYTHONPATH": str(REPO / "src"),
            "FOURC_ROOT": os.environ.get(
                "FOURC_ROOT", str(Path.home() / "4C")),
            "FOURC_BINARY": os.environ.get(
                "FOURC_BINARY", str(Path.home() / "4C/build/4C")),
            "LD_LIBRARY_PATH": os.environ.get(
                "LD_LIBRARY_PATH", "/opt/4C-dependencies/lib"),
        },
        "default_on": True,
    },
}

MODES = ("plan", "accept", "autonomous")
DEFAULT_MODE = "accept"
def default_model() -> str:
    """The first backend that can actually run, never the fake one.

    The default used to be the mock, so the first thing a newcomer did was
    produce a fabricated result that looked exactly like real work."""
    from . import claude_code
    if claude_code.available():
        return CLAUDE_CODE_ID
    if openrouter_key():
        return next(iter(OPENROUTER_MODELS))
    return next(k for k in MODELS if k != "mock")


DEFAULT_MODEL = "mock"   # kept for the test suite; the interface never offers it
