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

# Each entry: id used in the API → human label, serving endpoint port,
# on-disk weights. Pointing at the *-Instruct checkpoints — the base
# Qwen2.5 weights that were on disk before are next-token completion
# models and CAN'T do tool calling, so the agent loop never closes.
MODELS = {
    "qwen2.5-7b":  {"label": "Qwen 2.5 7B Instruct",  "port": 8000},
    "qwen2.5-14b": {"label": "Qwen 2.5 14B Instruct", "port": 8001},
    "qwen2.5-32b": {"label": "Qwen 2.5 32B Instruct", "port": 8002},
    "mock":        {"label": "Mock LLM (no GPU)", "port": None},
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


def _env_files() -> list[tuple[Path, str]]:
    """Where a key may be written down: .env in the openPASO folder, then a file
    named by OPENPASO_ENV_FILE (for a key kept elsewhere on this machine)."""
    out = [(REPO / ".env", ".env in the openPASO folder")]
    extra = os.environ.get("OPENPASO_ENV_FILE")
    if extra:
        out.append((Path(extra).expanduser(), "the file named by OPENPASO_ENV_FILE"))
    return out


def openrouter_key_source() -> tuple[str | None, str | None]:
    """The key and, in words a person can check, where it was read from."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"], "the OPENROUTER_API_KEY environment variable"
    for candidate, words in _env_files():
        if not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue          # unreadable is "no key here", not a broken server
        for raw in lines:
            line = raw.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value, words
    return None, None


def openrouter_key() -> str | None:
    """The key from the environment, or from a .env file (see _env_files)."""
    return openrouter_key_source()[0]

# MCP servers selectable in the UI. The openPASO server is the main one;
# additional rows are placeholders for future plug-ins.
def _fourc_env() -> dict[str, str]:
    """Where 4C is, when it is where this machine says it is.

    An exported FOURC_BINARY is read by the backend as an explicit override and
    stops its own search, so exporting a guess at a path that does not exist
    told a person with 4C elsewhere that they do not have 4C. What is set by
    hand is passed on; a guess is offered only when it is actually there."""
    out: dict[str, str] = {}
    for name, guess, ok in (("FOURC_ROOT", Path.home() / "4C", Path.is_dir),
                            ("FOURC_BINARY", Path.home() / "4C/build/4C", Path.is_file)):
        given = os.environ.get(name)
        if given:
            out[name] = given
        elif ok(guess):       # a directory where a directory is meant, a file where a file is
            out[name] = str(guess)
    return out


def solver_library_path() -> str:
    """Where the solvers' shared libraries are, composed the way the agent does
    it: every required directory, then whatever was inherited, in order, with
    duplicates and directories that do not exist dropped. Defaulting to one of
    them left preCICE out of Claude Code runs only."""
    wanted = ["/opt/4C-dependencies/lib", "/opt/precice/lib"]
    seen, parts = set(), []
    for d in wanted + [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]:
        if d in seen:
            continue
        seen.add(d)
        if Path(d).is_dir():
            parts.append(d)
    return ":".join(parts)


def server_python() -> str:
    """The interpreter the openPASO MCP server runs on.

    The same order the agent uses: OPENPASO_PYTHON, then the repo's own
    environment. Hard-coding the second one here meant an install that sets the
    override reported no solvers and could not start a Claude Code run, while
    the other path worked."""
    for candidate in (os.environ.get("OPENPASO_PYTHON"), str(REPO / ".venv/bin/python")):
        if candidate and Path(candidate).is_file():
            return candidate
    # Only this install's own environment, or the one named in OPENPASO_PYTHON.
    # A third candidate pointing at another checkout would silently run someone
    # else's openPASO, with whatever solvers that one has.
    return str(REPO / ".venv/bin/python")      # so the failure names this path


MCP_SERVERS = {
    "openpaso": {
        "label": "openPASO — Open Agentic Simulation System",
        "command": server_python(),
        "args": ["-m", "server"],
        "cwd": str(REPO / "src"),
        "env_extra": {
            "PYTHONPATH": str(REPO / "src"),
            **_fourc_env(),
            "LD_LIBRARY_PATH": solver_library_path(),
        },
        "default_on": True,
    },
}

# "autonomous" was offered and read by nothing: it behaved exactly like accept.
# A control that does nothing is a small lie, so it is gone; old records that
# say "autonomous" load as accept.
MODES = ("plan", "accept")
DEFAULT_MODE = "accept"
MODE_INFO = {
    "plan":   {"label": "Ask before each step",
               "detail": ("openPASO stops before every step, including a critic's steps, and waits for you "
                          "to run or skip it.")},
    "accept": {"label": "Run without asking",
               "detail": ("openPASO runs tools, shell commands and solvers without stopping to ask. "
                          "Commands start in the run's folder but can read and write elsewhere on this machine. "
                          "You can still send corrections, end a step or stop the run.")},
}

DOCS_URL = "https://hereon-institutems.github.io/openPASO/"
def default_model() -> str | None:
    """A backend that can run without further set-up, or None.

    Never the fake model: it once produced a fabricated result that looked
    exactly like real work. Never a local model either: a fresh install has no
    model server, and naming one would only fail later. With no Claude Code and
    no OpenRouter key there is no model, and the interface says how to get one."""
    from . import claude_code
    if claude_code.available():
        return CLAUDE_CODE_ID
    if openrouter_key():
        return next(iter(OPENROUTER_MODELS))
    return None


DEFAULT_MODEL = "mock"   # kept for the test suite; the interface never offers it

# The fake model answers without a model and runs no solver. The test suites
# need it; nothing else may have it, so asking for it over the API is refused
# unless the server was started with this set.
ALLOW_TEST_MODEL = os.environ.get("OPENPASO_TEST_MODEL", "") not in ("", "0", "false", "no")

# How many runs may work at the same time on this machine. A solver can take
# every core it is given, so this is a guard against a slowdown nobody notices,
# not a licence limit.
MAX_RUNNING = int(os.environ.get("OPENPASO_MAX_RUNNING", "4"))
