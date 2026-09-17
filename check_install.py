#!/usr/bin/env python3
"""Did the install work? Answer without an API key and without an AI app.

WHY THIS FILE EXISTS. The README used to end its install section by saying
that if `python -m server` prints "Starting openPASO MCP server", that is
success. It is not. It proves only that Python can import the server. It says
nothing about the solver the reader was told to install two lines earlier, and
every other way of listing solvers goes through a model -- which costs money or
needs an AI app. A reader with neither finished the whole install and had no
way to learn whether anything worked. Measured on someone following the README
for the first time, that was the one thing that stopped them.

So: one command, no key, no network, no model.

    python check_install.py

It prints what openPASO can see on THIS machine and what to type to fix what is
missing, then exits 0 if at least one solver is usable and 1 if none is.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

OK, NO, HM = "✔", "✘", "—"
MIN_PYTHON, MAX_TESTED_PYTHON = (3, 10), (3, 13)


def _first_line(text: object, limit: int = 88) -> str:
    """Backends report failures as whole tracebacks. One line is the news.

    THE MIDDLE IS WHAT GETS DROPPED, NOT THE END. These lines end in the path
    the backend was found at, and that path is the whole point: it is how a
    reader tells openPASO's own virtual environment apart from some other
    Python on the same machine. Cutting the tail off throws away the evidence
    and keeps the prefix every line shares.
    """
    line = str(text or "").strip().splitlines()
    head = line[0] if line else ""
    if len(head) <= limit:
        return head
    keep = limit - 3
    return head[: keep // 3] + "…" + head[-(keep - keep // 3):]


def _looks_like_a_version(text: object) -> bool:
    """Only a real version number is worth printing beside a name.

    Backends answer this question badly in both directions: FEBio returns its
    own fatal error text, and deal.II returns "9.x (version detection failed)".
    Neither is a version, and showing either next to a tick reads as though the
    check itself is confused.
    """
    import re

    value = str(text or "").strip()
    if not re.match(r"^\d+\.\d", value):
        return False
    return not re.search(r"fail|error|unknown|detect", value, re.IGNORECASE)


def _install_hint(name: str) -> str:
    """The shortest published way to get this backend, from openPASO's own
    setup routes -- so this file cannot drift away from what setup_backend does."""
    try:
        from core.backend_setup import SETUP_ROUTES
    except Exception:                                    # noqa: BLE001
        return ""
    routes = SETUP_ROUTES.get(name) or []
    if not routes:
        return ""
    route = routes[0]
    for command in route.get("commands") or []:
        parts = [str(p) for p in command]
        if parts and parts[0] == sys.executable:
            parts[0] = "python"
        return " ".join(parts)
    return str(route.get("description", ""))


def check_python() -> bool:
    version = sys.version_info[:2]
    if version < MIN_PYTHON:
        print(f"{NO} Python {version[0]}.{version[1]} is too old. openPASO needs "
              f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.")
        return False
    if version > MAX_TESTED_PYTHON:
        print(f"{HM} Python {version[0]}.{version[1]} is newer than anything tested "
              f"(up to {MAX_TESTED_PYTHON[0]}.{MAX_TESTED_PYTHON[1]}). It may work.")
        return True
    print(f"{OK} Python {version[0]}.{version[1]} — supported.")
    return True


def check_openpaso() -> bool:
    try:
        import server                                    # noqa: F401
    except Exception as error:                           # noqa: BLE001
        print(f"{NO} openPASO itself does not import: {_first_line(error)}")
        print("   Your virtual environment is probably not active. Run:")
        print("     source .venv/bin/activate     (Windows: .venv\\Scripts\\activate)")
        print("   If that is not it, run `pip install -e .` again in this folder.")
        return False
    print(f"{OK} openPASO imports, and its tools are registered.")
    return True


def check_solvers() -> int:
    """Print every solver openPASO knows, and return how many are usable."""
    from core.registry import list_backends, load_all_backends

    load_all_backends()
    rows = list_backends()
    usable = [r for r in rows if r["status"] == "available"]

    print()
    print(f"Solvers openPASO can see on this machine — {len(usable)} of {len(rows)}:")
    print()
    for row in sorted(rows, key=lambda r: (r["status"] != "available", r["name"])):
        mark = OK if row["status"] == "available" else NO
        version = (f" {row['version']}" if _looks_like_a_version(row["version"]) else "")
        print(f"  {mark} {row['display_name']}{version}")
        if row["status"] == "available":
            print(f"      {_first_line(row['message'])}")
        else:
            hint = _install_hint(row["name"])
            print(f"      not installed" + (f" — to get it:  {hint}" if hint else ""))
    return len(usable)


def check_agent_packages() -> bool:
    """Option B (`python run_agent.py`) needs these; Option A (an AI app) does not."""
    try:
        import langgraph                                 # noqa: F401
        import openai                                    # noqa: F401
    except Exception:                                    # noqa: BLE001
        print(f"{HM} The agent packages are not installed. You need them only for "
              f"`python run_agent.py`.")
        print("   To get them:  pip install -r langgraph_eval/requirements-langgraph.txt")
        return False
    print(f"{OK} The agent packages are installed, so `python run_agent.py` can run.")
    return True


def check_key() -> bool:
    """Is a key present? THE KEY ITSELF IS NEVER PRINTED, not even partly."""
    env_file = REPO / ".env"
    in_shell = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    in_file = False
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                in_file = bool(value) and "your-key-here" not in value
    if in_shell or in_file:
        where = "your shell" if in_shell else str(env_file)
        print(f"{OK} An OpenRouter key is set (in {where}). Its value is not shown here.")
        return True
    if not env_file.is_file():
        print(f"{HM} No OpenRouter key yet, and no .env file. You need one ONLY for "
              f"`python run_agent.py`.")
        print("   To make one:  cp .env.example .env     then paste your key into it")
    else:
        print(f"{HM} No OpenRouter key yet. You need one ONLY for `python run_agent.py`.")
        print(f"   Paste it after OPENROUTER_API_KEY= in {env_file}")
    return False


def main() -> int:
    # Loading the backends logs a line per backend to stderr, and this file is
    # read by someone who has just been told the install is finished. Its own
    # output is the answer; the log is noise. Line buffering keeps the two
    # streams in the order they were written when the output is piped.
    import logging

    logging.disable(logging.INFO)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):                    # pragma: no cover
        pass

    print("openPASO — checking this install. No API key and no network are used.")
    print()
    healthy = check_python()
    healthy = check_openpaso() and healthy
    if not healthy:
        print()
        print(f"{NO} Fix the lines marked {NO} above, then run this again.")
        return 1

    usable = check_solvers()
    print()
    check_agent_packages()
    check_key()
    print()

    if usable == 0:
        print(f"{NO} No solver is usable, so openPASO has nothing to run a simulation "
              f"with.")
        print("   Install the easiest one and run this again:")
        print("     pip install scikit-fem")
        return 1
    print(f"{OK} Install is working: {usable} solver(s) ready. openPASO will use these "
          f"and tell a model the others are missing.")
    print("   Next: pick how you talk to it — README.md \"Quick start\", or https://hereon-institutems.github.io/openPASO/use/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
