"""The `openpaso` command: the server, and two things around it.

    openpaso                     start the MCP server on stdio (what an AI app runs)
    openpaso doctor              what this install can do: Python, openPASO, every solver,
                                 the mesh generator -- no key, no network
    openpaso install <solver>    check first; install only what is missing, the way
                                 setup_backend would; say what to do by hand where
                                 no package exists (4C, deal.II, FEBio, SPARTA)

Both commands read the same registry the server uses (core.registry) and the
same setup routes the setup_backend tool uses (core.backend_setup.SETUP_ROUTES),
so what they report and what they run cannot drift from what the server does.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys

OK, NO, HM = "[ok]", "[!!]", "[--]"


def _first_line(text: object, limit: int = 88) -> str:
    line = str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _looks_like_a_version(text: object) -> bool:
    value = str(text or "").strip()
    return bool(re.match(r"^\d+\.\d", value)) and not re.search(r"fail|error|unknown|detect", value, re.I)


def _routes(name: str) -> list[dict]:
    try:
        from core.backend_setup import SETUP_ROUTES
    except Exception:                                    # noqa: BLE001
        return []
    return list(SETUP_ROUTES.get(name) or [])


def _route_command(route: dict) -> list[str] | None:
    """The first command of a setup route, with this interpreter where the route names one."""
    for command in route.get("commands") or []:
        parts = [str(p) for p in command]
        if parts and (parts[0].endswith("python") or parts[0].endswith("python3") or parts[0] == sys.executable):
            parts[0] = sys.executable
        return parts
    return None


def _hint(name: str) -> str:
    routes = _routes(name)
    if not routes:
        return ""
    cmd = _route_command(routes[0])
    if cmd:
        shown = list(cmd)
        if shown[0] == sys.executable:
            shown[0] = "python"
        return " ".join(shown)
    return str(routes[0].get("description", ""))


def _rows() -> list[dict]:
    from core.registry import list_backends, load_all_backends
    load_all_backends()
    return list_backends()


def doctor() -> int:
    """What this install can do. Returns 0 when openPASO imports and at least one solver works."""
    logging.disable(logging.INFO)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):                    # pragma: no cover
        pass
    print("openPASO -- checking this install. No API key and no network are used.")
    print()
    v = sys.version_info
    if v < (3, 10):
        print(f"{NO} Python {v[0]}.{v[1]} is too old. openPASO needs 3.10 to 3.13.")
        return 1
    print(f"{OK} Python {v[0]}.{v[1]}" + (" -- supported." if v < (3, 14) else " -- newer than anything tested; 3.10 to 3.13 are."))
    try:
        rows = _rows()
    except Exception as exc:                             # noqa: BLE001
        print(f"{NO} openPASO itself does not import: {_first_line(exc)}")
        return 1
    print(f"{OK} openPASO imports, and its tools are registered.")
    usable = [r for r in rows if r["status"] == "available"]
    print()
    print(f"Solvers openPASO can see on this machine -- {len(usable)} of {len(rows)}:")
    print()
    for row in sorted(rows, key=lambda r: (r["status"] != "available", r["name"])):
        mark = OK if row["status"] == "available" else NO
        version = f" {row['version']}" if _looks_like_a_version(row.get("version")) else ""
        print(f"  {mark} {row['display_name']}{version}")
        if row["status"] == "available":
            print(f"      {_first_line(row.get('message'))}")
        else:
            hint = _hint(row["name"])
            print(f"      not installed -- to get it:  openpaso install {row['name']}"
                  + (f"   (runs: {hint})" if hint else ""))
    print()
    try:
        import gmsh  # noqa: F401
        print(f"{OK} The mesh generator (Gmsh) is installed, so `generate_mesh` can build meshes.")
    except Exception:                                    # noqa: BLE001
        print(f"{HM} The mesh generator is not installed -- to get it:  pip install gmsh")
    print()
    if not usable:
        print(f"{NO} No solver works yet. Start with:  openpaso install skfem")
        return 1
    print(f"{OK} Ready: an AI app pointed at the `openpaso` command can use {len(usable)} solver(s).")
    return 0


def install(name: str, yes: bool = False) -> int:
    """Check first; install only what is missing, the way setup_backend would."""
    logging.disable(logging.INFO)
    rows = {r["name"]: r for r in _rows()}
    if name not in rows:
        print(f"{NO} Unknown solver '{name}'. Known: {', '.join(sorted(rows))}")
        return 2
    row = rows[name]
    if row["status"] == "available":
        print(f"{OK} {row['display_name']} is already installed and working"
              f"{' (' + row['version'] + ')' if _looks_like_a_version(row.get('version')) else ''}: "
              f"{_first_line(row.get('message'))}")
        return 0
    routes = _routes(name)
    if not routes:
        print(f"{NO} {row['display_name']} is not installed, and openPASO has no automatic route for it.")
        return 1
    route = routes[0]
    cmd = _route_command(route)
    print(f"{HM} {row['display_name']} is not installed: {_first_line(row.get('message'))}")
    print(f"    Route: {route.get('description', '')}"
          + (f" (about {route['typical_minutes']} min)" if route.get("typical_minutes") else ""))
    if not cmd:
        # no package to install: say exactly what to do by hand
        print("    No package exists for it. What to do:")
        for step in route.get("steps") or route.get("commands") or [route.get("description", "")]:
            print(f"      {step if isinstance(step, str) else ' '.join(str(p) for p in step)}")
        return 1
    shown = " ".join("python" if p == sys.executable else p for p in cmd)
    print(f"    Will run:  {shown}")
    if not yes and sys.stdin.isatty():
        answer = input("    Proceed? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            print("    Not installed.")
            return 1
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        print(f"{NO} The command failed (exit {done.returncode}). Nothing else was changed.")
        return done.returncode
    # Measure, do not assume -- in a FRESH process, because a backend's finder
    # may cache what it found before the install (measured: Kratos, installed a
    # moment earlier into this very interpreter, still read "not found" here
    # while `openpaso doctor` in a new process saw it).
    probe = ("import logging,sys; logging.disable(logging.CRITICAL)\n"
             "from core.registry import load_all_backends, get_backend\n"
             "load_all_backends(); st, msg = get_backend(%r).check_availability()\n"
             "print('AVAILABLE' if st.value == 'available' else 'MISSING', (msg or '')[:120])") % name
    done = subprocess.run([sys.executable, "-c", "import openpaso\n" + probe], capture_output=True, text=True)
    verdict = (done.stdout.strip().splitlines() or [""])[-1]
    if verdict.startswith("AVAILABLE"):
        print(f"{OK} {row['display_name']} is installed and working: {verdict[len('AVAILABLE '):]}")
        return 0
    print(f"{NO} The command finished, but openPASO still cannot use {row['display_name']}: "
          f"{verdict[len('MISSING '):] if verdict.startswith('MISSING') else _first_line(done.stderr)}")
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from server import main as serve   # the flat import; see openpaso/__init__.py
        serve()
        return 0
    ap = argparse.ArgumentParser(prog="openpaso", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="start the MCP server on stdio (the default when no command is given)")
    sub.add_parser("doctor", help="what this install can do: Python, openPASO, every solver, the mesh generator")
    p_in = sub.add_parser("install", help="check first; install a solver only if it is missing")
    p_in.add_argument("solver", help="skfem, ngsolve, kratos, dune, fenics, dealii, fourc, febio or sparta")
    p_in.add_argument("-y", "--yes", action="store_true", help="do not ask before running the install command")
    args = ap.parse_args(argv)
    if args.cmd in (None, "serve"):
        from server import main as serve
        serve()
        return 0
    if args.cmd == "doctor":
        return doctor()
    return install(args.solver, yes=args.yes)
