"""Find and end every process a run started.

Stopping a run used to cancel the coroutine that was waiting on the work, and
nothing else. `run_bash` waits on its command in a worker thread, so the command
and every solver it launched kept running; a solver started through openPASO's
own tools lives under the openPASO server process, which was left alone too. The
screen said "You stopped this run" while the computation carried on.

A run's processes are recognised by what they share, read from /proc so no
extra dependency is needed:

* the openPASO server for the run, and everything it starts, carry
  ``OPENPASO_CELL_WORKDIR=<the run's work directory>`` in their environment;
* shell commands the agent runs start inside the run's work directory;
* Claude Code is launched with the same environment marker.

Everything below a matching process is included, because a child that changed
directory or dropped the variable is still that run's work.
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

_UID = os.getuid()


def _pids() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def _owned(pid: int) -> bool:
    try:
        return os.stat(f"/proc/{pid}").st_uid == _UID
    except OSError:
        return False


def _ppid(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # the command name is in parentheses and may itself contain spaces
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _environ_has(pid: int, needle: bytes) -> bool:
    try:
        return needle in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False


def _cwd_under(pid: int, root: str) -> bool:
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return cwd == root or cwd.startswith(root + os.sep)


def _started_at(pid: int) -> float | None:
    """When a process started, in epoch seconds."""
    try:
        boot = _boot_time()
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _boot_time() -> float:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    return 0.0


def run_processes(workdir: Path, since: float | None = None) -> list[int]:
    """Every live process belonging to the run whose work directory is given.

    ``since`` is when the run began. A process that merely sits in the folder
    (a terminal someone opened there) is only claimed when it started after the
    run did; anything carrying the run's marker is the run's work whenever it
    started."""
    root = str(Path(workdir).resolve())
    marker = f"OPENPASO_CELL_WORKDIR={root}".encode()
    me = os.getpid()
    pids = [p for p in _pids() if p != me and _owned(p)]
    parent = {p: _ppid(p) for p in pids}

    seeds = {p for p in pids if _environ_has(p, marker)
             or (_cwd_under(p, root)
                 and (since is None or (_started_at(p) or 0) >= since - 1))}
    # the web server itself may have been started from inside a run folder;
    # never include it or its ancestors
    seeds.discard(me)

    found = set(seeds)
    changed = True
    while changed:
        changed = False
        for p, pp in parent.items():
            if p not in found and pp in found:
                found.add(p)
                changed = True
    return sorted(found)


def _is_openpaso_server(pid: int) -> bool:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return any(a == b"-m" and i + 1 < len(argv) and argv[i + 1] == b"server"
               for i, a in enumerate(argv))


def end_step_processes(workdir: Path, grace: float = 3.0, since: float | None = None) -> int:
    """End what the run's current work started, but not the openPASO server the
    run is connected to: the run continues after one step is ended."""
    return _end([p for p in run_processes(workdir, since) if not _is_openpaso_server(p)], grace)


def end_run_processes(workdir: Path, grace: float = 3.0, since: float | None = None) -> int:
    """TERM, wait, then KILL every process of the run. Returns how many ended."""
    return _end(run_processes(workdir, since), grace)


def _end(targets: list[int], grace: float) -> int:
    """How many of these processes are gone afterwards — not how many were
    asked to go. Reporting the request as the result told the user that every
    process had been ended while some were still alive."""
    if not targets:
        return 0
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in targets:
            try:
                os.kill(p, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not any(_alive(p) for p in targets):
                return len(targets)
            time.sleep(0.1)
    return sum(1 for p in targets if not _alive(p))


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return False
