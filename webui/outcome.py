"""How a run ended, decided once.

The server, the run list and the saved record all read this. The interface
mirrors the same rule for records written before a turn carried its outcome.

The rule that matters: ``completed`` is a claim about the physics. It is not
"no exception escaped" and it is not "a solver tool was called". A solver tool
reports its own verdict, and only a result it reports as completed and
trustworthy is a finished result. A call that returned "Unknown solver: abaqus"
computed nothing; a result openPASO's attestation marks as not trustworthy ran
but is not a result anyone should report.
"""
from __future__ import annotations

import json
import re

# Tools whose return can be a solver result. run_bash is deliberately absent:
# a shell command that exits zero proves a shell command exited zero.
SOLVER_TOOLS = frozenset({
    "run_simulation", "run_with_generator", "coupled_solve",
    "couple", "couple_precice",
})

RUNNING = "running"
COMPLETED = "completed"        # a solver ran and openPASO verified its result
UNVERIFIED = "unverified"      # a solver ran, openPASO did not verify the result
NO_RESULT = "no_result"        # ended cleanly, no solver result
FAILED = "failed"
INTERRUPTED = "interrupted"


def _payload(raw: str) -> dict | None:
    """The JSON object inside a tool result, which MCP wraps as a repr of text
    blocks. None when the result is not a JSON report (e.g. an error string)."""
    text = raw or ""
    m = re.search(r"'text':\s*(['\"])([\s\S]*?)\1\s*[,}]", text)
    if m:
        text = m.group(2).encode("utf-8").decode("unicode_escape", "ignore")
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    # fall back to reading the two fields that matter
    st = re.search(r'"status"\s*:\s*"([A-Za-z_]+)"', text)
    tr = re.search(r'"trustworthy_result"\s*:\s*(true|false)', text)
    if not st:
        return None
    out = {"status": st.group(1)}
    if tr:
        out["trustworthy_result"] = tr.group(1) == "true"
    return out


def classify_solver_result(raw: str) -> str:
    """'verified', 'unverified' or 'failed' for one solver tool result."""
    p = _payload(raw)
    if not p:
        return "failed"
    status = str(p.get("status", "")).lower()
    if not status.startswith("completed"):
        return "failed"
    if status == "completed" and p.get("trustworthy_result") is True:
        return "verified"
    return "unverified"


def solver_verdict(events) -> str | None:
    """The best solver evidence in these events: 'verified', 'unverified',
    'failed' (a solver tool was called and computed nothing), or None."""
    seen = None
    for e in events:
        if e.get("type") == "tool_result" and e.get("tool") in SOLVER_TOOLS:
            v = classify_solver_result(e.get("result") or "")
            if v == "verified":
                return v
            if v == "unverified" or seen is None:
                seen = v
    return seen


def solver_ran(events) -> bool:
    """True when some solver tool produced a result, verified or not."""
    return solver_verdict(events) in ("verified", "unverified")


def clean_outcome(events) -> str:
    """The outcome of a turn that ended without an error."""
    v = solver_verdict(events)
    return COMPLETED if v == "verified" else UNVERIFIED if v == "unverified" else NO_RESULT


def _turns(events):
    """Split the log into turns. A turn starts at `turn_start` (or, in records
    written before that event existed, at `user_msg`)."""
    turns, cur, has_marker = [], [], any(e.get("type") == "turn_start" for e in events)
    starter = "turn_start" if has_marker else "user_msg"
    for e in events:
        if e.get("type") == starter and cur:
            turns.append(cur)
            cur = []
        cur.append(e)
    if cur:
        turns.append(cur)
    return turns


def fold(events, *, live_default: str = RUNNING) -> str:
    """How the latest turn of a run ended, from its event log.

    Each turn is judged on its own. An error decides the turn, and an error that
    carries no outcome of its own is a failure. A clean end is judged by the
    solver evidence, never by what the log's `done` event claimed: records
    written before today said "completed" for turns that computed nothing."""
    turns = _turns(events)
    if not turns:
        return live_default
    turn = turns[-1]
    outcome = live_default
    for e in turn:
        t = e.get("type")
        if t == "error":
            outcome = e.get("outcome") or FAILED
        elif t == "done":
            if outcome in (FAILED, INTERRUPTED):
                continue
            stated = e.get("outcome")
            if stated in (FAILED, INTERRUPTED):
                outcome = stated
            else:
                outcome = clean_outcome(turn)
    return outcome
