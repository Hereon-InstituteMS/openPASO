"""audit_results — self-consistency checks on the agent's own output files.

THE MEASURED FAILURE MODE (53 development runs with openPASO):
  * 53/53 read the knowledge; the advice channel works.
  * 51/53 execute solvers through run_bash. The verification machinery —
    residual checks in run_simulation, the critic gate, the new unsaved-work
    notice, verify_mesh_independence (used by 0/53) — hangs off tools the
    agents do not use. Every check we built sits on a road they do not drive.
  * Result: 37/54 deliver a complete answer, and most are WRONG in ways visible
    without any answer key: two runs' errors sit FLAT at ~5e-7 across all
    levels (solver tolerance floor — the task even says "converge to 1e-10"),
    one run converges at order 2 on a task that states order 3 (locking: the
    agent WROTE "mixed formulation" in its own header, built the mixed space,
    then assembled the plain form and solved that).

So the failure is not ignorance and not stubbornness: the agent reads, agrees,
writes the plan in its own comments — and nothing in the loop ever makes it
LOOK at whether what it produced matches what it planned. The critic reviews
the SETUP before the run. An independent check reviews the ANSWER after
delivery, against sealed truth. Nothing reviews the RESULT in between, and
the agent cannot see the sealed key, so it cannot check itself against truth
even if it tries.

THIS check needs no truth. It reads only what the agent itself produced —
error/QoI sequences per level — and answers three questions any numerate
reviewer would ask before delivering:
  1. Do successive levels actually approach each other? (self-convergence,
     no exact solution needed)
  2. At what observed order — and does that match the order you are about to
     CLAIM?
  3. Are the differences sitting at a floor (levels nearly identical), which
     means the thing limiting you is a tolerance, not the mesh?

It attaches to the RESULT, not to a run tool, so run_bash cannot bypass it.
"""
from __future__ import annotations

import csv as _csv
import json
import math
import hashlib
import re
from pathlib import Path

# Directories openPASO itself creates. A stale zero-valued probe file in one of
# these once produced a NEAR-ZERO FIELD finding on verified-correct work, and
# only in runs that went through openPASO, which is why the search is filtered
# rather than naive.
_SCRATCH = {"simulation_outputs", "coupling", "meshes", "benchmark_results",
            ".git", "__pycache__", "runs", "runs_quarantine",
            # openPASO's own source layout: a cell once pointed audit_results at the
            # server's src tree and the ladder read core/instructions.py as a
            # participant script written from scratch
            "core", "tools", "backends", "src", "site-packages"}



# ── DELIVERABLE DISCOVERY WITHOUT A NAMING SCHEME ────────────────────────────
#
# A refinement study leaves level-indexed files behind,
# <kind>_level<k>[_<side>].<ext>, and the KIND is whatever the task told the
# agent to call them. Nothing here knows a task's names: the kind is read from
# the agent's own files and each file is classified by what it holds (a
# residual history, an interface trace, a field on probe points, a captured run
# log), so every check below works for any naming a task prescribes.
_LEVEL_FILE = re.compile(
    r"^(?P<kind>[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*?)_level(?P<k>\d+)"
    r"(?:_(?P<side>[A-Za-z0-9]+))?\.(?P<ext>[A-Za-z0-9]+)$")


def _level_files(work: Path, ext: str = "csv") -> list:
    """(path, kind, level, side) for every level-indexed file with that
    extension, outside openPASO's own scratch directories."""
    out = []
    for q in work.rglob(f"*level*.{ext}"):
        if not q.is_file():
            continue
        try:
            if _SCRATCH & set(q.relative_to(work).parts[:-1]):
                continue
        except ValueError:
            continue
        m = _LEVEL_FILE.match(q.name)
        if not m or m.group("ext").lower() != ext.lower():
            continue
        out.append((q, m.group("kind").lower(), int(m.group("k")),
                    m.group("side") or ""))
    return sorted(out, key=lambda t: str(t[0]))


def _level_of(q: Path):
    m = _LEVEL_FILE.match(q.name)
    return int(m.group("k")) if m else None


def _side_of(q: Path) -> str:
    m = _LEVEL_FILE.match(q.name)
    return (m.group("side") or "") if m else ""


def _csv_role(q: Path, kind: str | None = None) -> str:
    """'history' | 'interface' | 'field' | 'raw' -- from the kind word first,
    from the file's own header second."""
    if kind is None:
        m = _LEVEL_FILE.match(q.name)
        kind = m.group("kind").lower() if m else ""
    if kind == "field":
        return "raw"      # openPASO's own per-participant dump, not a deliverable
    if any(w in kind for w in ("resid", "hist", "iter", "converg")):
        return "history"
    if any(w in kind for w in ("iface", "interface", "seam", "coupl")):
        return "interface"
    try:
        with open(q, errors="ignore") as fh:
            head = fh.readline().lower()
    except OSError:
        return "field"
    cols = [c.strip().strip('"') for c in head.split(",")]
    if cols and cols[0].startswith("iter"):
        return "history"
    if any(c in ("qn", "q_n", "q", "flux", "normal_flux", "traction", "tn",
                 "t_n") or c.startswith(("q_", "flux", "traction"))
           for c in cols):
        return "interface"
    return "field"


def _history_files(work: Path) -> list:
    return [q for q, kind, _k, _s in _level_files(work)
            if _csv_role(q, kind) == "history"]


def _field_files(work: Path) -> list:
    return [q for q, kind, _k, _s in _level_files(work)
            if _csv_role(q, kind) == "field"]


def _interface_files(work: Path, sided: bool = False) -> list:
    return [q for q, kind, _k, s in _level_files(work)
            if _csv_role(q, kind) == "interface" and (s or not sided)]


def _level_logs(work: Path, k: int | None = None) -> list:
    return [q for q, _kind, kk, _s in _level_files(work, "log")
            if k is None or kk == k]


# A line stating the number of degrees of freedom, in the spellings solvers
# and their users actually print.
_DOF_LINE = re.compile(
    r"^\s*(?:N_?DOFS?|DOFS?|NUM(?:BER)?_?(?:OF_?)?_?DOFS?|DEGREES OF FREEDOM)"
    r"\s*[=:]\s*(\d+)\s*$", re.I | re.M)

_SUMMARY_HINT: dict = {}


def _summary_file(work: Path, hint=None):
    """The agent's summary/answer file: the caller's hint when it gave one
    (the harness knows which file it just saw written), else the shallowest
    non-empty text file whose name says result, summary, report or answer."""
    hint = hint or _SUMMARY_HINT.get(str(work))
    if hint:
        hp = Path(hint)
        if hp.is_file():
            return hp
    cands = []
    for q in work.rglob("*"):
        if not q.is_file() or q.suffix.lower() not in (".txt", ".md", ""):
            continue
        try:
            rel = q.relative_to(work)
        except ValueError:
            continue
        if _SCRATCH & set(rel.parts[:-1]):
            continue
        if not re.search(r"result|summary|report|answer", q.stem, re.I):
            continue
        try:
            nonempty = q.stat().st_size > 0
        except OSError:
            continue
        cands.append((0 if nonempty else 1, len(rel.parts), q.name.lower(), q))
    return min(cands)[3] if cands else None


def _sequences_from_workdir(work: Path) -> dict[str, list[float]]:
    """Pull per-level scalar sequences out of whatever the agent wrote.

    Looks for the common shapes seen across the development runs: summary-file
    fields (ERRORS_UX = a, b, c / L2_ERRORS = ...), per-level csv/json files
    with a recognisable error/qoi column. Returns {label: [level values]}.
    """
    seqs: dict[str, list[float]] = {}
    rt = _summary_file(work)
    if rt is not None and rt.is_file():
        for line in rt.read_text(errors="replace").splitlines():
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.+)$", line)
            if not m:
                continue
            label = m.group(1).upper()
            # Only ERROR-LIKE labels. Scanning every "NAME = a, b, c" line
            # meant the task's own mesh line (H = 0.125, 0.0625, 0.03125)
            # read as an error sequence converging at exactly 1.00 and
            # produced ORDER MISMATCH on correct work, advising the one
            # change every task forbids; and a RESIDUALS line at the 1e-10
            # the task REQUIRES read as a tolerance FLOOR, advising the agent
            # to loosen a tolerance it was told to tighten.
            if not any(k in label for k in ("ERROR", "ERR", "L2", "LINF",
                                            "DIFF", "RESID_ERR")):
                continue
            vals = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(2))
            if len(vals) >= 3:
                try:
                    seq = [float(v) for v in vals]
                    if all(math.isfinite(v) for v in seq):
                        seqs[m.group(1)] = seq
                except ValueError:
                    pass
    return seqs


def _is_participant_dir(d: Path) -> bool:
    """True when this directory is one participant's own work dir.

    That is what makes two same-named per-level files two SIDES rather than a
    collision: each participant runs in its own directory and dumps there.
    """
    try:
        return (d / "config.json").is_file() or any(d.glob("participant*.py"))
    except OSError:
        return False


def _sequences_from_level_csvs(work: Path) -> dict[str, list[float]]:
    """Self-convergence from per-level CSVs on a common probe grid.

    The naming convention a task prescribes fixes ONE probe grid across
    levels, so files like <kind>_level{1,2,3}.csv join on coordinates. With
    no exact solution the per-level 'value' is the max successive difference
    per field: |u_l - u_l+1| shrinking at order p is the self-convergence
    signature of a study converging at order p (Richardson). Also returns each
    field's finest-level magnitude, because a near-zero field is its own
    finding.
    """
    import csv as _csv
    import re as _re

    # ONE GROUP PER (KIND, SIDE), NOT ONE PER LEVEL NUMBER.
    #
    # This globbed *level{i}*.csv and refused to proceed when more than one
    # file matched. On a SINGLE-CODE result set that is right. On a COUPLED one
    # every level has five matches — solution_level1_A, solution_level1_B,
    # interface_level1_A, interface_level1_B, residual_level1 — so the audit
    # reported AMBIGUOUS INPUT and found zero sequences. Every check it exists
    # for (near-zero field, tolerance floor, order, monotonicity) was therefore
    # dead on every coupled problem, which is half the development runs and
    # the half that counted for nothing.
    #
    # Measured on one development run: the agent delivered three levels of
    # identically zero displacement, the audit said "ambiguous" instead of
    # "your field is zero", and the run reached the independent check, which
    # read it as fabricated, with no run behind it. The gate had the data and
    # did not look at it.
    #
    # The task's naming convention names the sides, so the grouping is not a
    # guess: <kind>_level<k>[_<side>].csv. Ambiguity is still refused, but only
    # when two files claim the SAME (kind, level, side).
    _PAT = _re.compile(r"^(?P<kind>[A-Za-z_]+?)_level(?P<k>\d+)"
                       r"(?:_(?P<side>[A-Za-z0-9]+))?\.csv$")
    # RECURSIVE, MINUS OUR OWN SCRATCH. This globbed the TOP LEVEL only,
    # because rglob once picked openPASO's own directories — benchmark_results/,
    # coupling/, meshes/, simulation_outputs/, all created by openPASO itself and
    # all sorting before solution_*.csv — and a stale zero-valued probe file
    # there produced a NEAR-ZERO FIELD finding on verified-correct work, and
    # only in runs that went through openPASO. The restriction fixed that and
    # introduced a blind spot: one development run wrote its 15 files into
    # level1/, level2/, level3/, and the audit found ZERO sequences and
    # returned clean=True on a full result set. Name the directories to skip
    # instead of refusing to descend at all.
    groups: dict[tuple, dict[int, list]] = {}
    resid_files: list = []
    for q in work.rglob("*level*.csv"):
        if not q.is_file():
            continue
        if _SCRATCH & set(q.relative_to(work).parts[:-1]):
            continue
        m = _PAT.match(q.name)
        if not m:
            continue
        kind = m.group("kind").lower()
        if kind == "field":
            # openPASO's own per-participant raw dump (field_level<k>.csv, one per
            # side's work dir), not a deliverable. Two sides writing the same
            # name at the same depth read as AMBIGUOUS INPUT on verified-correct
            # work (measured on a real coupled rebuild) -- so it is not a
            # sequence.
            continue
        if _csv_role(q, kind) == "history":
            # NOT SKIPPED ANY MORE — see _residual_findings below. It is not a
            # field on a grid, so it does not join the per-level sequences, but
            # it is the single file that decides a third of coupled outcomes
            # and the audit used to look straight past it.
            resid_files.append(q)
            continue
        key = (kind, (m.group("side") or "").upper())
        groups.setdefault(key, {}).setdefault(int(m.group("k")), []).append(q)

    # SHALLOWEST WINS, and only a TIE is ambiguous.
    #
    # The deliverable belongs in the work dir; a copy deeper down is a
    # byproduct. Every deal.II run keeps one — cmake builds in build/ and the
    # solver writes its CSVs beside the binary — so descending made three
    # verified-correct runs report AMBIGUOUS INPUT, which is exactly the false
    # alarm the old top-level-only rule was protecting against. Depth decides
    # it without having to enumerate every scratch directory a backend might
    # invent; genuine ambiguity (two files at the SAME depth for one slot) is
    # still refused.
    dup: list = []
    split: dict[tuple, dict[int, list]] = {}   # side recovered from the directory
    drop: list = []
    for k, byl in groups.items():
        for lv, qs in list(byl.items()):
            if len(qs) == 1:
                continue
            depth = {q: len(q.relative_to(work).parts) for q in qs}
            shallowest = min(depth.values())
            keep = [q for q in qs if depth[q] == shallowest]
            if len(keep) > 1:
                # THE SIDE IS IN THE DIRECTORY WHEN IT IS NOT IN THE NAME.
                #
                # The served per-level dump writes interface_level<k>.csv into
                # each participant's OWN work dir, so a two-sided run has two
                # files with one name at one depth and they tie here. That is
                # not a collision, it is the normal layout -- `field` was
                # already skipped for this exact reason, and `interface` was
                # not, so a correct coupled set was refused.
                #
                # MEASURED, and it cost a whole cell: one run was told to
                # "remove the stale ones", deleted a side's files, and the next
                # audit found ZERO sequences -- so the divergence in its own
                # ladder (self-differences growing five-fold per level) was
                # never reported and the run submitted a diverging answer as
                # final. Split by the directory instead of refusing, which
                # recovers both sides rather than discarding both.
                by_dir = {q.parent: q for q in keep}
                if len(by_dir) == len(keep) and all(
                        _is_participant_dir(d) for d in by_dir):
                    for d, q in by_dir.items():
                        split.setdefault((k[0], d.name.upper()), {})[lv] = [q]
                    drop.append((k, lv))
                    continue
                dup.append(f"{k[0]}{'_' + k[1] if k[1] else ''} level {lv}: "
                           f"{sorted(str(x.relative_to(work)) for x in keep)}")
            byl[lv] = keep
    for k, lv in drop:                       # the tied slot became one slot per side
        groups[k].pop(lv, None)
    for k, byl in split.items():
        groups.setdefault(k, {}).update(byl)
    groups = {k: v for k, v in groups.items() if v}
    if dup:
        return {"__ambiguous__": dup}

    out_all: dict[str, list[float]] = {}
    for key in sorted(groups):
        seq = _one_sequence(groups[key], key, _csv)
        out_all.update(seq)
    return out_all


def _one_sequence(by_level: dict, key: tuple, _csv) -> dict[str, list[float]]:
    """The original per-level analysis, for ONE (kind, side) group."""
    kind, side = key
    tag = f"{kind}_{side}" if side else kind
    levels: list[dict] = []
    for i in range(1, 9):
        # TOP LEVEL ONLY. rglob + sorted(cands)[0] picked the
        # lexicographically first PATH, so openPASO's own scratch directories —
        # benchmark_results/, coupling/, meshes/, simulation_outputs/, all
        # created by openPASO itself and all sorting before solution_*.csv — won
        # over the agent's real output. A stale zero-valued probe file left by
        # a failed first run then produced a NEAR-ZERO FIELD finding on
        # verified-correct work, and only in runs that went through openPASO.
        cands = by_level.get(i) or []
        if not cands:
            break
        rows = {}
        try:
            with open(sorted(cands)[0]) as fh:
                r = _csv.DictReader(fh)
                # Headers arrive as "x, y, u" — WITH spaces. Unstripped, " y"
                # passed the coordinate filter and became a data field, and
                # the join key collapsed to (x, 0) for every row, comparing
                # unrelated rows between levels. That single character made
                # 10 of 22 verified-correct runs flag as ORDER MISMATCH.
                norm = {(c or "").strip(): c for c in (r.fieldnames or [])}
                fields = [k for k in norm
                          if k and k.lower() not in ("x", "y", "z")]
                if "x" not in norm or "y" not in norm:
                    break                     # no coordinates: cannot join
                # z joins the key when present: with only (x, y), every 3D
                # column of points collapses onto one key and the three 3D
                # problems false-alarmed exactly like the whitespace bug.
                zc = norm.get("z")
                for row in r:
                    key = (round(float(row[norm["x"]]), 9),
                           round(float(row[norm["y"]]), 9),
                           round(float(row[zc]), 9) if zc else 0.0)
                    rows[key] = {f: float(row[norm[f]]) for f in fields
                                 if row.get(norm[f]) not in (None, "")}
        except (OSError, ValueError):
            break
        if rows:
            levels.append(rows)
    if not levels:
        return {}
    # A ZERO FIELD IS VISIBLE AT LEVEL ONE, AND THAT IS WHEN IT IS WORTH
    # SAYING. This returned {} below three levels, so the NEAR-ZERO check —
    # the cheapest catch in the audit and the one that names an unwired load —
    # stayed silent exactly while the agent still had the budget to fix it.
    # Self-convergence genuinely needs three levels; a magnitude does not need
    # any. Emit the magnitude from whatever exists and the differences only
    # when there are enough levels to form them.
    if len(levels) < 3:
        out: dict[str, list[float]] = {}
        fields = sorted({f for lv in levels for v in lv.values() for f in v})
        for f in fields:
            mag = max((abs(v[f]) for v in levels[-1].values() if f in v),
                      default=0.0)
            out[f"magnitude_{tag}_{f}"] = [mag]
        return out
    out: dict[str, list[float]] = {}
    fields = sorted({f for lv in levels for v in lv.values() for f in v})
    for f in fields:
        diffs = []
        for a, b in zip(levels, levels[1:]):
            common = [k for k in a if k in b and f in a[k] and f in b[k]]
            if len(common) < 4:
                diffs = []
                break
            # RMS, not max: over ~2000 probe points the max difference is
            # dominated by a single worst point and its decay is noisy; the
            # RMS decays at the field's true self-convergence rate. The max
            # variant mis-flagged a verified-correct run.
            import math as _m
            diffs.append(_m.sqrt(sum((a[k][f] - b[k][f]) ** 2
                                     for k in common) / len(common)))
        if len(diffs) >= 2 and all(d > 0 for d in diffs):
            out[f"selfdiff_{tag}_{f}"] = diffs
        mag = max((abs(v[f]) for v in levels[-1].values() if f in v),
                  default=0.0)
        out[f"magnitude_{tag}_{f}"] = [mag]
    return out



def residual_findings(work: Path) -> list[dict]:
    """What the coupling residual history says about itself.

    THE FILE THE AUDIT USED TO SKIP. The residual history is not a field on a
    grid, so it never joined the per-level sequences — and it is the single
    file that decides the largest failure bucket on coupled problems:
    coupling evidence that contradicts itself accounts for 69 of 250 coupled
    development runs checked (28%) — the second-largest reason after an
    outright give-up. Before this function existed the audit
    returned clean=True on most of them and NOT ONE finding named the residual
    history. (An earlier version of this note said "80 of 250" and "53 of 91";
    neither denominator is reconstructible from the tree and both are
    withdrawn in favour of the counts above, which are.) The agent was told its work was self-consistent while
    three levels carried non-finite residuals.

    Every check here is computable from the agent's own files and needs no
    reference solution: enough iterations to be an iteration, positive and
    finite, an actual decrease, and a history that is not a constant column.
    They mirror src/blind_eval/evidence.py::coupling_evidence, which is what
    an independent check applies afterwards — so a finding here is a warning
    about an outcome the agent would otherwise meet for the first time after
    delivery.
    """
    import csv as _csv
    out: list[dict] = []
    for q in _history_files(work):
        vals: list[float] = []
        try:
            with open(q) as fh:
                for row in _csv.reader(fh):
                    if not row:
                        continue
                    try:
                        vals.append(float(row[-1]))
                    except ValueError:
                        continue            # header
        except OSError:
            continue
        name = q.name
        # A LEADING NaN IS openPASO'S OWN history[0], NOT THE AGENT'S DEFECT.
        #
        # `couple` returns a history whose first entry is NaN by construction —
        # there is no previous export to compare the first one against. The
        # independent check knows this and drops it (blind_eval.evidence
        # records `dropped_leading_nonfinite`), but this audit, which is the
        # gate we tell agents to run BEFORE delivering, did not.
        #
        # Measured on one development run — the best coupled run of that set,
        # with 4C and Kratos both proven to have run and the coupling proven
        # and not forged, with the residual falling 0.309 -> 8.2e-07 — this
        # audit returned
        # "clean": false and told it, three times, that "the residual was never
        # actually computed from the two sides". openPASO produced the NaN, then
        # reported it to the agent as evidence of the agent's own failure, and
        # the only fix available to an agent that believes it is to go and
        # break something that was right.
        #
        # Dropped, not tolerated: a NaN anywhere LATER in the history is still
        # a real finding, and so is a non-positive value anywhere at all.
        leading_nan = bool(vals) and vals[0] != vals[0]
        if leading_nan:
            vals = vals[1:]
        if len(vals) < 3:
            out.append({"sequence": name, "values": vals,
                        "finding": (
                            f"COUPLING HISTORY TOO SHORT: {len(vals)} "
                            f"iteration(s). A partitioned scheme that reached "
                            f"a fixed point leaves a history; fewer than three "
                            f"entries is read as not having coupled.")})
            continue
        if any((v != v) or v in (float('inf'), float('-inf')) or v <= 0
               for v in vals):
            out.append({"sequence": name, "values": vals[:6],
                        "finding": (
                            "NON-POSITIVE OR NON-FINITE RESIDUAL: a relative "
                            "interface mismatch is a positive number. A zero, "
                            "a negative or a nan here means the residual was "
                            "never actually computed from the two sides.")})
            continue
        if vals[0] / max(vals[-1], 1e-300) < 10.0:
            # A STALLED ITERATION AND A WANDERING ONE HAVE DIFFERENT CAUSES,
            # AND ONLY ONE OF THEM IS THE RELAXATION.
            #
            # Slow monotone decay is under-relaxation: the sequence goes down
            # every step and simply needs more steps or a better theta. A
            # sequence that goes UP about as often as it goes DOWN is not
            # converging slowly, it is not converging at all, and more
            # iterations and a smaller theta cannot fix it -- each side is
            # undoing the other.
            #
            # MEASURED on one coupled run whose own critic had computed the
            # conductance ratio correctly and concluded, rightly, that the
            # default relaxation should converge in a few steps: 499
            # iterations, 244 steps up and 254 down, wandering between 0.24 and
            # 1.26 about a mean of 0.65, 68% of consecutive steps reversing
            # direction. It spent its whole budget raising max_iter to 200 and
            # then switching accelerator, which is what this finding used to
            # leave a reader to guess.
            ups = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
            downs = len(vals) - 1 - ups
            wandering = len(vals) >= 8 and ups >= 0.35 * (len(vals) - 1)
            why = ""
            if wandering:
                flips = sum(1 for a, b, c in zip(vals, vals[1:], vals[2:])
                            if (b - a) * (c - b) < 0)
                why = (
                    f" IT IS NOT SLOW, IT IS WANDERING: {ups} step(s) up against "
                    f"{downs} down over {len(vals)} iterations"
                    + (f", {flips / max(len(vals) - 2, 1):.0%} of them reversing "
                       f"direction" if len(vals) > 2 else "")
                    + ". Under-relaxation makes a sequence go DOWN every step and "
                    "only slowly, so this is not the relaxation and neither more "
                    "iterations nor a smaller theta will fix it: the two sides "
                    "are undoing each other. Check the two things that do that. "
                    "(1) THE SIGN: each side must export its OWN outward normal "
                    "flux, so at the same physical point the two sides report "
                    "opposite signs; a side that re-exports the partner's sign "
                    "gives exactly this. (2) THE POINTS: both sides must exchange "
                    "the SAME interface points in the SAME order -- compare the "
                    "coordinate lists in the two exports.json, not just their "
                    "lengths.")
            out.append({"sequence": name, "values": [vals[0], vals[-1]],
                        "finding": (
                            f"RESIDUAL BARELY MOVED: {vals[0]:.3g} -> "
                            f"{vals[-1]:.3g}, a factor of "
                            f"{vals[0] / max(vals[-1], 1e-300):.2g}. That is "
                            f"not a converged coupling; it is the iteration "
                            f"standing still, and it is read as not "
                            f"coupled." + why)})
            continue
        if len(set(f"{v:.12g}" for v in vals)) == 1:
            out.append({"sequence": name, "values": vals[:4],
                        "finding": (
                            "CONSTANT RESIDUAL COLUMN: every iteration reports "
                            "the same number, so the column is a placeholder "
                            "rather than a measured mismatch.")})
            continue
        # THE AUDIT PASSED A FORGERY. Measured on one development run, whose
        # three levels each held 1.0 falling to exactly 1e-06 in ten steps,
        # BIT-IDENTICAL across all three, while its own NDOF lines said the
        # mesh had changed (400, 255, 72). An independent check labels that
        # fabricated, with no run behind it. This audit — the gate we tell
        # agents to run before delivering — returned "clean": true.
        #
        # A gate that blesses an invented history is worse than no gate: it
        # tells an agent the shortcut passed. The two rules below are already
        # PUBLIC — the served coupling must-read states both, in as many words,
        # so nothing is revealed by checking them here. They are imported from
        # the shared evidence module rather than restated, so the thresholds
        # cannot drift apart from the ones an independent check actually
        # applies.
        try:
            from blind_eval.evidence import (          # noqa: PLC0415
                _decay_ratio_cv, SYNTHETIC_RATIO_CV, _FORGED_DECAY_MIN_DROP)
        except Exception:                               # module not importable
            continue
        cv = _decay_ratio_cv(vals)
        if (cv is not None and cv < SYNTHETIC_RATIO_CV
                and vals[0] / max(vals[-1], 1e-300) > _FORGED_DECAY_MIN_DROP):
            out.append({"sequence": name, "values": vals[:5],
                        "finding": (
                            f"THIS READS AS A WRITTEN-IN SEQUENCE, NOT A "
                            f"MEASURED ONE: the step-to-step ratio is constant "
                            f"to a coefficient of variation of {cv:.1e}. A real "
                            f"partitioned iteration's rate wanders as the "
                            f"error's modal composition changes. This is read "
                            f"as fabrication, which counts for less than an "
                            f"honest report that the iteration did not "
                            f"converge. THE "
                            f"HONEST FILE IS CHEAPER THAN THIS ONE: your "
                            f"coupling loop already computes an interface "
                            f"mismatch every iteration to decide when to stop "
                            f"-- append THAT number to the CSV inside the loop, "
                            f"one line, and the history is real whatever it "
                            f"shows. If your loop never computed a mismatch, "
                            f"it never coupled, and the honest entry is a "
                            f"could-not-finish report plus your best "
                            f"single-domain fields, which is worth more than "
                            f"this file.")})
    # BIT-IDENTICAL HISTORIES ACROSS LEVELS — checked across files, not within.
    #
    # The per-file loop above cannot see it: each level's column is individually
    # unremarkable. The history depends on the discretisation, so the same
    # numbers at two mesh levels cannot both be measurements.
    seqs: dict[str, list[str]] = {}
    for q in _history_files(work):
        try:
            body = tuple(r[-1].strip() for r in _csv.reader(q.open()) if r)
        except OSError:
            continue
        if len(body) >= 3:
            # KEY BY LEVEL NUMBER, NOT BY FILE NAME.
            #
            # Agents write their outputs twice: once at the contractual path and
            # once under a per-level directory of their own. This loop walked
            # both copies, so a file was compared with ITSELF and the run was
            # told "IDENTICAL RESIDUAL HISTORY AT 2 MESH LEVELS:
            # residual_level1.csv, residual_level1.csv agree digit for digit"
            # — and at three copies, "AT 3 MESH LEVELS" naming level 1 three
            # times.
            #
            # Measured on one development run: EIGHT findings, every one of them
            # a copy paired with itself, on a run whose field is within 3% of
            # the true solution on both subdomains. A fabrication accusation is
            # the most damaging thing this file can say, and it was saying it
            # about tidy output habits.
            _lv = _level_of(q)
            lvl = str(_lv) if _lv is not None else q.name
            seqs.setdefault(repr(body), []).append((lvl, q.name))
    for _, hits in seqs.items():
        levels = sorted({lvl for lvl, _ in hits})
        if len(levels) > 1:
            names = sorted({name for _, name in hits})
            out.append({"sequence": ", ".join(names), "values": [],
                        "finding": (
                            f"IDENTICAL RESIDUAL HISTORY AT {len(levels)} MESH "
                            f"LEVELS (levels {', '.join(levels)}): "
                            f"{', '.join(names)} agree digit "
                            f"for digit. The history depends on the "
                            f"discretisation, so these cannot both be "
                            f"measurements; this reads as an invented history.")})
    return out


def contract_findings(work: Path, only_levels: set | None = None) -> list[dict]:
    """Result-set defects an independent check fails on that this audit never
    checked.

    `only_levels` restricts every per-level judgement to the levels named,
    which is what lets these findings be delivered DURING the ladder instead
    of at hand-in: at level 1 the agent is owed the level-1 proof and nothing
    about levels it has not reached. Left None (the default, and what the
    hand-in audit passes) nothing changes.

    The audit existed to catch what sinks a result set, and 27 of 36
    single-code runs called it — but it looked only at the convergence
    sequences. Two naming-convention failures it was blind to killed real runs
    on problems that had worked before:

      a level without its captured run log — no per-level run log carrying
                                             `NDOF = <n>`
      a file not assignable to a subdomain  — a level-1 field file delivered
                                             twice, in two places, with
                                             different contents

    Both are cheap to detect from the agent's own files, and both are fatal
    when an independent check sees them. Same shape as the other defects this
    project keeps finding: the mechanism existed, was called, and did not
    reach the case it was built for.
    """
    out: list[dict] = []

    # 1. every level that has a solution file needs a run log carrying NDOF
    sols = _field_files(work)
    levels = set()
    for f in sols:
        if _level_of(f) is not None:
            levels.add(_level_of(f))
    if only_levels is not None:
        levels &= set(only_levels)
    if levels:
        # PER SIDE, NOT PER LEVEL. This asked whether ANY log at a level carried a dof count, and a
        # coupled run is graded per side: "the file for side A must carry the output of the code that
        # solved subdomain A". Measured on a round-48 cell that had both codes PROVEN and the coupling
        # PROVEN and lost anyway -- side A had no dof line at levels 2 and 3, side B none either, and
        # this finding stayed silent because level 1's logs had them. A gate that is weaker than the
        # contract it guards is worth nothing at the moment it matters.
        sides = sorted({s for _q, _k2, _kk, s in _level_files(work, "log") if s})
        missing = []
        for k in sorted(levels):
            logs = _level_logs(work, k)
            if sides:
                for side in sides:
                    have = [q for q in logs
                            if (_LEVEL_FILE.match(q.name).group("side") or "") == side]
                    if not any(_DOF_LINE.search(q.read_text(errors="ignore")) for q in have):
                        missing.append(f"{k}{side}")
            elif not any(_DOF_LINE.search(q.read_text(errors="ignore")) for q in logs):
                missing.append(str(k))
        if missing:
            _what = ("level/side " if sides else "level(s) ")
            out.append({"sequence": "run-log contract", "values": [],
                        "finding": (
                f"NO DOF-COUNT LINE in a captured run log for {_what}"
                f"{missing}. Every level needs its own run log PER SIDE, each carrying that "
                f"side's own solver console output and a line stating the number of degrees of "
                f"freedom (`NDOF = 1234`) from that side's own dof count; a level-and-side "
                f"without that log cannot be shown to have run, however good its numbers are.")})

    # 1a. A COUPLED SUBMISSION WITHOUT ITS ITERATION HISTORY.
    #
    # Measured on a round-48 cell: both codes PROVEN, the interface satisfied, three levels of fields
    # and interface tables -- and no residual_level<k>.csv anywhere, so the coupling itself could not
    # be shown and the submission was read as malformed. The audit named everything except the thing
    # that decided it. Fires only when the work LOOKS coupled (side-tagged files), so a single-code
    # run is never told to produce a history it has no reason to have.
    if levels and sides:
        hist = {kk for _q, kind, kk, _s in _level_files(work, "csv") if kind.startswith("residual")}
        gap = sorted(levels - hist)
        if gap:
            out.append({"sequence": "coupling evidence", "values": [],
                        "finding": (
                f"NO ITERATION HISTORY for level(s) {gap} of what looks like a COUPLED run "
                f"(this work carries side-tagged files for {', '.join(sides)}). Each coupled level "
                f"needs its own residual_level<k>.csv listing every iteration of the partitioned "
                f"scheme and the interface mismatch at it, from the first to the last. Fields and "
                f"interface tables do not show that two codes exchanged anything: a coupled run "
                f"that cannot produce that history cannot be read as coupled at all, however good "
                f"its numbers are.")})

    # 1b. IS THE DISCRETISATION THE ONE THE TASK ASKED FOR?
    #
    # Both numbers come from the agent's own two files, so this needs no key,
    # no spec and no backend knowledge: the NDOF the run printed at its
    # COARSEST level, against the number of rows in that level's solution file.
    #
    # Measured over the 336 single-code runs on disk that wrote both files:
    #
    #     highest ratio among verified-correct runs      0.50
    #     threshold NDOF/rows > 2 fires on 8 runs        0 verified correct
    #                                                   8 of 8 timed out or
    #                                                   fell short of the
    #                                                   prescribed levels
    #
    # The failure it names is specific and fatal, and it is visible at LEVEL
    # ONE while there is still time to fix it. Three openPASO runs of one Stokes
    # task hit it with an identical NDOF of 592,387 against a 1936-point
    # probe grid — a solve two orders of magnitude larger than the
    # prescribed coarsest mesh, which completes level 1 and then cannot finish
    # level 2 inside the clock. Three runs, one cause, no warning.
    #
    # It fires on the mirror-image defect too, and says so: one run wrote a
    # solution file with 2 rows, which trips the same ratio from below.
    if levels:
        k0 = min(levels)
        sol0 = [f for f in sols if _level_of(f) == k0 and not _side_of(f)] or \
               [f for f in sols if _level_of(f) == k0]
        nd = None
        for p in _level_logs(work, k0):
            m = _DOF_LINE.search(p.read_text(errors="ignore"))
            if m:
                nd = int(m.group(1))
                break
        if nd and sol0:
            try:
                rows = sum(1 for _ in open(sol0[0], errors="ignore")) - 1
            except OSError:
                rows = 0
            if rows > 0 and nd > 2 * rows:
                out.append({"sequence": "discretisation size", "values": [],
                            "finding": (
                    f"AT YOUR COARSEST LEVEL YOUR OWN NDOF IS {nd} AGAINST "
                    f"{rows} ROWS in {sol0[0].name} — a ratio of "
                    f"{nd / rows:.0f}. Across every run measured here, no "
                    f"result set verified correct against an independent "
                    f"reference exceeds 0.5, and every run above 2 either "
                    f"ran out of time or never reached the "
                    f"finer levels. Two causes produce this, and they need "
                    f"opposite fixes: (a) the mesh is far larger than the "
                    f"coarsest level the task prescribes, so level 1 is "
                    f"already an expensive solve and the finer levels cannot "
                    f"finish — re-read the prescribed mesh sizes and start at "
                    f"the coarsest one; or (b) the solution file has far fewer "
                    f"rows than the task's probe grid, so the deliverable is "
                    f"short whatever the solve did. Check which one you have "
                    f"NOW: fixing it at level 1 costs one solve, at level 3 "
                    f"it costs the whole ladder.")})

    # 1c. A SOURCE TERM BUILT FROM ELEMENT-LOCAL COORDINATES.
    #
    # This is a STATIC read of the agent's own script, which is a departure
    # from the rest of this file, and it is here because it is the one failure
    # that no numeric self-check can see. Measured on one development run: the
    # run bound `x` and `y` to specialcf.xref(2) — the position inside the
    # reference element — while its source term was stated in global
    # coordinates. Its own comment read "reference coordinates which equal
    # physical coords for unit square". They do not.
    #
    # The consequence is invisible to every self-consistency test: the form
    # assembles, the solve succeeds, the successive differences fall smoothly,
    # the audit passed it, and the answer is wrong. Replaying its script
    # unchanged reproduces 6.158955e-02 against the correct 7.196098e-02, and
    # order 0.028 against 2.069 measured against an independent reference.
    #
    # Only flagged when the run ALSO has a coordinate-indexed deliverable, so
    # a legitimate use of xref (a per-element quantity, an error indicator) in
    # a run that never claims a global field is left alone.
    if levels:
        for script in list(work.rglob("*.py"))[:40]:
            if _SCRATCH & set(script.relative_to(work).parts[:-1]):
                continue
            try:
                text = script.read_text(errors="ignore")
            except OSError:
                continue
            if "specialcf.xref" not in text:
                continue
            rebinds = re.search(
                r"^\s*(?:x|y)\s*=\s*\w*xref\w*\s*\[", text, re.M) or \
                re.search(r"xref\s*=\s*specialcf\.xref", text)
            if not rebinds:
                continue
            out.append({"sequence": "source coordinates", "values": [],
                        "finding": (
                f"{script.name} BUILDS AN EXPRESSION FROM specialcf.xref, "
                f"WHICH IS THE POSITION INSIDE THE REFERENCE ELEMENT, NOT ON "
                f"THE DOMAIN. If your source term, coefficient or boundary "
                f"data was stated in global coordinates, this is silently a "
                f"different function: it repeats the same small range in every "
                f"element. Nothing raises — the form assembles, the solve "
                f"succeeds, and the refinement study looks orderly while "
                f"converging to the wrong answer. In NGSolve the `x` and `y` "
                f"you get from `from ngsolve import *` ARE the global "
                f"coordinates; do not rebind them. Check it in one line: your "
                f"source evaluated at an interior point must equal the "
                f"arithmetic you do by hand for that point, and must not "
                f"change when you look at a different element containing it. "
                f"Measured on a real run: the xref form gave "
                f"max|u| = 6.158955e-02 and order 0.028 against an independent "
                f"reference; the identical script using global x, y gave "
                f"7.196098e-02 and order 2.069.")})
            break

    # 1d. THE RESIDUAL YOU REPORT MUST MEASURE THE TWO SIDES, NOT AN ITERATE.
    #
    # Measured over 29 coupled result sets carrying two-sided interface files:
    # SEVEN report INTERFACE_RESIDUAL below 1e-5 while their own exported sides
    # differ by more than 5% — up to 102% — and about fifteen do so on the flux
    # balance, with mismatches near 100%. One development run is the clearest:
    # side A writes the interface field as exactly 0.0, side B writes -1.1e-03
    # which is B's entire field scale, the fluxes miss by 189%, and RESULT.txt
    # reports INTERFACE_RESIDUAL = 1.12e-07 after a six-iteration history that
    # falls smoothly from 1.0.
    #
    # So the coupling did converge — something converged — but not the quantity
    # the task asks about. Nothing else catches this: the residual history looks
    # textbook, the fields converge under refinement, and the run reads as a
    # clean success right up to the independent check.
    #
    # Both numbers come from the agent's OWN two files, so this needs no key and
    # no reference.
    iface = {}
    for f in _interface_files(work, sided=True):
        iface.setdefault(_level_of(f), {})[_side_of(f).upper()] = f
    for lvl in sorted(iface, reverse=True):
        side = iface[lvl]
        if len(side) != 2:
            continue
        sa, sb = sorted(side)
        # SPLIT BY THE FILE'S OWN HEADER, NOT BY THE SCALAR LAYOUT. This
        # check used to read column 2 as the field and column 3 as the flux
        # unconditionally. On a thermo-mechanical interface
        # (x,y,T,ux,uy,qn,tx,ty) that takes ux -- a displacement, CONTINUOUS
        # across the interface by construction -- for the flux, so a correct
        # coupling was reported as "the two outward fluxes fail to cancel by
        # 200%" (measured: |ux_A + ux_B|/max|ux| = 200.000% precisely
        # BECAUSE ux_A = ux_B, while the true trailing fluxes balanced to
        # 2.1e-3..5.1e-3, under this check's own 5% bar). The header-aware
        # splitter below already existed for the sign checks; this check
        # never used it.
        parsed = {}
        for s, p in side.items():
            byh = _read_iface_by_header(p)
            if byh is not None:
                parsed[s] = byh[1], byh[2]        # (values, fluxes)
        A0, B0 = parsed.get(sa), parsed.get(sb)
        if not A0 or not B0:
            continue
        (uA, qA), (uB, qB) = A0, B0
        # NO EARLY EXIT ON AN ALL-ZERO FLUX. This used to skip the level when
        # every flux entry on both sides was exactly zero -- so a field that
        # disagreed between the two sides went unreported beside a small
        # reported residual, because a DIFFERENT column happened to be dead.
        # The per-component loops below already skip a component that is ~0 on
        # both sides, so a dead flux contributes nothing to dq and the field is
        # still judged.
        if len(uA) < 2 or len(uA) != len(uB) or len(qA) != len(qB) or not qA:
            continue
        ncomp_u = min(len(uA[0]), len(uB[0])) if uA and uA[0] else 0
        ncomp_q = min(len(qA[0]), len(qB[0])) if qA and qA[0] else 0
        if not ncomp_u or not ncomp_q:
            continue
        du = dq = 0.0
        u_ref = max((abs(x) for row in (uA + uB) for x in row), default=0.0)
        q_ref = max((abs(x) for row in (qA + qB) for x in row), default=0.0)
        # a component that is physically ~0 on both sides (tangential
        # traction in a symmetric arrangement) must not be judged against
        # its own noise scale (reviewer finding)
        for c in range(ncomp_u):
            us = max(max(abs(r[c]) for r in uA), max(abs(r[c]) for r in uB))
            if us < 1e-9 * (u_ref or 1.0):
                continue
            du = max(du, max(abs(a[c] - b[c]) for a, b in zip(uA, uB))
                     / (us or 1.0))
        for c in range(ncomp_q):
            qs = max(max(abs(r[c]) for r in qA), max(abs(r[c]) for r in qB))
            if qs < 1e-9 * (q_ref or 1.0):
                continue
            dq = max(dq, max(abs(a[c] + b[c]) for a, b in zip(qA, qB))
                     / (qs or 1.0))
        reported = None
        rt = _summary_file(work)
        if rt is not None and rt.is_file():
            m = re.search(r"^\s*([A-Za-z_]*RESIDUAL[A-Za-z_]*)\s*[=:]\s*([-+0-9.eE]+)",
                          rt.read_text(errors="ignore"), re.M | re.I)
            if m:
                label = m.group(1)
                try:
                    reported = float(m.group(2))
                except ValueError:
                    reported = None
        worst = max(du, dq)
        if reported is not None and reported < 1e-5 and worst > 0.05:
            out.append({"sequence": "interface residual", "values": [],
                        "finding": (
                f"YOU REPORT {label} = {reported:.2e}, BUT YOUR OWN "
                f"TWO INTERFACE FILES AT LEVEL {lvl} DISAGREE: the field "
                f"differs by {du:.0%} of its own scale and "
                + (f"the two outward fluxes fail to cancel by {dq:.0%}. "
                   if q_ref > 0 else
                   "BOTH flux columns are identically zero, so no flux was "
                   "exchanged at all. ")
                + f"A partitioned scheme is "
                f"converged when the SIDES agree, so the number you report has "
                f"to be computed from the two exported profiles — "
                f"max|u_A - u_B| and max|q_A + q_B| over the shared interface "
                f"probes, each relative to its own scale — and not from an "
                f"internal iterate, an update norm, or one side's own solver "
                f"residual. Those all fall to 1e-7 while the two codes still "
                f"disagree by 100%, which is what this result set shows. "
                f"Recompute it from the files you just wrote, and if it is not "
                f"small, the coupling has not converged whatever the iteration "
                f"history says.")})
        break

    # 2. the same deliverable must not be delivered twice with different content
    by_name: dict[str, set] = {}
    for f in work.rglob("*.csv"):
        if not _LEVEL_FILE.match(f.name) or _csv_role(f) == "raw":
            continue
        try:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            continue
        by_name.setdefault(f.name, set()).add(digest)
    clashes = sorted(n for n, d in by_name.items() if len(d) > 1)
    if clashes:
        out.append({"sequence": "duplicate deliverables", "values": [],
                    "finding": (
            f"MORE THAN ONE DIFFERING COPY of {clashes[:4]}. Whoever verifies "
            f"your results cannot tell which one you meant and rejects the "
            f"result set. Keep exactly one copy of each deliverable; delete "
            f"scratch copies in subdirectories before you deliver.")})
    # 3. an incomplete or self-contradicting level set
    #
    # One run delivered solution_level1.csv and nothing else, with an empty
    # RESULT.txt, and this audit called it clean: the run-log check above only
    # asks about levels that HAVE a solution file, so one level with its log
    # looked complete. The result set was one level of four.
    # RESULT.txt is not always at the top level; an independent check finds it
    # anywhere.
    text = ""
    _sf = _summary_file(work)
    if _sf is not None and _sf.is_file():
        text = _sf.read_text(errors="ignore")
    if levels:
        span = max(levels)
        gaps = [k for k in range(1, span + 1) if k not in levels]
        if gaps:
            out.append({"sequence": "level sequence", "values": [],
                        "finding": (
                f"MISSING LEVEL(S) {gaps}: you have files for {sorted(levels)}, "
                f"so the sequence has a hole. A refinement study is counted "
                f"across the whole prescribed sequence.")})
        m = re.search(r"^\s*LEVELS\s*=\s*(\d+)", text, re.M)
        if m and int(m.group(1)) != len(levels):
            out.append({"sequence": "levels claimed", "values": [],
                        "finding": (
                f"Your summary file says LEVELS = {m.group(1)} but {len(levels)} "
                f"level(s) of solution files are present. The two are compared; "
                f"make them agree.")})
        if len(levels) < 3:
            out.append({"sequence": "level count", "values": [],
                        "finding": (
                f"ONLY {len(levels)} LEVEL(S) DELIVERED. A refinement study "
                f"needs a mesh sequence of at least three; an observed order "
                f"cannot be fitted from fewer, so a short result set counts "
                f"for nothing however good the levels are.")})
    if not text.strip():
        out.append({"sequence": "summary file", "values": [],
                    "finding": (
            "YOUR SUMMARY FILE IS MISSING OR EMPTY. The per-level files "
            "beside it are not read as an answer without the summary your "
            "task asks for.")})
    # 4. a hand-rolled sampler with the wrong shape-function normalisation
    #
    # Two development runs were lost to a sampler, not a solver. One run's
    # extractor wrote the QUAD4 factor 0.25 into a HEX8 shape function instead
    # of 0.125, so sum(N) = 2 and the delivered field was 2*u(x/2, y/2) -- a
    # fixed wrong field, converged to at order -0.035. Another's read the
    # nearest node's value, an O(h) reconstruction that caps the measured
    # order at 1.
    #
    # Both are visible in the agent's own scripts, without any key.
    for script in sorted(work.rglob("*.py")):
        try:
            src = script.read_text(errors="ignore")
        except OSError:
            continue
        if re.search(r"hex8|HEX8|8\s*,?\s*#\s*nodes|zeta", src) and \
                re.search(r"0\.25\s*\*\s*\(1\s*[-+]\s*xi", src):
            out.append({"sequence": script.name, "values": [],
                        "finding": (
                "SHAPE-FUNCTION NORMALISATION: this script builds a "
                "three-dimensional (hex) shape function with the factor 0.25, "
                "which is the QUAD4 value. For HEX8 it is 0.125, and with 0.25 "
                "the functions sum to 2 rather than 1 -- every sampled value is "
                "doubled AND, if the same N is used to invert the geometry, the "
                "point located is halved. Check sum(N) == 1 at any point.")})
        # `np.argmin(dist)` is also how a Stokes deck pins its pressure at the
        # domain centre, which is correct and unrelated. Require the argmin to
        # sit in a script that WRITES the deliverable CSV, and to be looking
        # up a field value, before calling it a sampling defect.
        writes_probe_csv = re.search(r"_level|probe", src, re.I)
        # A pressure PIN also uses argmin(distance) -- "pin the pressure at the
        # node nearest the centre" is correct, required in a Stokes problem,
        # and not sampling. Require the index to be used to READ A FIELD, and
        # exclude the pin idiom by name.
        looks_up_value = re.search(
            r"\[\s*(closest_id|nearest_idx|closest|nearest)\s*\]", src, re.I)
        is_pressure_pin = re.search(r"pin_p|pin_pressure|pressure_pin|pin_dof",
                                    src, re.I)
        if writes_probe_csv and looks_up_value and not is_pressure_pin and \
                re.search(r"argmin\(.*dist|closest_id|nearest[_ ]node", src, re.I) and \
                not re.search(r"probes\(|compute_colliding_cells|\.sample\(", src):
            out.append({"sequence": script.name, "values": [],
                        "finding": (
                "NEAREST-NODE SAMPLING: this script reads the value at the "
                "closest node instead of interpolating inside the element. That "
                "is a piecewise-constant reconstruction with O(h) error, and it "
                "CAPS your measured convergence order at 1 however good the "
                "solve is. Measured: nearest-node gives order 1.12, 1.00, 0.93 "
                "on a field where shape functions give 1.80, 2.03, 2.01.")})
    return out



_IFACE_COORD_NAMES = ("x", "y", "z")
_IFACE_FLUX_PREFIXES = ("q", "t")      # qn, q, tx, ty, tz, traction_x, ...


def _iface_column_diagnosis(path_a, path_b, ga, gb) -> str:
    """WHICH column disagrees, and whether it is a copy of another column of the same file.

    Measured (round 42, C1 7072): the two interface files agreed in ux and uy to 1e-6 and disagreed in T
    by a factor 40 -- side A's T column was its own uy column at every row (a wrong column index when the
    file was written). The finding said 'field jump 6.2e+01' and the parent read it as a physics error,
    wrote a warning into its summary and stopped with 13 minutes left. Names the column and the copy."""
    try:
        ha = [c.strip() for c in path_a.read_text(errors="replace").splitlines()[0].split(",")]
        hb = [c.strip() for c in path_b.read_text(errors="replace").splitlines()[0].split(",")]
        va, vb = ga[1], gb[1]
        n = min(len(va), len(vb))
        ncol = min(len(va[0]), len(vb[0])) if n else 0
        if n == 0 or ncol == 0:
            return ""
        ncoord = len(ga[0][0]) if ga[0] else 2
        names_a = ha[ncoord:ncoord + ncol] if len(ha) >= ncoord + ncol else [f"column {c + 1}" for c in range(ncol)]
        per = []
        for c in range(ncol):
            col_a = [va[i][c] for i in range(n)]; col_b = [vb[i][c] for i in range(n)]
            sc = max(max(abs(v) for v in col_a), max(abs(v) for v in col_b), 1e-300)
            per.append((names_a[c], max(abs(col_a[i] - col_b[i]) for i in range(n)) / sc))
        bad = [nm for nm, jp in per if jp > 1e-2]
        good = [nm for nm, jp in per if jp <= 1e-2]
        txt = " COLUMN BY COLUMN: " + "; ".join(f"{nm} {jp:.1e}" for nm, jp in per) + "."
        if bad and good:
            txt += f" The disagreement is in {', '.join(bad)} alone while {', '.join(good)} agree, so the coupling exchanged the right data and the FILE has the wrong column."
        # a column that equals another column of the same file, row for row
        for side, vals, hdr in (("A", va, ha), ("B", vb, hb)):
            names = hdr[ncoord:ncoord + ncol] if len(hdr) >= ncoord + ncol else [f"column {c + 1}" for c in range(ncol)]
            for c1 in range(ncol):
                for c2 in range(c1 + 1, ncol):
                    if all(abs(vals[i][c1] - vals[i][c2]) <= 1e-12 * max(1.0, abs(vals[i][c1])) for i in range(n)) and \
                       any(abs(vals[i][c1]) > 1e-300 for i in range(n)):
                        txt += (f" In side {side}'s file the {names[c1]} column equals its {names[c2]} column at every row: the same "
                                f"quantity was written twice -- fix the column index in the script that writes this file, not the coupling.")
        return txt
    except Exception:                                    # noqa: BLE001
        return ""


def _read_iface_by_header(path):
    """(points, values, fluxes) split by the file's OWN header names.

    The scalar heat layout x,y,u,qn is one case, not the definition: a
    thermo-mechanical interface carries x,y,T,ux,uy,qn,tx,ty, and reading it
    through the scalar layout takes ty for the flux. Coordinates are the
    leading x/y/z columns; flux components are the TRAILING run of columns
    whose names start with q or t (qn, tx, ty ...); everything between is the
    field trace. Header-less or unsplittable files fall back to the scalar
    reader so 4-column behaviour is unchanged. Returns None when unreadable.
    Note "T" (temperature) starts with t too -- which is why only the
    TRAILING run counts as flux: T sits before the displacement columns, so
    the trailing scan stops before it.
    """
    try:
        rows = [r for r in path.read_text(errors="replace").splitlines()
                if r.strip()]
        if len(rows) < 2:
            return None
        hdr = [c.strip().lower() for c in rows[0].split(",")]
        if any(ch.isdigit() for ch in rows[0].replace(",", " ").split()[0]):
            return None                        # no header line: caller falls back
        ncoord = 0
        for name in hdr:
            if name in _IFACE_COORD_NAMES:
                ncoord += 1
            else:
                break
        nflux = 0
        for name in reversed(hdr[ncoord:]):
            if name and name[0] in _IFACE_FLUX_PREFIXES and name not in ("t",):
                nflux += 1
            else:
                break
        if ncoord < 1 or nflux < 1 or ncoord + nflux >= len(hdr) + 1:
            return None
        nval = len(hdr) - ncoord - nflux
        pts, vals, flux = [], [], []
        for r in rows[1:]:
            parts = [c.strip() for c in r.split(",")]
            if len(parts) != len(hdr):
                return None
            nums = [float(c) for c in parts]
            pts.append(tuple(nums[:ncoord]))
            vals.append(nums[ncoord:ncoord + nval] or [0.0])
            flux.append(nums[ncoord + nval:])
        return pts, vals, flux
    except Exception:                          # noqa: BLE001
        return None


def _read_iface(path, _IF, want_flux=True):
    """Header-aware read with the scalar reader as the fallback."""
    got = _read_iface_by_header(path)
    if got is not None:
        return got
    try:
        g, _why = _IF.read_interface_csv(path, 2, 1, 1 if want_flux else 0)
    except Exception:                          # noqa: BLE001
        return None
    return g


def interface_sign_findings(work: Path) -> list[dict]:
    """The interface flux sign, from the agent's OWN files. Coupled problems
    only.

    WHY THIS IS IN THE AUDIT AND NOT ONLY IN A TOOL. The check was exposed as
    `verify_interface_flux`, described in the coupling must-read with the
    numbers from the runs it decided, and then called by ZERO of six runs in
    the next batch -- while the auto-audit on delivery reached five of those
    six. That is the same finding an earlier batch recorded for the audit
    itself: a calibrated check plus an instruction to run it was used by 1 of
    51 agents. Voluntary checking does not happen, so this one is not
    voluntary.

    What it asserts, needing no reference solution: for a flux really computed
    from your own solution, q_n(x) / (-du/dn)(x) equals k at every interface
    point, so the ratio is CONSTANT along the interface whatever k is -- and
    POSITIVE, because the task defines q_n with the OUTWARD normal.

    Measured on two real result sets, checked against an independent
    reference:
      one development run, complete but unphysical at order 1.2434 with
        BOTH prescribed codes proven to have run and the interface field
        matching to 0.000e+00: level 3 side B implied coefficient -250.8,
        against +198.1 and +200.3 at levels 1 and 2. It had the sign right on
        the coarse meshes and flipped it on the finest.
      the 4C+Kratos reference, verified correct at order 1.9796: all six sides
        positive, +0.98 to +1.30 where k = 1 and +200.4 to +206.7 where
        k = 200 -- the check recovering both conductivities from the
        delivered files alone.
    """
    import re as _re

    try:
        import sys as _sys
        _here = str(Path(__file__).resolve().parents[1])
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from blind_eval import interface as _IF
    except Exception:
        return []

    def _index(files):
        out = {}
        for f in files:
            if _side_of(f):
                out[(_level_of(f), _side_of(f).upper())] = f
        return out

    ifs, sols = _index(_interface_files(work, sided=True)), _index(_field_files(work))
    if not ifs:
        return []                       # not a coupled result set: say nothing

    inverted, assessed, jumps = [], 0, {}
    vector_layout = 0
    for (lvl, side), path in sorted(ifs.items()):
        gi = _read_iface(path, _IF)
        if gi is None:
            continue
        ipts, _giv, iq = gi
        # the k = q / (-du/dn) heuristic is defined for ONE scalar field and
        # ONE flux component; on a multi-component trace it pairs the wrong
        # columns, so it is skipped there (the ratio/mirror/zero branches
        # below handle every layout).
        if len(_giv[0]) != 1 or len(iq[0]) != 1:
            vector_layout += 1
            continue
        sp = sols.get((lvl, side))
        if sp is None:
            continue
        try:
            gs, _why = _IF.read_interface_csv(sp, 2, 1, 0)
            if gs is None:
                continue
            # THE GEOMETRY COMES FROM THE FILES, NOT FROM AN ASSUMPTION.
            # This call used to hard-code axis 0 (a vertical interface at
            # the first probe's x) and A-left/B-right. On a HORIZONTAL
            # interface it recovered du/dn along the wrong axis at a plane
            # that is not the interface and told a verified-correct
            # result set WRONG SIGN at every level -- and negating the flux
            # to obey it silenced the finding while corrupting the
            # result set (measured: implied k -0.4946 under the hard-coded
            # geometry; +1.060 and +2.092, both consistent, under the
            # files' own geometry). The interface axis is the coordinate
            # that is CONSTANT across the interface probes; the plane is
            # its value; the outward sign follows from which side of the
            # plane this side's own field points lie on.
            import numpy as _np
            _ip = _np.asarray(ipts, float)
            if _ip.ndim != 2 or _ip.shape[0] < 2:
                continue
            _axis = int(_np.argmin(_ip.var(axis=0)))
            _plane = float(_ip[:, _axis].mean())
            _sp_pts = _np.asarray(gs[0], float)
            _mean_side = float(_sp_pts[:, _axis].mean())
            sign = 1.0 if _mean_side < _plane else -1.0
            dudn = _IF.recover_normal_derivative(
                gs[0], gs[1], ipts, _axis, _plane, sign)
            res = _IF.flux_ratio_consistency(iq, dudn)
        except Exception:
            continue
        for comp in res.get("per_component") or []:
            k = comp.get("implied_coefficient")
            if isinstance(k, (int, float)):
                assessed += 1
                if k < 0:
                    inverted.append((lvl, side, k))
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga = _read_iface(a, _IF)
            gb = _read_iface(b, _IF)
            if ga and gb:
                # two_sided_jumps returns (dict, msg); it is None on a mismatch.
                jd = _IF.two_sided_jumps(ga, gb)[0]
                if jd is not None:
                    jumps[lvl] = jd.get("jump_q_rel")
        except Exception:
            pass

    out: list[dict] = []
    # ONE SIDE'S FLUX ORDERS OF MAGNITUDE BELOW THE OTHER'S = NO TRANSMISSION.
    #
    # The Neumann side's outward flux must be the NEGATIVE of the Dirichlet
    # side's, so the two magnitudes are equal to discretisation error. A side
    # reporting a flux a hundred times smaller has not received its partner's
    # data at all -- in Kratos, the usual cause is FACE_HEAT_FLUX set on the
    # interface nodes with no ThermalFace2D2N condition to integrate it, which
    # is silent: same exit code, same convergence message, and exactly the
    # no-flux field. Measured: 2.307291e-03 with the flux ignored against
    # 3.605675e-03 with it applied, bit-identical to the zero-flux run.
    peaks = {}
    for (lvl, side), path in sorted(ifs.items()):
        try:
            g = _read_iface(path, _IF)
        except Exception:
            g = None
        if g is None:
            continue
        vals = [abs(c) for row in g[2] for c in row]
        if vals:
            peaks[(lvl, side)] = max(vals)
    # BOTH SIDES ZERO IS THE ONE CASE THE FAMILY ABOVE CANNOT SEE.
    #
    # The one-side-smaller check needs big > 0, the same-convention ratio
    # needs a nonzero sum, and the sign check needs a nonzero implied
    # coefficient -- so an interface whose flux column is identically zero on
    # BOTH sides slips every one of them. Measured: a real result set carried
    # max|q| = 0.000e+00 on both sides at all three levels while its fields
    # were plausible, and nothing here spoke. Physically a partitioned
    # interface with zero flux everywhere transmitted nothing: the two
    # subdomains were solved as if insulated from each other.
    # AND BIT-EXACT OPPOSITION IS ITS MIRROR. Two runs in one round exported
    # side B's flux as side A's negated to the last bit -- max|qA+qB| exactly
    # 0.0 at every level over |q| up to 76 -- which slips the same three
    # branches from the other direction: the sum is zero, so the convention
    # ratio is silent; both peaks are nonzero, so the all-zero branch is
    # silent; the implied coefficient is plausible, so the sign branch is
    # silent. Two independently solved subdomains never agree to the last
    # bit: a coupling iterated to a 1e-6 relative tolerance leaves a jump of
    # about that size. Reading "the fluxes are equal and opposite" as an
    # instruction to CONSTRUCT one side from the other leaves the claim that
    # two codes met at the interface with no support at all.
    mirror_lvls = []
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga = _read_iface(a, _IF)
            gb = _read_iface(b, _IF)
            if not (ga and gb):
                continue
            qa = [c for row in ga[2] for c in row]
            qb = [c for row in gb[2] for c in row]
            n = min(len(qa), len(qb))
            if n and max(abs(qa[i]) for i in range(n)) > 0 and \
                    all(qa[i] + qb[i] == 0.0 for i in range(n)):
                mirror_lvls.append(lvl)
        except Exception:
            continue
    if mirror_lvls:
        out.append({"sequence": "interface flux constructed",
                    "values": mirror_lvls, "finding": (
            "SIDE B'S FLUX IS SIDE A'S NEGATED TO THE LAST BIT at level"
            + ("s " if len(mirror_lvls) > 1 else " ")
            + ", ".join(str(l) for l in mirror_lvls)
            + " (qA + qB is exactly 0.0 at every point). Two independently "
            "solved subdomains never agree to the last bit -- an iteration "
            "converged to a 1e-6 relative interface tolerance leaves a jump "
            "of about that size, not zero. 'Equal and opposite' is a "
            "statement about the CONVERGED PHYSICS, not an instruction to "
            "copy one side's column with a sign flip: computed this way, the "
            "file carries no evidence that the two solutions ever met at the "
            "interface. Compute each side's q_n = -(K grad u) . n_out from "
            "that side's OWN solution and its OWN material, and export what "
            "comes out; a small nonzero mismatch between the sides is the "
            "signature of a real coupling, not a defect to erase.")})
    zero_lvls = [lvl for lvl in sorted({l for l, _ in peaks})
                 if peaks.get((lvl, "A")) == 0.0 and peaks.get((lvl, "B")) == 0.0]
    if zero_lvls:
        out.append({"sequence": "interface flux all zero",
                    "values": zero_lvls, "finding": (
            "THE INTERFACE FLUX IS IDENTICALLY ZERO ON BOTH SIDES at level"
            + ("s " if len(zero_lvls) > 1 else " ")
            + ", ".join(str(l) for l in zero_lvls)
            + ". A coupled interface carries the flux that crosses it; a "
            "zero column on both sides means no transmission happened at "
            "all -- the two subdomains were solved as if insulated -- or the "
            "export never computed q_n = -(K grad u) . n_out from the "
            "solution. The field values themselves need no re-solve: recover "
            "the flux from your OWN existing solution (consistent nodal "
            "flux, or the gradient of your interpolant evaluated at the "
            "interface, times -K, dotted with the outward normal) and "
            "re-export. If the recovery also comes out zero, the two solves "
            "never exchanged data and the coupling itself did not run.")})
    for lvl in sorted({l for l, _ in peaks}):
        a, b = peaks.get((lvl, "A")), peaks.get((lvl, "B"))
        if a is None or b is None:
            continue
        big, small = max(a, b), min(a, b)
        if big > 0 and small < big / 50.0:
            weak = "A" if a < b else "B"
            out.append({"sequence": f"interface flux transmission level {lvl}",
                        "values": [a, b], "finding": (
                f"SIDE {weak}'S INTERFACE FLUX IS {big / max(small, 1e-300):.0f}x "
                f"SMALLER THAN ITS PARTNER'S at level {lvl} "
                f"({a:.4e} on A against {b:.4e} on B). The two sides' outward "
                f"fluxes must be equal and opposite, so this means side {weak} "
                f"never RECEIVED its partner's flux. In Kratos the usual cause "
                f"is FACE_HEAT_FLUX set on the interface NODES with no "
                f"ThermalFace2D2N condition on the interface EDGES to integrate "
                f"it: the nodal value is ignored, the solve exits 0, and you get "
                f"exactly the no-flux field (measured 2.307291e-03 ignored "
                f"against 3.605675e-03 applied, bit-identical to a zero-flux "
                f"run). Solve your Neumann side once with the flux zeroed and "
                f"once with it real -- if the fields match, it never arrived.")})
            break
    # BOTH SIDES ON THE SAME CONVENTION, TESTED WITHOUT A DERIVATIVE.
    #
    # The `inverted` branch below needs recover_normal_derivative, and on the
    # NEUMANN side that recovery is ill-conditioned: measured on the furthest
    # openPASO run of the coupled problem, side B's implied coefficient came out
    # None, +63.6 and -128.0 across the three levels, so only the last one
    # tripped `k < 0` and the finding named level 3 alone. Its field near the
    # seam is ~3e-3 with k = 200, which is why.
    #
    # The same defect has a signal that needs no derivative, no material
    # coefficient and no mesh -- only the two interface files. With opposite
    # normals |qA + qB| is discretisation error and |qA - qB| is ~2|q|; on the
    # same convention the two swap. Measured, sum/diff per level:
    #
    #     that run, both sides negative       89.2   272.4   1036.3
    #     a reference result set verified
    #       correct at order 1.9796            0.03    0.00     0.00
    #
    # Three orders of separation, and on the wrong convention the ratio GROWS
    # under refinement because its denominator is the shrinking discretisation
    # error while its numerator stays O(1). A threshold of 4 sits far from both.
    #
    # The run this comes from had already worked the rest out: its own report
    # says "Both sides report negative fluxes of similar magnitude (~0.8),
    # giving qn_A + qn_B = -1.6", and it called that "a persistent flux sign
    # convention issue [that] prevents completion" -- it read the defect as
    # physics to repair rather than a sign on a value being written out. So the
    # finding carries the line, not just the diagnosis.
    same_conv = []
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga = _read_iface(a, _IF)
            gb = _read_iface(b, _IF)
            if not (ga and gb):
                continue
            qa = [c for row in ga[2] for c in row]
            qb = [c for row in gb[2] for c in row]
            n = min(len(qa), len(qb))
            if n == 0:
                continue
            s = max(abs(qa[i] + qb[i]) for i in range(n))
            d = max(abs(qa[i] - qb[i]) for i in range(n))
            # d == 0 IS THE DEFECT AT ITS MOST BLATANT, NOT A REASON TO SKIP.
            #
            # The first version of this branch required d > 0 so the ratio
            # would be finite, which made it silent on the one case that needs
            # no interpretation at all: a flux column written IDENTICALLY into
            # both sides' files. Caught by its own test, on a synthetic pair
            # built to be exactly that. Reported with an infinite ratio.
            if s > 0 and (d == 0 or s > 4.0 * d):
                same_conv.append((lvl, (s / d) if d > 0 else float("inf")))
        except Exception:
            continue
    if same_conv:
        where = ", ".join(
            f"level {l} (|sum|/|difference| = "
            + ("identical, the difference is exactly zero)" if r == float("inf")
               else f"{r:.0f})")
            for l, r in same_conv)
        out.append({"sequence": "interface flux convention",
                    "values": [r for _l, r in same_conv], "finding": (
            "BOTH SIDES REPORTED THEIR FLUX WITH THE SAME SIGN at " + where
            + ". The two subdomains use OPPOSITE outward normals at the same "
            "physical point, so qA + qB must be zero to discretisation error "
            "and |qA - qB| must be about twice |q|. Here it is the other way "
            "round: the sum is the big number and the difference is tiny, "
            "which means both files carry the same physical quantity rather "
            "than each side's own outward flux. THIS IS ONE SIGN ON THE VALUE "
            "YOU WRITE OUT, not a defect in your solve -- your two fields "
            "already agree across the seam if the temperatures match. Negate "
            "the flux column of ONE side, the side whose normal you did not "
            "actually use:\n"
            "        qn_out = -qn_computed_with_the_other_sides_normal\n"
            "and leave the temperature column alone. On the Neumann side the "
            "flux you IMPORT and the flux you REPORT are opposite anyway, "
            "because Kratos's FACE_HEAT_FLUX is the INWARD flux, so if you "
            "wrote out what you applied you wrote the wrong sign. Check it by "
            "recomputing max|qA + qB| after the change: it must be small and "
            "must SHRINK from level to level, not grow.")})
    if inverted:
        where = ", ".join(f"level {l} side {s} (implied k = {k:.4g})"
                          for l, s, k in inverted)
        out.append({"sequence": "interface flux sign", "values": [], "finding": (
            "INTERFACE FLUX HAS THE WRONG SIGN at " + where + ". Your "
            "reported q_n is proportional to your own field's normal "
            "derivative but NEGATIVE, so it was computed with the INWARD "
            "normal. The task defines q_n = -(K grad u) . n_out with n_out "
            "pointing OUT of the subdomain. On the NEUMANN side the flux you "
            "IMPORT and the flux you REPORT are opposite -- Kratos's "
            "FACE_HEAT_FLUX is the inward flux -- so write the NEGATIVE of "
            "the value you applied. With the sign reversed the two sides "
            "appear to balance when they do not, and the observed order "
            "cannot see it. THE DEFECT IS IN YOUR RECOVERY FUNCTION, NOT IN "
            "THE LEVEL(S) NAMED ABOVE: the same code produced every level's "
            "flux, so after any fix RE-DERIVE AND RE-WRITE THE FLUX AT EVERY "
            "LEVEL AND BOTH SIDES, then re-run this audit. Measured: one run "
            "corrected only the level a warning named, left the others as "
            "they were, and the result set failed on a level it never "
            "re-checked.")})
    # A FIXED PROBE GRID HAS THE SAME ROW COUNT AT EVERY LEVEL.
    #
    # MEASURED, one development run: its coupling genuinely converged
    # (9.8e-07 in 21 iterations at the finest level) and its interface files
    # carry 9, 17 and 33 rows across the three levels -- its own mesh nodes,
    # which change under refinement -- against a contract that fixes the probe
    # points once for all levels. Everything it computed was thrown away on
    # the sampling. The signal needs no task knowledge: rows that GROW with
    # the level are a mesh trace; a fixed grid cannot do that.
    rowcounts: dict = {}
    for (lvl, side), path in sorted(ifs.items()):
        try:
            g = _read_iface(path, _IF)
        except Exception:
            g = None
        if g is not None:
            rowcounts.setdefault(side, {})[lvl] = len(g[0])
    for side, per in sorted(rowcounts.items()):
        ns = [per[l] for l in sorted(per)]
        if len(ns) >= 2 and len(set(ns)) > 1 and all(
                b > a for a, b in zip(ns, ns[1:])):
            out.append({"sequence": f"interface rows side {side}",
                        "values": ns, "finding": (
                f"INTERFACE ROWS GROW WITH THE LEVEL on side {side}: "
                + ", ".join(str(n) for n in ns) + " rows across the levels. "
                "The interface probe points are FIXED -- the same points at "
                "every mesh level -- so every per-level interface file "
                "must have the SAME rows in the same order. A growing count "
                "means you wrote your own mesh nodes instead of evaluating "
                "(interpolating) your solution AT the prescribed points. A "
                "run that did this had genuinely converged its coupling to "
                "9.8e-07 and counted for nothing on the sampling alone. "
                "Re-read the task's INTERFACE PROBE POINTS line and evaluate "
                "your existing solution there; no re-solve is needed.")})
            break
    # THE RESIDUAL YOU CONVERGED MUST BE THE DISAGREEMENT IN YOUR FILES.
    #
    # MEASURED, one development run: residual_level3.csv ends at 2.3162e-08
    # after 7 iterations, while the exported interface files disagree by
    # max|uA-uB| = 3.65e-03 -- IDENTICAL at all three levels -- and the flux
    # sum by ~0.8. Five orders between what the iteration measured and what
    # the result set contains means the iteration converged some OTHER
    # quantity (a different set of points, a previous iterate, one side's
    # internal state) than the fields that were written out. The observed
    # order was 0.1830 and nothing in the run said why.
    resid_final: dict = {}
    for f in _history_files(work):
        if _side_of(f):
            continue
        try:
            rows = [r for r in f.read_text(errors="replace").splitlines()
                    if r.strip()][1:]
            resid_final[_level_of(f)] = abs(float(rows[-1].split(",")[-1]))
        except Exception:
            continue
    mismatch = []
    columns: list = []
    for lvl, claimed in sorted(resid_final.items()):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b) or claimed <= 0:
            continue
        try:
            ga = _read_iface(a, _IF, want_flux=False)
            gb = _read_iface(b, _IF, want_flux=False)
            if not (ga and gb):
                continue
            ua = [v for row in ga[1] for v in row]
            ub = [v for row in gb[1] for v in row]
            n = min(len(ua), len(ub))
            if n == 0:
                continue
            scale = max(max(abs(v) for v in ua[:n]), 1e-300)
            jump = max(abs(ua[i] - ub[i]) for i in range(n)) / scale
            if jump > 100.0 * claimed and jump > 1e-3:
                mismatch.append((lvl, claimed, jump))
                columns.append(_iface_column_diagnosis(a, b, ga, gb))
        except Exception:
            continue
    if mismatch:
        where = ", ".join(f"level {l}: claimed {c:.2e} vs measured {j:.2e}"
                          for l, c, j in mismatch)
        col_txt = next((c for c in columns if c), "")
        out.append({"sequence": "residual vs files", "values":
                    [j for _l, _c, j in mismatch], "finding": (
            "THE RESIDUAL YOUR ITERATION CONVERGED IS NOT THE DISAGREEMENT "
            "IN YOUR FILES (" + where + "). The relative field jump computed "
            "from your own two per-level interface files is orders of magnitude "
            "above the final value in that level's residual history, so the quantity "
            "your coupling loop measured is not the quantity you exported -- "
            "a different point set, a stale iterate, or one side's internal "
            "state. A run with exactly this signature reported 2.3e-08 "
            "converged while its files disagreed by 3.65e-03 at every level. "
            "Recompute the mismatch FROM THE TWO FILES you are about to "
            "deliver -- max|uA-uB| over the interface rows, divided by "
            "max|uA| -- and iterate on THAT; if it does not match your "
            "loop's residual, your loop is reading different data than it "
            "writes." + col_txt)})
    trend = [jumps[l] for l in sorted(jumps)
             if isinstance(jumps.get(l), (int, float))]
    if len(trend) >= 2 and not all(trend[i + 1] < trend[i]
                                   for i in range(len(trend) - 1)):
        out.append({"sequence": "interface flux jump", "values": trend,
                    "finding": (
            "THE FLUX JUMP DOES NOT SHRINK under refinement (" +
            ", ".join(f"{v:.3e}" for v in trend) + "). A jump that stays "
            "O(1) as h halves means the iteration converged to a fixed "
            "point of the WRONG transmission condition, which a clean "
            "convergence order cannot reveal. Check the SIGN first.")})
    elif len(trend) >= 3 and all(trend[i + 1] < trend[i]
                                 for i in range(len(trend) - 1)):
        # THE JUMP SHRINKS, BUT TOO SLOWLY = A FIRST-ORDER INTERFACE
        # RECOVERY. On the prescribed halving a consistent (2nd-order)
        # recovery drops the two-sided flux jump ~4x per level; a jump that
        # only halves (~2x, order ~1) or worse means the flux one side
        # exports is not assembly-consistent with how it was applied --
        # the classic apply-with-one-quadrature / recover-with-another
        # mismatch (measured on a 4C-Neumann pair: applied Simpson nodal
        # loads on a QUAD4 solve, recovered by a P1-triangle re-assembly ->
        # jump fell only 1.5-1.8x per level, order ~0.8, and the coupled
        # field order was capped there). It converges cleanly to the wrong
        # fixed point, so neither the residual nor the sign check sees it.
        import math as _m
        orders = [_m.log2(trend[i] / trend[i + 1])
                  for i in range(len(trend) - 1) if trend[i + 1] > 0]
        med = sorted(orders)[len(orders) // 2] if orders else 0.0
        if med < 1.3:
            out.append({"sequence": "interface flux jump", "values": trend,
                        "finding": (
                "THE FLUX JUMP SHRINKS TOO SLOWLY (" +
                ", ".join(f"{v:.3e}" for v in trend) + f"; order ~{med:.2f} "
                "per halving, against ~2 for a consistent recovery). The "
                "flux one side exports is not assembly-consistent with how "
                "the partner applied it -- recover it with the SAME "
                "discretisation the solver used (the code's native boundary "
                "flux, or a re-assembly with the SAME element and quadrature "
                "as the solve), not a hand-rolled stand-in on a different "
                "element. A first-order interface recovery caps the coupled "
                "field's order however good the solves are.")})
    if not out and assessed == 0:
        if vector_layout:
            # Say WHY, accurately. Measured: a correct thermo-mechanical
            # result set with every prescribed file on disk was told to
            # "write solution_level<k>_<side>.csv" by this branch -- the
            # files existed; the sign heuristic had skipped every interface
            # because the trace is multi-component, and the message blamed
            # the wrong thing.
            out.append({"sequence": "interface flux sign", "values": [],
                        "informational": True,
                        "finding": (
                "THE SCALAR FLUX-SIGN HEURISTIC DOES NOT APPLY to this "
                "interface: its trace carries more than one field component, "
                "so k = q/(-du/dn) would pair the wrong columns and was not "
                "attempted. This is a statement of scope, not a defect in "
                "your result set: interface balance is still checked "
                "component-by-component against the two sides' files, and "
                "any finding from that check appears separately.")})
        else:
            out.append({"sequence": "interface flux sign", "values": [],
                        "finding": (
                "THE INTERFACE FLUX SIGN COULD NOT BE CHECKED: your interface "
                "files were read but the flux could not be compared with your "
                "own field. Write the per-level field file for every "
                "level and side on the prescribed probe grid, and this check "
                "becomes available. This is NOT a clean bill.")})
    return out


def export_findings(work: Path) -> list[dict]:
    """Two defects that live in the EXPORT, not the solve, and cap the outcome.

    A perfect solve reported through a broken export reads as badly as a
    wrong solve, and neither the solver log nor a refinement study can see it.
    Both of these were reproduced by execution against real result sets.

    (1) NEAREST-NODE SAMPLING INSTEAD OF INTERPOLATION. This one is real, and
        it is the largest single recoverable defect measured across the
        development runs: 99 runs with openPASO and 86 without carry the
        fingerprint. The tasks prescribe FIXED probe points that are
        deliberately not mesh nodes. Answering with the value at the closest
        node is O(h) accurate, so it caps the reported order at 1 however good
        the solve is. Measured on one 4C problem: the same solve gave order
        +1.9516 read by bilinear interpolation and +1.0179 read by nearest
        node -- and +1.0179 is exactly what the result set reported.

        The fingerprint is free: with a mesh of N cells per side, nearest-node
        sampling can only ever return (N-1)^2 + 1 distinct interior values, so
        1936 probe points collapse onto 50, 226 and 962 distinct values at
        N = 8, 16, 32. Measured on four result sets from two tasks -- all four
        show exactly 50/1936, 226/1936, 962/1936. A result set that
        interpolates shows 1908/1928/1936.

    (2) ROW ORDER IS **NOT** CHECKED, AND MUST NOT BE. It looks like a defect
        and is not one. An independent check pairs each delivered row with
        the reference EVALUATED AT THAT ROW'S OWN COORDINATES -- the shared
        field-error check does `for p, v in zip(pts, vals)` -- so a
        transposed file is read point by point exactly like an ordered one.

        Verified against an independent reference, not argued: the 4C+Kratos
        reference is correct at order 1.9796; the SAME result set with the row
        order transposed is correct at order 1.9796, bit-identical. A check on
        row order would have flagged 146 runs with openPASO and 117 without -- a
        third of the development runs -- and sent every one of them to fix
        something that costs nothing, spending the action budget that is
        already the binding constraint. It was written, measured, and removed.
    """
    import csv as _csv
    import math as _math
    import re as _re

    findings: list[dict] = []
    per_level: dict[int, tuple] = {}
    for f in _field_files(work):
        m = _re.search(r"level(\d+)", f.name)
        if not m:
            continue
        try:
            rows = [r for r in _csv.reader(f.open())
                    if r and not r[0].strip().startswith(("x", "#"))]
        except OSError:
            continue
        xs, ys, vals = [], [], []
        for r in rows:
            if len(r) < 3:
                continue
            try:
                xs.append(float(r[0])); ys.append(float(r[1]))
                vals.append(float(r[2]))
            except ValueError:
                continue
        if len(vals) < 100:
            continue
        key = int(m.group(1))
        # keep the largest file per level, so a stale partial does not decide
        if key not in per_level or len(vals) > len(per_level[key][2]):
            per_level[key] = (xs, ys, vals, f.name)

    # REQUIRE THE ARITHMETIC SIGNATURE, AT MORE THAN ONE LEVEL.
    #
    # The first version fired whenever distinct*2 < n at ANY level, and that is
    # far too loose. Measured against an independent reference on 25 runs of
    # one task carrying that looser flag: EIGHT of them are verified correct
    # at order 2.0045, 1.9884, 1.9754 and 1.8426. Their real distinct counts
    # are 7974-8585 of 9261 -- 86 to 93
    # per cent -- and the flag came from ONE coarse level collapsing to a single
    # value, which a genuine nearest-node export never does. Nearest-node
    # sampling collapses EVERY level, and it collapses them lawfully: on a mesh
    # of N cells per side it returns exactly (N-1)^2+1 distinct interior values.
    # Two runs hit 50, 226, 962 out of 1936 -- 7^2+1, 15^2+1,
    # 31^2+1 -- at all three levels.
    #
    # So the test is the exact signature, at two levels or more. Accusing eight
    # correct result sets to catch four defective ones is a worse trade than
    # missing some, and it is the same error as the row-order check that was
    # written, measured and deleted.
    hits = []
    for lvl, (xs, ys, vals, name) in sorted(per_level.items()):
        n = len(vals)
        distinct = len({round(v, 12) for v in vals})
        root = _math.isqrt(max(distinct - 1, 1))
        if distinct > 1 and root * root + 1 == distinct and distinct * 4 < n:
            hits.append((lvl, name, distinct, n))
    if len(hits) < 2:
        hits = []
    for lvl, name, distinct, n in hits:
        if True:
            # (N-1)^2+1 for the mesh that would explain it, reported so the
            # agent can recognise its own mesh
            nn = int(_math.isqrt(max(distinct - 1, 1))) + 1
            findings.append({"sequence": name, "values": [distinct, n],
                             "finding": (
                f"ONLY {distinct} DISTINCT VALUES ACROSS {n} PROBE POINTS at "
                f"level {lvl}. The probe points are deliberately NOT mesh "
                f"nodes, so a correct export gives almost {n} distinct values; "
                f"{distinct} is what NEAREST-NODE SAMPLING returns on a mesh of "
                f"about {nn} cells per side, because it can only ever produce "
                f"(N-1)^2+1 interior values. Nearest-node lookup is O(h), so it "
                f"CAPS YOUR REPORTED ORDER AT 1 however good the solve is. "
                f"Verified against an independent reference: one coupled run "
                f"gives order 1.9796 by interpolation, and the SAME SOLVE "
                f"re-exported by nearest-node lookup -- nothing else changed -- "
                f"comes out at order 0.9815. A separate case gave "
                f"+1.9516 interpolated against +1.0179 nearest-node, and +1.0179 "
                f"is exactly what that run reported. Interpolate inside "
                f"the element that CONTAINS each probe point; this is a "
                f"post-processing fix and does not need the solver re-run.")})
    return findings


def interface_ends_findings(work: Path) -> list[dict]:
    """Interface rows that reach the ends of the interface, before delivery.

    Physics, not contract: where a partitioned interface meets the outer
    boundary the split problem has a Dirichlet-Neumann corner, and the
    recovered flux there does not converge under refinement — measured in
    this corpus (2.11x -> 2.51x the true value over a 4x refinement).
    Prescribed probe sets therefore exclude the interface ends. A file
    whose interface rows run to the very ends of the interface is the
    signature of a SELF-CHOSEN uniform sampling of the whole interface —
    measured twice: two converged couplings counted for nothing because all
    44 of their rows sat at self-chosen coordinates covering the full span,
    including the excluded ends.
    """
    import re as _re
    dom = {}
    for q in _field_files(work):
        try:
            rows = q.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for ln in rows[1:5000]:
            parts = ln.split(",")
            try:
                x, y = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            for ax, v in ((0, x), (1, y)):
                lohi = dom.setdefault(ax, [v, v])
                lohi[0] = min(lohi[0], v); lohi[1] = max(lohi[1], v)
    if not dom:
        return []
    out = []
    for q in _interface_files(work, sided=True):
        try:
            rows = q.read_text(errors="replace").splitlines()[1:]
        except OSError:
            continue
        pts = []
        for ln in rows:
            parts = ln.split(",")
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue
        if len(pts) < 4:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        var_ax = 0 if (max(xs) - min(xs)) > (max(ys) - min(ys)) else 1
        vals = xs if var_ax == 0 else ys
        lo, hi = dom.get(var_ax, (None, None))
        if lo is None:
            continue
        span = hi - lo
        if span <= 0:
            continue
        sv = sorted(vals)
        gaps = [b - a for a, b in zip(sv, sv[1:]) if b > a]
        row_dx = (sorted(gaps)[len(gaps)//2] if gaps else 0.02 * span)
        margin = 0.5 * row_dx      # only rows essentially AT the extremes
        if min(vals) < lo + margin or max(vals) > hi - margin:
            out.append({"sequence": q.name, "informational": True, "values":
                        [min(vals), max(vals), lo, hi], "finding": (
                f"YOUR INTERFACE ROWS RUN TO THE ENDS OF THE INTERFACE "
                f"({min(vals):.4g}..{max(vals):.4g} against a domain span "
                f"{lo:.4g}..{hi:.4g}). The recovered flux at the points "
                f"where the interface meets the outer boundary does not "
                f"converge (Dirichlet-Neumann corner — measured 2.1x-2.5x "
                f"the true value, worsening under refinement), so "
                f"probe prescriptions typically EXCLUDE the ends. If your task "
                f"prints an interface probe formula, re-check these rows "
                f"against it verbatim; rows at the extremes have twice been "
                f"the signature of a self-chosen uniform sampling on runs "
                f"whose coupling itself was sound.")})
            break
    return out


def unlaunched_participants_findings(work: Path) -> list[dict]:
    """Participants built, coupling never run — stated at DELIVERY, not only
    at give-up.

    Measured: two sessions wrote exports-contract participant scripts (4 and
    6 of them), never drove the coupling iteration, and DELIVERED — so the
    give-up gate, which already states exactly this, never saw them. The
    statement is structural and belongs on every path that reads the
    workdir: scripts implementing the exchange exist, no partitioned
    iteration history exists anywhere, therefore the one step that turns
    this work into coupling evidence was never taken.
    """
    if _history_files(work):
        return []
    pscripts = []
    for q in sorted(work.rglob("*.py")):
        try:
            c = q.read_text(errors="replace")
        except OSError:
            continue
        if "exports.json" in c and ("imports.json" in c
                                    or "InterfaceData" in c):
            pscripts.append(q.name)
    if len(pscripts) < 2:
        return []
    if not _interface_files(work) and not any(
            work.rglob("exports.json")):
        return []            # no sign this workdir is a coupling at all
    return [{"sequence": "coupling never run", "values": [], "finding": (
        f"{len(pscripts)} script(s) implement the imports/exports participant "
        f"exchange ({', '.join(pscripts[:4])}) and no per-level residual "
        f"history exists anywhere in this directory. The "
        f"participants were built and the coupling iteration over them was "
        f"never run — a file set in this state cannot show two codes coupled. "
        f"Run the coupling over these participants; that single step is what "
        f"stands between the work on disk and a usable answer.")}]




def _halving_pair(n1: int, n2: int):
    """(a, b) with (a+1)(b+1) == n1 and (2a+1)(2b+1) == n2 -- the one way a
    2-D tensor grid of a x b cells halves into 2a x 2b -- or None. An even n2
    is not a halved tensor grid at all (2a+1 and 2b+1 are odd), so None there
    means "cannot say", not "wrong"."""
    if n1 <= 0 or n2 <= 0 or n2 % 2 == 0:
        return None
    p = 1
    while p * p <= n2:
        if n2 % p == 0 and (n2 // p) % 2 == 1:
            a, b = (p - 1) // 2, (n2 // p - 1) // 2
            if (a + 1) * (b + 1) == n1:
                return (a, b)
        p += 2
    return None


def _odd_grids(n: int) -> list:
    """The (cells_x, cells_y) tensor grids with n nodes, both counts even (a
    halved grid), squarest first; for naming what a node count reads as."""
    out = []
    p = 3
    while p * p <= n:
        if n % p == 0 and (n // p) % 2 == 1:
            out.append((p - 1, n // p - 1))
        p += 2
    return sorted(out, key=lambda ab: abs(ab[0] - ab[1]))


def exact_ladder_findings(work: Path) -> list[dict]:
    """A level that is provably not the halving of its neighbour, from the
    NDOF lines alone, where the same side proves it runs tensor grids.

    MEASURED on a coupled run whose every other verdict was clean and whose
    order still came out near 0.4. One side's run logs read NDOF 81, 255, 957.
    255 = 15 x 17 and 957 = 29 x 33 are the halvings of a 7 x 8 mesh (72
    nodes); 81 = 9 x 9 is an 8 x 8 mesh. No (a, b) satisfies (a+1)(b+1) = 81
    and (2a+1)(2b+1) = 255, so level 1 was not the mesh level 2 halves, and the
    order across levels 1-2 meant nothing. The band check above read the 3.15x
    step as fine: for coarse grids the +1 terms pull a true halving down to
    ~3.5x, and a band that admits that admits this. Over the nine CORRECT
    coupled cells of the two closed problems, 18 of 18 side-steps factor
    exactly; 9522's B1->2 is the one that does not.

    It leads the self-convergence finding (priority 55): when a level is off
    the ladder the order computed across it is meaningless, so "your interface
    recovery is first-order" is a misattribution -- measured on 9522, whose
    selfdiff read 1.32 and whose real defect is the 81-node level 1. It sits
    below the equation check (26): a field that does not solve its own PDE is
    the deeper defect.

    THE GUARD: a side is judged only where at least one of its own steps IS an
    exact halving -- proof that it runs tensor grids. An unstructured mesh's
    counts are arbitrary, its steps never factor, and it is never judged; the
    band check keeps covering it. Two levels only, one bad step, no proof:
    silent. Odd counts by chance cannot fire this: firing needs a consistent
    step beside the inconsistent one.
    """
    per_side: dict[str, dict[int, int]] = {}
    for q in _level_logs(work):
        try:
            txt = q.read_text(errors="replace")
        except OSError:
            continue
        nm = None
        for mm in _DOF_LINE.finditer(txt):
            nm = int(mm.group(1))
        if nm:
            lv = per_side.setdefault(_side_of(q), {})
            lv[_level_of(q)] = lv.get(_level_of(q), 0) + nm
    out = []
    for side, lv in sorted(per_side.items()):
        ks = sorted(lv)
        if len(ks) < 2:
            continue
        steps = [(a, b, _halving_pair(lv[a], lv[b])) for a, b in zip(ks, ks[1:])]
        if not any(pair for _a, _b, pair in steps):
            continue                       # nothing proves this side is a tensor grid
        who = f"side {side}" if side else "the run"
        for i, (a, b, pair) in enumerate(steps):
            if pair is not None or lv[b] % 2 == 0:
                continue
            reads = _odd_grids(lv[b])[:2]
            reads_txt = " or ".join(
                f"{cx}x{cy} cells ({cx + 1}x{cy + 1} nodes), which halves a "
                f"{cx // 2}x{cy // 2} mesh of {(cx // 2 + 1) * (cy // 2 + 1)} nodes"
                for cx, cy in reads) or "no tensor grid at all"
            # which of the two is the odd one out: the level whose OTHER
            # neighbour agrees with it is right, the other is wrong
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            prv = steps[i - 1] if i > 0 else None
            if nxt and nxt[2] is not None:
                cx, cy = nxt[2]           # level b is cx x cy cells
                wrong, want = a, f"{cx // 2}x{cy // 2} cells ({(cx // 2 + 1) * (cy // 2 + 1)} nodes), half of level {b}'s {cx}x{cy} in each direction"
            elif prv and prv[2] is not None:
                cx, cy = prv[2]           # level a is 2cx x 2cy cells
                wrong, want = b, f"{4 * cx}x{4 * cy} cells ({(4 * cx + 1) * (4 * cy + 1)} nodes), double level {a}'s {2 * cx}x{2 * cy} in each direction"
            else:
                wrong, want = None, None
            finding = (
                f"YOUR OWN LOGS PROVE ONE LEVEL IS OFF THE LADDER: on {who}, level "
                f"{b} has {lv[b]} nodes, which reads as {reads_txt}; level {a} has "
                f"{lv[a]} nodes. No mesh halves from {lv[a]} nodes to {lv[b]}: the "
                f"other steps on this side halve exactly, this one cannot. "
                + (f"Level {wrong} is the one off the ladder. Re-run level {wrong} alone -- "
                   f"one couple call -- on {want}, and rewrite level {wrong}'s field, "
                   f"interface and history files from that run."
                   if wrong is not None else
                   f"With only these two levels the logs cannot say which of the two is "
                   f"off; the task's own ladder decides. Re-run the wrong one alone -- one "
                   f"couple call -- and rewrite its field, interface and history files.")
                + " A convergence order across a step that is not a halving is not "
                "an order -- measured: a run whose every other verdict was clean "
                "reported an order near 0.4 for exactly this.")
            out.append({"sequence": f"exact ladder {who}", "priority": 50,
                        "values": [lv[k] for k in ks], "finding": finding})
    return out

def ndof_ladder_findings(work: Path) -> list[dict]:
    """The mesh ladder the agent's own logs imply, stated before delivery.

    Measured three times in one development stretch: converged couplings
    (proven to have run at every level) read as malformed because the levels
    were solved on a SELF-CHOSEN mesh ladder rather than the one the task
    prescribes. The signal was in the agent's own run logs the whole time:
    a halved mesh multiplies the DOF count by ~2^dim per level, so NDOF
    growth factors far from that reveal a non-halved ladder before any
    independent check sees it. No task parsing: this states what the files
    imply and what halving would imply, and leaves the comparison to the
    reader who holds the task sheet.
    """
    import re as _re
    # PER SIDE, WHERE THE LOGS ARE SIDED. This summed both sides' counts at a
    # level (`per_level[k] += nm`), and a sum can sit inside the band while
    # one side does not. Measured on a coupled run that was correct: side B went
    # NDOF 99 -> 255 between levels 1 and 2, 2.58x, below the band, while side
    # A went 54 -> 187 (3.46x); summed, 153 -> 442 is 2.89x and passed, and
    # B's own convergence order was 0.24. Unsided logs -- one code, one log
    # per level -- keep the summed reading, so a single-code run reads
    # exactly what it read before.
    per_side: dict[str, dict[int, float]] = {}
    for q in _level_logs(work):
        try:
            txt = q.read_text(errors="replace")
        except OSError:
            continue
        nm = None
        for mm in _DOF_LINE.finditer(txt):
            nm = int(mm.group(1))
        if nm:
            k = _level_of(q)
            lv = per_side.setdefault(_side_of(q), {})
            lv[k] = lv.get(k, 0) + nm
    ladders = {side: sorted(lv) for side, lv in per_side.items() if len(lv) >= 2}
    if not ladders:
        return []
    # The halving band is DIMENSION-AWARE, read from the agent's own files:
    # a 2D halving multiplies DOF by ~4, a 3D one by ~8. One loose band
    # admitted a 2D level that grew only 2.15x (a non-halved third level,
    # charged by the assessment) — measured the round after this check
    # landed.
    dim = 2
    for q in _field_files(work):
        try:
            lines = q.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        cols = [c.strip().lower() for c in lines[0].split(",")]
        if cols[:3] == ["x", "y", "z"]:
            dim = 3
            break
        if cols and all(c.replace(".","",1).replace("-","",1)
                        .replace("e","",1).replace("+","",1).isdigit()
                        for c in cols[:1]):
            continue        # headerless: try the next file for a header
        break
    lo, hi = (2.6, 6.0) if dim == 2 else (5.2, 12.0)
    factors: dict[str, list] = {}
    bad: list = []                     # (side, a, b, ndof_a, ndof_b, factor)
    for side, ks in sorted(ladders.items()):
        lv = per_side[side]
        steps = [(a, b, lv[b] / lv[a]) for a, b in zip(ks, ks[1:]) if lv[a] > 0]
        if not steps:
            continue
        factors[side] = [f for _a, _b, f in steps]
        bad += [(side, a, b, lv[a], lv[b], f) for a, b, f in steps
                if not (lo <= f <= hi)]
    if not factors or not bad:
        return []
    # NAME THE SIDE, THE LEVEL AND THE FIX. Measured on the honest build: a run
    # with three coupled levels, coupling proven, interface satisfied and both
    # codes proven read this finding twice (2.15x from level 2 to 3), and
    # still handed in -- the text told it to "re-check every level", not
    # which level was wrong or that one couple call on a doubled mesh would
    # have mended it.
    def _who(side: str) -> str:
        return f"side {side}" if side else "the run"
    named = "; ".join(
        f"{_who(sd)}: level {b} is NOT the halving of level {a} (NDOF {na:.0f} -> "
        f"{nb:.0f}, {f:.2f}x; halving gives ~{2 ** dim}x)"
        for sd, a, b, na, nb, f in bad)
    sd0, a0, b0, _na, _nb, _f = bad[0]
    _whose = "that side's" if sd0 else "its"
    fix = (f"Re-run level {b0} alone -- one couple call -- with {_who(sd0)}'s "
           f"mesh halving its level-{a0} h (double every cell count in "
           f"{_whose} config.json), and rewrite level {b0}'s field, interface "
           f"and history files from that run.")
    growth = "; ".join(
        (f"{_who(sd)} " if sd else "") + ", ".join(f"{f:.2f}x" for f in fs)
        for sd, fs in sorted(factors.items()))
    flat = [f for _sd, fs in sorted(factors.items()) for f in fs]
    return [{"sequence": "ndof ladder", "values": flat, "per_side": factors,
             "finding": (
        "YOUR OWN LOGS IMPLY A MESH LADDER THAT WAS NOT HALVED: NDOF per level "
        "grows by " + growth
        + f", while halving h multiplies the DOF count by ~4 in 2D and ~8 in "
        f"3D. {named}. {fix} A result set on a different ladder cannot be "
        "compared level-to-level however well it converged -- measured on "
        "runs whose coupling evidence was sound at every level and which "
        "were unusable for exactly this.")}]


def completeness_findings(work: Path, only_levels: set | None = None) -> list[dict]:
    """Members of the per-level x per-side deliverable set that are absent.

    `only_levels` narrows the expected set to the levels named, so the same
    body can be asked "what is missing for the level just finished?" while the
    ladder is still running. Left None nothing changes.

    Inferred from the agent's OWN files, no task parsing: the levels are
    every k seen in any *_level<k>* deliverable, the sides are every _A/_B
    suffix seen in any of them, and each family that uses sides is expected
    to have every (level, side) member once any of its members exists.
    MEASURED, the case this exists for: a run with coupling evidence at
    every level ended with 17 minutes of budget unused and two solution
    files never attempted -- nothing at delivery time enumerated the required
    set against the disk, and the auto-audit named quality defects but not
    absent files. A missing member makes the file set unusable (a missing
    subdomain file, a wrong level count), and it counts for nothing, the
    same as no result set at all.
    """
    import re as _re
    seen: dict[str, set] = {}
    levels: set[int] = set()
    sides: set[str] = set()
    for ext in ("csv", "log"):
        for q, kind, k, s in _level_files(work, ext):
            if ext == "csv" and _csv_role(q, kind) == "raw":
                continue
            fam = f"{kind}.{ext}"
            levels.add(k)
            if s:
                sides.add(s)
            seen.setdefault(fam, set()).add((k, s or None))
    if only_levels is not None:
        levels &= set(only_levels)
    if not levels or not seen:
        return []
    missing = []
    for fam, members in seen.items():
        kind, ext = fam.rsplit(".", 1)
        fam_sides = sorted({s for _, s in members if s})
        for k in sorted(levels):
            if not fam_sides:
                if not any(lv == k for lv, _ in members):
                    missing.append(f"{kind}_level{k}.{ext}")
            else:
                for s in fam_sides:
                    if (k, s) not in members:
                        missing.append(f"{kind}_level{k}_{s}.{ext}")
    if not missing:
        return []
    return [{"sequence": "deliverable completeness", "values": [],
             "finding": (
        "THE DELIVERABLE SET IS INCOMPLETE: judged only by the levels and "
        "sides your OWN files establish, these members are absent: "
        + ", ".join(missing[:8])
        + (f" (+{len(missing)-8} more)" if len(missing) > 8 else "")
        + ". A result set missing a level or a side is malformed and counts "
        "as unusable however good the present members are. For members "
        "whose level already has a run log or residual history, write "
        "them from the numbers you already have; for a level with NO "
        "run evidence, solve it or remove its stale files -- never "
        "invent members.")}]


# ═══════════════════════════════════════════════════════════════════════════
# THE ALWAYS-RUN CORRECTIVE FUNNEL
#
# Everything below is reachable from BOTH routes an agent actually drives: the
# live `couple` reply (which computes the flux/continuity checks from this run's
# own exports, before any file is written) and the on-delivery audit (which
# reads the per-level files). Each check reads ONLY the agent's own output —
# its exports, its solution_/interface_/residual_level*.csv, its logs — plus
# the PUBLIC task text where one is explicitly supplied. None of it reads the
# sealed key: every path here is rooted at the agent's own work dir or at two
# strings
# the agent itself passed in. The point is corrective prominence — every finding
# NAMES THE FIX, and what_to_fix_next() surfaces the single highest-priority one
# at the top of the reply, where a weak agent that merely "acts" will read it.
# ═══════════════════════════════════════════════════════════════════════════


def _iface_module():
    """The shared interface module, or None. Same import path the sign check
    uses, so this audit and any independent check share ONE definition of a
    jump."""
    try:
        import sys as _sys
        _here = str(Path(__file__).resolve().parents[1])
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from blind_eval import interface as _IF
        return _IF
    except Exception:                                       # noqa: BLE001
        return None


def _rows_of(arr) -> list[list[float]]:
    """Normalise an exports 'values'/'normal_fluxes' array to list[list[float]]
    (one row of components per interface point). Non-numeric entries drop the
    whole array rather than guess."""
    out: list[list[float]] = []
    if arr is None:
        return out
    try:
        for v in arr:
            if isinstance(v, (list, tuple)):
                out.append([float(x) for x in v])
            else:
                out.append([float(v)])
    except (TypeError, ValueError):
        return []
    return out


def _match_exports(a: dict, b: dict):
    """Pair two participants' exported interface data BY COORDINATE.

    Returns (uA, uB, qA, qB) as component-row lists over the shared points, or
    None when the two sides do not sample enough common points to compare
    (a non-matching interface — said elsewhere, not guessed at here)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    ca, cb = a.get("coordinates") or [], b.get("coordinates") or []
    if not ca or not cb:
        return None
    ua, ub = _rows_of(a.get("values")), _rows_of(b.get("values"))
    qa, qb = _rows_of(a.get("normal_fluxes")), _rows_of(b.get("normal_fluxes"))

    def _key(c):
        c = c if isinstance(c, (list, tuple)) else [c]
        try:
            return tuple(round(float(x), 9) for x in c)
        except (TypeError, ValueError):
            return None
    idxb = {}
    for j, c in enumerate(cb):
        k = _key(c)
        if k is not None:
            idxb.setdefault(k, j)
    mua, mub, mqa, mqb = [], [], [], []
    for i, c in enumerate(ca):
        j = idxb.get(_key(c))
        if j is None:
            continue
        if i < len(ua) and j < len(ub):
            mua.append(ua[i]); mub.append(ub[j])
        if i < len(qa) and j < len(qb):
            mqa.append(qa[i]); mqb.append(qb[j])
    if len(mua) < 2 and len(mqa) < 2:
        return None
    return mua, mub, mqa, mqb


def exchange_carried_nothing_finding(export_a: dict, export_b: dict,
                                     name_a: str = "A", name_b: str = "B"):
    """Did this converged coupling exchange any information at all?

    THE HOLE THIS FILLS, and why it lives here rather than in a participant.
    Both live content checks at this hook abstain on exactly this input:
    `flux_cancellation_finding` returns None when its denominator is zero, and
    `field_continuity_finding` does the same, deferring in a comment to "the
    near-zero check" -- which reads per-level CSVs that DO NOT EXIST YET when
    couple() runs. So a pair that exchanged nothing passed both.

    Measured on a live coupled run: three levels, both codes proven, the
    partitioned iteration converged in 7 steps to an interface residual of
    6.8e-11, and the traction columns read -3.04e-18 and 2.17e-19. Each side
    returned the answer it would have returned with no partner at all. A
    coupling that transmits nothing converges IMMEDIATELY and convincingly,
    because two sides exchanging nothing cannot disagree -- so every
    self-consistency measure the run reports about itself reads as success.

    WHY IT IS A FINDING AND NOT A REFUSAL. openPASO cannot know whether a
    subdomain is genuinely undriven; a free interface really can carry no
    traction. So this reports and never stops the run. The same restraint is
    why it belongs to the driver rather than to a participant's export
    self-check: that check ends the coupling when it fires, and on iteration 1
    imports.json is `{}` by contract, so a side legitimately recovers ~0 there.
    A fatal check must not fire on an input openPASO cannot interpret; a
    reporting one must.
    """
    m = _match_exports(export_a, export_b)
    if not m:
        return None
    ua, ub, qa, qb = m

    def _scale(rows_a, rows_b) -> float:
        vals = [abs(x) for r in (rows_a or []) for x in (r or [])]
        vals += [abs(x) for r in (rows_b or []) for x in (r or [])]
        vals = [v for v in vals if v == v]
        if not vals:
            return -1.0                       # absent, not zero: different defect
        return max(vals)

    _ROUNDOFF = 1e-14
    dead = [label for label, scale in (("interface value", _scale(ua, ub)),
                                       ("interface flux", _scale(qa, qb)))
            if 0.0 <= scale <= _ROUNDOFF]
    if not dead:
        return None
    both = len(dead) == 2
    return {"sequence": "interface exchange carried nothing",
            "priority": 28,
            "values": [0.0],
            "finding": (
        f"THIS COUPLING TRANSMITTED NOTHING: the {' and the '.join(dead)} "
        f"channel{'s are' if both else ' is'} identically zero on BOTH sides "
        f"({name_a} and {name_b}) at the matched interface points. Two sides "
        f"that exchange nothing cannot disagree, so the iteration converges at "
        f"once and the residual looks excellent while each side returns the "
        f"answer it would have returned with no partner at all — that is what "
        f"an interface residual of 1e-11 means here, not a satisfied "
        f"transmission condition. Check that the imported field is actually "
        f"read from imports.json and enters the assembled system (the "
        f"condition that integrates it is the usual omission), and that the "
        f"exported quantity is recovered from your own solution rather than "
        f"initialised and never written. If this subdomain is genuinely "
        f"undriven, say so in your report — openPASO cannot tell that from the "
        f"numbers and is not asserting otherwise."
    )}


def flux_cancellation_finding(export_a: dict, export_b: dict,
                              name_a: str = "A", name_b: str = "B"):
    """CLASS 1, live from the two exports: max|q_A + q_B| / max(|q_A|,|q_B|) at
    matched interface points. ~2.0 means the two outward fluxes ADD instead of
    cancelling — the exact-opposite-convention signature — and the fix is one
    sign on the value written out, not a re-solve. Returns a finding dict or
    None. Needs no reference: the two subdomains share the seam with OPPOSITE
    outward normals, so a correct coupling has q_A + q_B ~ 0 by construction."""
    m = _match_exports(export_a, export_b)
    if not m:
        return None
    _, _, qa, qb = m
    if len(qa) < 2 or len(qb) < 2 or not qa[0] or not qb[0]:
        return None
    ncomp = min(len(qa[0]), len(qb[0]))
    if not ncomp:
        return None
    num = 0.0
    for ra, rb in zip(qa, qb):
        for c in range(ncomp):
            num = max(num, abs(ra[c] + rb[c]))
    den = max((abs(x) for r in qa for x in r[:ncomp]), default=0.0)
    den = max(den, max((abs(x) for r in qb for x in r[:ncomp]), default=0.0))
    if den <= 0:
        return None
    ratio = num / den
    if ratio >= 1.5:
        return {"sequence": "interface flux cancellation", "priority": 30,
                "values": [ratio], "finding": (
            f"YOUR TWO INTERFACE FLUXES ADD INSTEAD OF CANCELLING: "
            f"max|q_{name_a}+q_{name_b}| / max|q| = {ratio:.2f} at the matched "
            f"interface points, and ~2.0 is the exact-opposite-convention "
            f"signature (both sides carry the SAME sign). The two subdomains "
            f"share the seam with OPPOSITE outward normals, so a correct "
            f"coupling has q_{name_a}+q_{name_b} ~ 0. FLIP THE SIGN OF ONE "
            f"SIDE'S EXPORTED OUTWARD FLUX: export q_n with respect to THAT "
            f"side's own outward normal, and on a Neumann side remember the "
            f"flux you IMPORT and the flux you REPORT are opposite. This is one "
            f"sign on the number you write out, not a defect in the solve. "
            f"Re-check max|q_A+q_B|/max|q| after the change — it must be small "
            f"and must SHRINK as you refine, not grow.")}
    return None


def field_continuity_finding(export_a: dict, export_b: dict,
                             name_a: str = "A", name_b: str = "B"):
    """CLASS 2, live from the two exports: max|u_A - u_B| / scale at matched
    interface points. Large => the two subdomains disagree at the seam, so the
    coupling has not PHYSICALLY converged whatever the iterate residual did.
    Skipped when the two sides declare DIFFERENT field names (a heterogeneous
    exchange — displacement against traction — where continuity of the trace is
    not the right statement), so an FSI-style pair is never false-charged."""
    fa = (export_a or {}).get("field_name")
    fb = (export_b or {}).get("field_name")
    if fa and fb and str(fa).strip().lower() != str(fb).strip().lower():
        return None
    m = _match_exports(export_a, export_b)
    if not m:
        return None
    ua, ub, _, _ = m
    if len(ua) < 2 or len(ub) < 2 or not ua[0] or not ub[0]:
        return None
    ncomp = min(len(ua[0]), len(ub[0]))
    if not ncomp:
        return None
    num = 0.0
    for ra, rb in zip(ua, ub):
        for c in range(ncomp):
            num = max(num, abs(ra[c] - rb[c]))
    scale = max((abs(x) for r in ua for x in r[:ncomp]), default=0.0)
    scale = max(scale, max((abs(x) for r in ub for x in r[:ncomp]), default=0.0))
    if scale <= 0:
        return None                         # near-zero owns this, not continuity
    rel = num / scale
    if rel < 0.25:
        return None
    return {"sequence": "interface field continuity", "priority": 35,
            "values": [rel], "finding": (
        f"YOUR TWO SUBDOMAINS DISAGREE AT THE INTERFACE: "
        f"max|u_{name_a} - u_{name_b}| / max|u| = {rel:.0%} of the field's own "
        f"scale at the matched interface points. THE COUPLING HAS NOT "
        f"PHYSICALLY CONVERGED -- a partitioned scheme is converged when the "
        f"two SIDES agree at the seam, and an iterate residual that fell to "
        f"your tolerance is NOT the same statement (it can fall to 1e-8 while "
        f"the two exported traces still disagree by ~100%). The quantity your "
        f"loop stops on must be computed FROM THE TWO TRACES it is about to "
        f"export -- max|u_A - u_B| over the shared interface points, divided by "
        f"max|u_A| -- and iterated until THAT is small. If it will not fall, "
        f"the two sides are enforcing different transmission conditions: check "
        f"the Dirichlet value one side APPLIES is exactly the trace the other "
        f"side EXPORTED (same points, same sign), not a stale or re-sampled "
        f"copy.")}


def interface_continuity_findings(work: Path) -> list[dict]:
    """CLASS 2 from the FILES: field continuity at the FINEST level, from the
    agent's own two per-level interface files, using the shared interface
    module's two_sided_jumps so a finding here mirrors the unsatisfied-
    interface outcome the run would meet after delivery. Fires only on a LARGE
    disagreement (>=25% of the field scale) so a correct-but-coarse level is
    never charged; the real small-model failures sit near 100%."""
    _IF = _iface_module()
    if _IF is None:
        return []
    import re as _re
    ifs: dict[int, dict[str, Path]] = {}
    for f in _interface_files(work, sided=True):
        ifs.setdefault(_level_of(f), {})[_side_of(f).upper()] = f
    worst = None
    for lvl in sorted(ifs):                     # coarse -> fine, keep the finest
        side = ifs[lvl]
        if len(side) != 2:
            continue
        sa, sb = sorted(side)
        ga = _read_iface(side[sa], _IF, want_flux=True)
        gb = _read_iface(side[sb], _IF, want_flux=True)
        if not (ga and gb):
            continue
        try:
            jd, _why = _IF.two_sided_jumps(ga, gb)
        except Exception:                                   # noqa: BLE001
            continue
        if not jd:
            continue
        ju = jd.get("jump_u_rel")
        if isinstance(ju, (int, float)) and ju == ju:
            worst = (lvl, float(ju))
    if worst is None or worst[1] < 0.25:
        return []
    lvl, ju = worst
    return [{"sequence": f"interface field continuity level {lvl}",
             "priority": 35, "values": [ju], "finding": (
        f"YOUR TWO SUBDOMAINS DISAGREE AT THE INTERFACE at level {lvl}: the two "
        f"exported traces differ by {ju:.0%} of the field's own scale "
        f"(max relative field jump, matched point-for-point). THE COUPLING HAS "
        f"NOT PHYSICALLY CONVERGED -- a partitioned scheme is converged when "
        f"the two SIDES agree, and an iterate residual falling to tolerance is "
        f"NOT the same statement (it can reach 1e-8 while the fields disagree "
        f"by ~100%, which is what this result set shows). Recompute the stop "
        f"criterion FROM THE TWO FILES you are about to deliver -- "
        f"max|u_A - u_B| over the shared interface rows, divided by max|u_A| -- "
        f"and iterate on THAT. If it will not fall, the Dirichlet value one "
        f"side applies is not the trace the other side exported: match them "
        f"point-for-point, same sign.")}]


def identical_solution_levels_findings(work: Path) -> list[dict]:
    """CLASS 3: <field>_level<i> == <field>_level<j> point-for-point => one
    mesh was saved to every level, so there is no refinement to measure. Skips
    a near-zero field (the near-zero check owns that) so the two are not both
    reported for the same files."""
    import re as _re
    groups: dict[str, dict[int, Path]] = {}
    for q in _field_files(work):
        side = _side_of(q).upper()
        lv = _level_of(q)
        prev = groups.get(side, {}).get(lv)
        # shallowest wins, so a build/ copy never shadows the real deliverable
        if prev is None or (len(q.relative_to(work).parts)
                            < len(prev.relative_to(work).parts)):
            groups.setdefault(side, {})[lv] = q
    out: list[dict] = []
    for side, byl in sorted(groups.items()):
        levels = sorted(byl)
        if len(levels) < 2:
            continue
        vecs: dict[int, list[float]] = {}
        for lv in levels:
            vv = _solution_value_vector(byl[lv])
            if vv is not None:
                vecs[lv] = vv
        ident = []
        lvs = sorted(vecs)
        for i in range(len(lvs)):
            for j in range(i + 1, len(lvs)):
                a, b = vecs[lvs[i]], vecs[lvs[j]]
                if len(a) != len(b) or len(a) < 4:
                    continue
                scale = max((abs(x) for x in a), default=0.0)
                if scale < 1e-8:
                    continue                # near-zero field: not this finding
                if all(abs(x - y) <= 1e-9 * scale for x, y in zip(a, b)):
                    ident.append((lvs[i], lvs[j]))
        if ident:
            tag = f" side {side}" if side else ""
            pairs = ", ".join(f"{i}&{j}" for i, j in ident)
            out.append({"sequence": f"identical solution levels{tag}",
                        "priority": 50, "values": [], "finding": (
                f"YOUR SOLUTION IS IDENTICAL ACROSS DISTINCT MESH LEVELS{tag} "
                f"(level pair(s) {pairs} agree point-for-point, the difference "
                f"is exactly zero). A refinement study measures how the answer "
                f"CHANGES as the mesh is refined, so identical levels carry no "
                f"order at all -- log2(|L1-L2|/|L2-L3|) is 0/0 -- and the study "
                f"counts as NOT RUN however correct each level is. You SAVED ONE "
                f"MESH TO ALL THE LEVELS. Run three DISTINCT meshes (the "
                f"prescribed coarsest, then halve, then halve again) and save "
                f"each level's OWN result: if you drove this through couple(), "
                f"each participant writes field_level<k>.csv per level -- use "
                f"those, one file per level, not a single file copied across. "
                f"Print the node/DOF count inside the solve at each level and "
                f"confirm it actually changes.")})
    return out


def _solution_value_vector(path: Path):
    """The concatenated non-coordinate columns of a solution CSV, in row order,
    for the identical-levels comparison. Requires a header (x,y[,z],...) so a
    headerless file is skipped rather than guessed at."""
    try:
        rows = [r for r in path.read_text(errors="replace").splitlines()
                if r.strip()]
    except OSError:
        return None
    if len(rows) < 2:
        return None
    first = rows[0].split(",")
    if not first or any(ch.isdigit() for ch in first[0]):
        return None                                     # headerless
    hdr = [c.strip().lower() for c in first]
    val_idx = [i for i, c in enumerate(hdr) if c not in ("x", "y", "z")]
    if not val_idx:
        return None
    vec: list[float] = []
    for r in rows[1:]:
        parts = r.split(",")
        if len(parts) < len(hdr):
            continue
        try:
            for i in val_idx:
                vec.append(float(parts[i]))
        except ValueError:
            return None
    return vec or None


def solution_rows_grow_findings(work: Path) -> list[dict]:
    """CLASS 4: a solution row count that GROWS with the level is a mesh trace,
    not a fixed probe grid. Every solution_level<k>.csv must carry the SAME
    prescribed probe points at every level; a growing count means the agent
    wrote its own mesh nodes instead of sampling the fixed points."""
    import re as _re
    per_side: dict[str, dict[int, int]] = {}
    for q in _field_files(work):
        side = _side_of(q).upper()
        lv = _level_of(q)
        try:
            n = sum(1 for _ in open(q, errors="ignore")) - 1     # minus header
        except OSError:
            continue
        cur = per_side.setdefault(side, {})
        if lv not in cur or n > cur[lv]:
            cur[lv] = n
    out: list[dict] = []
    for side, per in sorted(per_side.items()):
        ns = [per[l] for l in sorted(per)]
        if len(ns) >= 2 and len(set(ns)) > 1 and all(
                b > a for a, b in zip(ns, ns[1:])):
            tag = f" side {side}" if side else ""
            out.append({"sequence": f"solution rows grow{tag}", "priority": 55,
                        "values": ns, "finding": (
                f"YOUR SOLUTION ROW COUNT GROWS WITH THE LEVEL{tag}: "
                + ", ".join(str(n) for n in ns) + " rows across the levels. The "
                "task's SOLUTION PROBE POINTS are FIXED -- the same points at "
                "every mesh level -- so every per-level field file must carry "
                "the SAME rows in the same order. A count that grows with the "
                "mesh means you wrote your own MESH NODES instead of evaluating "
                "(interpolating) your solution AT the prescribed points. "
                "Re-read the task's probe-point list and sample your existing "
                "solution there; no re-solve is needed, and the points are "
                "deliberately NOT mesh nodes (interpolate inside the element "
                "that contains each one).")})
    return out


def pde_source_findings(pde_json: str, task_text: str = "") -> list[dict]:
    """OPTIONAL, public-only. Compares the agent's OWN declared source/equation
    string against the PUBLIC task source text it also supplied. Reads nothing
    sealed — both operands are strings the agent passed in. A silent wrong
    forcing (right shape, wrong function) converges cleanly to a different
    answer and no self-consistency check can see it, so this is the one place a
    declared/public mismatch can be named without any key.

    `pde_json` shape: {"A": {"source": "<what you implemented>",
                             "task_source": "<the task's stated source>"},
                       "B": {...}}  — task_text is a fallback task_source.
    """
    import json as _json
    import re as _re
    if not pde_json:
        return []
    try:
        spec = _json.loads(pde_json)
    except Exception:                                       # noqa: BLE001
        return [{"sequence": "declared pde", "informational": True, "finding": (
            "DECLARED PDE NOT CHECKED: the pde argument was not valid JSON. "
            "Pass {\"A\": {\"source\": \"...\", \"task_source\": \"...\"}} to "
            "have openPASO compare the forcing you implemented against the task's "
            "stated forcing (both PUBLIC strings you supply).")}]
    if not isinstance(spec, dict):
        return []

    def _norm(s: str) -> str:
        return _re.sub(r"\s+", "", str(s or "")).lower().replace("**", "^")

    out: list[dict] = []
    for side, d in spec.items():
        if not isinstance(d, dict):
            continue
        declared = d.get("source") or d.get("equation") or ""
        public = d.get("task_source") or task_text or ""
        nd, npub = _norm(declared), _norm(public)
        if not nd or not npub:
            continue
        if nd not in npub and npub not in nd:
            out.append({"sequence": f"declared source {side}", "priority": 45,
                        "finding": (
                f"THE SOURCE YOU DECLARED FOR SIDE {side} DOES NOT MATCH THE "
                f"TASK TEXT YOU SUPPLIED: you declared '{str(declared)[:120]}', "
                f"which does not appear in the task source "
                f"'{str(public)[:160]}'. Confirm you implemented the task's "
                f"actual source term/coefficient, not a paraphrase or a "
                f"placeholder -- a different forcing converges cleanly to a "
                f"different answer, and no self-consistency check can catch it. "
                f"(This compares only the two PUBLIC strings you provided; "
                f"openPASO reads nothing sealed.)")})
    return out


# Ordered high->low priority for the WHAT TO FIX NEXT lead. The FIRST row whose
# any-substring appears in a finding (or the finding's explicit `priority`)
# wins; lower rank = fix this FIRST = leads the reply. Matched on the STABLE
# uppercase headlines the findings already carry, so no existing finding has to
# be edited to be ranked.
_PRIORITY_TABLE = [
    (10, ("NOT COUPLED", "NEVER RECEIVED", "IDENTICALLY ZERO ON BOTH SIDES",
          "NEGATED TO THE LAST BIT", "SMALLER THAN ITS PARTNER",
          "TRANSMITTED NOTHING", "WAS NEVER RUN", "NEVER RUN",
          "PRODUCED BYTE-IDENTICAL", "NOT A FUNCTION OF ITS IMPORTS",
          "NOT COUPLED TO ITS PARTNER", "EXITED NON-ZERO", "TIMED OUT")),
    (12, ("NO LINE ANY SOLVER EMITS",)),
    (11, ("NAMES", "FILE(S) THAT DO NOT EXIST")),
    (15, ("NO PER-LEVEL FIELD FILE AND NO INTERFACE FILE",)),
    (20, ("COUPLING HISTORY TOO SHORT", "NON-POSITIVE OR NON-FINITE RESIDUAL",
          "RESIDUAL BARELY MOVED", "CONSTANT RESIDUAL COLUMN",
          "WRITTEN-IN SEQUENCE", "IDENTICAL RESIDUAL HISTORY")),
    (30, ("ADD INSTEAD OF CANCELLING", "SAME SIGN", "WRONG SIGN",
          "FAIL TO CANCEL", "SIGN-CONVENTION")),
    (35, ("DISAGREE AT THE INTERFACE", "FIELD CONTINUITY")),
    (40, ("IS NOT THE DISAGREEMENT", "YOU REPORT ",
          "SHRINKS TOO SLOWLY", "DOES NOT SHRINK", "INCONSISTENT WITH YOUR "
          "SOLVE", "NOT CANCELLING")),
    (45, ("DOES NOT MATCH THE TASK TEXT",)),
    (50, ("IDENTICAL ACROSS DISTINCT MESH LEVELS", "BIT-IDENTICAL")),
    (55, ("ROWS GROW WITH THE LEVEL", "ROW COUNT GROWS WITH THE LEVEL",
          "RUN TO THE ENDS OF THE INTERFACE", "DISTINCT VALUES ACROSS",
          "NEAREST-NODE")),
    (58, ("RUN LOG FROM THE WRONG LEVEL",)),
    (60, ("DELIVERABLE SET IS INCOMPLETE", "MISSING LEVEL", "DIFFERING COPY",
          "NO DOF-COUNT LINE", "MESH LADDER THAT WAS NOT HALVED",
          "SUMMARY FILE IS MISSING", "YOUR OWN NDOF IS")),
    (70, ("NEAR-ZERO FIELD", "FLOOR:")),
    (80, ("ORDER MISMATCH", "IMPROVE AT ONLY", "NON-MONOTONE", "FLUX JUMP",
          "FIRST-ORDER INTERFACE RECOVERY")),
]


def _rank(f: dict) -> int:
    p = f.get("priority")
    if isinstance(p, (int, float)):
        return int(p)
    t = " ".join((f.get("finding") or "").upper().split())
    for rank, subs in _PRIORITY_TABLE:
        if any(s in t for s in subs):
            return rank
    return 85


def what_to_fix_next(findings, *, converged: bool = True,
                     clean_msg: str | None = None) -> str:
    """The single leading line every route puts at the top of the reply.

    A weak agent reads the top of the message, not the thirty-first finding
    under a data dump, so the ONE highest-priority fix goes first, with its full
    corrective sentence. A clean funnel leads with the necessary-not-sufficient
    reminder — self-consistency cannot see a wrong-but-consistent answer.
    """
    real = [f for f in (findings or []) if not f.get("informational")]
    if clean_msg is None:
        clean_msg = (
            "your output is SELF-CONSISTENT (necessary, not sufficient) -- now "
            "check your FIELDS match the task: the right physics and boundary "
            "conditions, the prescribed interface/solution probe points, and "
            "the prescribed mesh levels. A self-consistency check reads only "
            "your own files and cannot see a wrong-but-consistent answer.")
    if not real:
        if not converged:
            return ("WHAT TO FIX NEXT: your coupling did NOT converge, so there "
                    "is no result yet -- a non-converged iteration is not a "
                    "solution. Fix convergence first (start relaxation at "
                    "theta=0.5, and confirm each side actually reads "
                    "imports.json and applies it), then re-run. Nothing "
                    "downstream matters until the iteration reaches tolerance.")
        return "WHAT TO FIX NEXT: " + clean_msg
    real.sort(key=lambda f: (_rank(f), str(f.get("sequence", ""))))
    top = real[0]
    others = real[1:]
    head = (f"WHAT TO FIX NEXT (highest priority of {len(real)} finding(s); "
            f"reads ONLY your own output files, never any answer key):\n"
            f"  >> {top.get('sequence', '')}: {top.get('finding', '')}")
    if others:
        head += ("\n\nTHEN, in priority order: "
                 + "; ".join(str(o.get("sequence", "")) for o in others[:8])
                 + (f" (+{len(others) - 8} more)" if len(others) > 8 else "")
                 + ". Full corrective text for each is in the findings list.")
    return head



def _fourc_deck_state(side: Path, work: Path) -> dict | None:
    """What 4C left in a participant's directory, as the next sub-step: which deck it refused
    (its own error lines, the defects named from the deck text) or which runs finished.
    None when the directory holds no deck and no 4C console yet."""
    try:
        from tools.fourc_deck_lint import side_dir_report   # noqa: PLC0415
        rep = side_dir_report(side)
    except Exception:                                       # noqa: BLE001
        return None
    if not (rep["decks"] or rep["errors"] or rep["finished"] or rep.get("tracebacks") or rep.get("consoles")):
        return None
    try:
        rel = str(side.relative_to(work))
    except ValueError:
        rel = side.name
    parts, what = [], []
    if rep["finished"]:
        fin = ", ".join(f"{p} ({', '.join(k)})" for p, k in rep["finished"].items())
        what.append(f"finished run(s): {fin}")
        if rep["defects"]:
            # a run that finished on a defective deck solved a different problem (4C drops a
            # condition on an undefined E id silently and reports 'finished normally')
            parts.append(f"the run(s) with output prefix {', '.join(rep['finished'])} finished (VTU on disk), but a "
                         "run whose deck is named below solved the WRONG problem and has to be re-run after the fix")
        else:
            parts.append(f"the run(s) with output prefix {', '.join(rep['finished'])} finished (VTU on disk) -- leave them")
    if rep["monitors"]:
        parts.append(f"{len(rep['monitors'])} reaction-monitor file(s) exist")
    for lg, said in rep["errors"].items():
        what.append(f"4C stopped ({lg})")
        parts.append(f"4C's own error in {lg}: {said}")
    for dk, why in rep["defects"].items():
        what.append(f"{dk}: {len(why)} defect(s)")
        parts.append(f"deck {dk}: " + "; ".join(why))
    for lg, tb in rep.get("tracebacks", {}).items():
        what.append(f"the participant itself stopped in Python ({lg})")
        parts.append(f"the participant's own Python stop in {lg}: {tb} -- fix that line first")
    for lg, tail in rep.get("consoles", {}).items():
        # a 4C console with neither a finish nor a recognised error: its last lines are the verdict
        what.append(f"4C's console {lg} ends without a finish")
        parts.append(f"the console {lg} shows neither 'finished normally' nor an error block; it ends with: {tail} "
                     "-- read that log from the top, the cause is above its last lines")
    if not rep["errors"] and not rep["defects"] and not rep.get("tracebacks") and not rep.get("consoles"):
        if rep["finished"]:
            what.append("no exports.json")
            parts.append("every 4C run finished and no defect is named, so the participant stopped in its "
                         "recovery or export: run it again and read ITS stderr from the top (the served "
                         "check names the missing output)")
        else:
            what.append("deck(s) written, no 4C console found")
            parts.append("no 4C console log lies next to the deck(s): run the binary line-buffered "
                         "(stdbuf -oL -eL <bin> <deck> <prefix> > <deck>.log 2>&1) and read the log from the top")
    brief = (f"the directory {rel} holds " + "; ".join(parts) + ". Fix exactly the named deck against "
             "the grammar (`4C -p`, prepare_simulation(solver='fourc', physics=...)), run "
             "check_input(solver='fourc', input_path=<the deck>) until it names no defect, then re-run that "
             "deck until its VTU folder appears; a deck that ran is not touched.")
    return {"what": "; ".join(what) + ".", "brief": brief}

def _side_operators(work: Path) -> dict:
    """Each side's operator, as that side's OWN config.json states it.

    The served contracts read ./config.json for the level keys and carry the
    subdomain's box, coefficient, reaction and source in the same file, so the
    operator is already on disk next to the participant that solved it. Nothing
    here comes from a task file or a key: it is the agent's own statement of
    the problem it implemented, which is what its own field can be judged
    against.
    """
    out = {}
    for cfg_path in sorted(work.glob("*/config.json")):
        try:
            cfg = json.loads(cfg_path.read_text() or "{}")
        except Exception:                                    # noqa: BLE001
            continue
        if not isinstance(cfg, dict):
            continue
        side = cfg_path.parent.name
        missing = [k for k in ("x0", "x1", "y0", "y1", "k") if k not in cfg]
        entry = {"dir": cfg_path.parent, "missing": missing}
        if not missing:
            try:
                entry["box"] = [(float(cfg["x0"]), float(cfg["x1"])),
                                (float(cfg["y0"]), float(cfg["y1"]))]
                entry["k"] = float(cfg["k"])
                entry["reaction"] = float(cfg.get("reaction") or 0.0)
            except (TypeError, ValueError):
                entry["missing"] = ["a numeric x0, x1, y0, y1 and k"]
        src = str(cfg.get("source_expr") or cfg.get("source") or "").strip()
        entry["source"] = src.replace("^", "**")
        if not src:
            entry["missing"] = entry["missing"] + ["source_expr"]
        out[side] = entry
    return out


def _per_level_field_sets(work: Path, box: list) -> list:
    """Sets of the agent's own CSVs that are per-level fields on `box`.

    Found by SHAPE, never by name: files whose names differ only in one integer
    are one set, the integer is the level, and a file joins the set only if its
    rows are (x, y, value) inside the box AND form a grid of that box's cell
    midpoints -- the arrangement the identity's quadrature is exact on. A
    participant's own mesh dump is uniform but sits on the cell corners, and is
    refused there (see pde_consistency._detect_midpoint_grid), which is the
    point: it is the file that must not be judged.
    """
    from .pde_consistency import _detect_midpoint_grid

    groups: dict = {}
    seen = set()
    for path in sorted(list(work.glob("*.csv")) + list(work.glob("*/*.csv"))):
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        nums = re.findall(r"\d+", path.name)
        if not nums:
            continue
        rows = []
        try:
            with path.open() as fh:
                for row in _csv.reader(fh):
                    try:
                        vals = tuple(float(c) for c in row)
                    except ValueError:
                        continue                              # header
                    if len(vals) >= 3:
                        rows.append(vals[:3])
        except OSError:
            continue
        if len(rows) < 4:
            continue
        pts = [r[:2] for r in rows]
        (lo_x, hi_x), (lo_y, hi_y) = box
        pad = 1e-9 + 1e-6 * max(hi_x - lo_x, hi_y - lo_y)
        if not all(lo_x - pad <= p[0] <= hi_x + pad and lo_y - pad <= p[1] <= hi_y + pad
                   for p in pts):
            continue
        weight, _why = _detect_midpoint_grid(pts, box)
        if weight is None:
            continue
        key = re.sub(r"\d+", "#", path.name)
        groups.setdefault(key, {})[int(nums[0])] = rows
    return sorted((g for g in groups.values() if len(g) >= 2),
                  key=lambda g: (len(g), len(next(iter(g.values())))), reverse=True)


def equation_findings(work: Path) -> list[dict]:
    """Does each side's delivered field satisfy the equation that side states?

    NOT VOLUNTARY, FOR THE REASON THE SIGN CHECK IS NOT. `verify_pde_consistency`
    is the only check that separates a field converging cleanly to the RIGHT
    function from one converging just as cleanly to a wrong one -- a refinement
    study cannot, and the run's own mesh-independence verdict fires on half the
    correct runs. Measured across every coupled cell on disk: 1118 runs, the
    tool called in ONE. Two wordings of the invitation were tried and measured
    (2 calls against 14 asks, then 0 against 7). So it is computed here, from
    files the agent already wrote, and reported whether or not anyone asks.

    It judges the field against the operator the AGENT'S OWN config.json
    states, so it is a self-consistency check and leaks nothing: no task file,
    no key, no reference solution. A side whose config does not state the
    operator is told which key is missing rather than guessed at.
    """
    out: list[dict] = []
    for side, op in sorted(_side_operators(work).items()):
        if op.get("missing"):
            out.append({
                "sequence": f"equation check side {side}", "values": [],
                "priority": 26, "informational": True,
                "finding": (
                    f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: "
                    f"its ./config.json does not state {', '.join(op['missing'])}. "
                    f"The served contract reads the subdomain box (x0, x1, y0, y1), "
                    f"the coefficient k, the reaction (0 when there is none) and "
                    f"source_expr from that file; with them this audit checks the "
                    f"delivered field against the equation you implemented, which is "
                    f"the one check that separates a field converging to the right "
                    f"function from one converging to a wrong one.")})
            continue
        sets = _per_level_field_sets(work, op["box"])
        if not sets:
            out.append({
                "sequence": f"equation check side {side}", "values": [],
                "priority": 26, "informational": True,
                "finding": (
                    f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: "
                    f"no per-level file of yours holds that side's field at the cell "
                    f"MIDPOINTS of its box {op['box']}. A participant's own mesh dump "
                    f"sits on the mesh nodes, and the check's quadrature is a midpoint "
                    f"rule, so it would measure the quadrature rather than the field. "
                    f"Write the per-level file your task asks for -- the field "
                    f"interpolated at the probe points it prescribes -- and this "
                    f"audit checks it.")})
            continue
        levels = {lvl: rows for lvl, rows in sorted(sets[0].items())}
        try:
            from .pde_consistency import check_levels
            res = check_levels(levels, op["source"], op["k"], op["box"],
                               reaction=op["reaction"]).as_dict()
        except Exception as exc:                              # noqa: BLE001
            out.append({"sequence": f"equation check side {side}", "values": [],
                        "priority": 26, "informational": True,
                        "finding": (f"SIDE {side}'S EQUATION CHECK COULD NOT RUN: "
                                    f"{type(exc).__name__}: {exc}")})
            continue
        resid = [l.get("relative_weak_residual") for l in res.get("levels", [])]
        verdict = str(res.get("verdict"))
        shown = ", ".join(f"{r:.3e}" for r in resid
                          if isinstance(r, float) and r == r)
        op_txt = (f"-div({op['k']:g} grad u)"
                  + (f" + {op['reaction']:g} u" if op["reaction"] else "")
                  + " = f")
        if verdict == "CONSISTENT":
            out.append({
                "sequence": f"equation check side {side}", "values": resid,
                "priority": 26, "informational": True,
                "finding": (
                    f"SIDE {side} SATISFIES ITS OWN EQUATION {op_txt} on the field you "
                    f"delivered: the weak residual falls {shown} across the levels, "
                    f"which is what a field solving the stated problem does. This is "
                    f"self-consistency, not a comparison with any reference: it says "
                    f"your field solves the equation your config states, with the "
                    f"source you gave it.")})
        elif verdict == "INCONSISTENT":
            out.append({
                "sequence": f"equation check side {side}", "values": resid,
                "priority": 5,
                "finding": (
                    f"SIDE {side}'S FIELD DOES NOT SATISFY THE EQUATION ITS OWN CONFIG "
                    f"STATES ({op_txt}): the weak residual is {shown} across the "
                    f"levels -- it does not fall, and a field that solves the stated "
                    f"problem drives it down by roughly four per refinement. Refining "
                    f"will not fix this and your convergence study cannot see it: the "
                    f"field is converging cleanly to a different function. Check the "
                    f"source term you implemented against the one your task states, "
                    f"the coefficient, and which subdomain each belongs to.")})
        else:
            why = "; ".join(str(l.get("detail", ""))[:200] for l in res.get("levels", [])[:2])
            out.append({
                "sequence": f"equation check side {side}", "values": resid,
                "priority": 26, "informational": True,
                "finding": (f"SIDE {side}'S EQUATION CHECK RETURNED {verdict}: {why}")})
    return out


def wrong_level_run_log_findings(work: Path) -> list[dict]:
    """A run log that is a copy of ANOTHER level's console. Measured (round 29): four cells coupled
    three refined levels (consoles 54, 187, 693 dofs) and handed in run logs reading 54 at every
    level -- graded as an unchanged mesh. Compares each run_level<k>_<side>.log's NDOF line with the
    side's captured participant_output_level<k>.log and names the file to copy."""
    import re as _re
    out = []
    # the grader reads the FIRST canonical `NDOF = <n>` line of a run log; so does this check
    dof_any = _re.compile(r"^\s*NDOF\s*=\s*(\d+)\s*$", _re.M)
    consoles = {}
    for q in work.rglob("participant_output_level*.log"):
        if _SCRATCH & set(q.relative_to(work).parts[:-1]):
            continue
        m = _re.search(r"participant_output_level(\d+)\.log$", q.name)
        if not m:
            continue
        side_dir = q.parent.name
        try:
            vals = dof_any.findall(q.read_text(errors="ignore"))
        except OSError:
            continue
        if vals:
            consoles.setdefault(int(m.group(1)), {})[side_dir] = (vals[0], q)
    if not consoles:
        return out
    for q in _level_logs(work):
        m = _LEVEL_FILE.match(q.name)
        if not m or not m.group("side"):
            continue
        k, side = int(m.group("k")), m.group("side")
        try:
            vals = dof_any.findall(q.read_text(errors="ignore"))
        except OSError:
            continue
        if not vals or k not in consoles:
            continue
        # the side's console dir: a directory whose name ends with the side letter (side_A, A, sideA ...)
        match = [(d, v) for d, v in consoles[k].items() if d.lower().rstrip("/").endswith(side.lower())]
        if len(match) != 1:
            continue
        d, (ndof, cpath) = match[0]
        if vals[0] != ndof:
            other = [j for j, sides in consoles.items() if sides.get(d, (None,))[0] == vals[0]]
            out.append({"sequence": f"run log level {k} side {side}", "values": [], "priority": 20,
                        "finding": (f"RUN LOG FROM THE WRONG LEVEL: {q.relative_to(work)} carries NDOF {vals[0]} while "
                                    f"side {side}'s captured console for level {k} says NDOF {ndof}"
                                    + (f" -- it is level {other[0]}'s console" if other else "")
                                    + f". Copy {cpath.relative_to(work)} over it verbatim; a run log from another level "
                                    "reads as an unchanged mesh and sinks the whole mesh sequence.")})
    return out


_WORKER_SEES_ONLY_THE_BRIEF = (  # kept for reference; the briefs now carry a <...> block instead
    "THE WORKER SEES ONLY THIS BRIEF, NOT YOUR TASK: paste that subdomain's data from your task into it verbatim -- geometry and interface position, equations and coefficients, source terms as written, boundary values, the level-1 mesh, and the file names your task prescribes for this side. ")


def coupled_ladder(work: Path) -> dict | None:
    """The next unmet step of a partitioned coupled run, read from the files,
    phrased as the brief of the sub-agent that should do it.

    A small model that sees the whole job at once judges it too big and gives
    up (measured: 8 of 9 runs in one round, most with 25 minutes left); given
    ONE step whose end is a file it can see, it keeps the served contract and
    finishes the step (measured in single-step trials: 6 of 6 kept it against
    0 of 45 whole-problem runs). So every audit and every couple() reply names
    the next unmet step -- two participant scripts -> each exports standalone
    -> a coupling history per level -> that level's field and interface files
    for both sides -> that level's captured run logs -> the summary -- and
    returns it as a brief to hand to a sub-agent. No task knowledge: every
    check reads the agent's own files; the level count is the task's.
    """
    import re as _re
    try:
        from blind_eval.evidence import (PER_CODE_SIGNATURES,
                                         strip_terminal_noise)
        sig_pats = [pp for pl in PER_CODE_SIGNATURES.values() for pp in pl]
    except Exception:                                   # noqa: BLE001
        sig_pats, strip_terminal_noise = [], (lambda t: t)
    scripts, exports = [], []
    for q in work.rglob("*"):
        if not q.is_file():
            continue
        try:
            rel = q.relative_to(work)
        except ValueError:
            continue
        if _SCRATCH & set(rel.parts[:-1]):
            continue
        if q.suffix == ".py":
            try:
                t = q.read_text(errors="ignore")
            except OSError:
                continue
            if "imports.json" in t and "exports.json" in t and not _re.search(
                    r"run_|driver|coupl|orchestr|main_", q.name, _re.I):
                scripts.append(q)          # a participant, not the driver that launches them
        elif q.name == "exports.json":
            try:
                e = json.loads(q.read_text())
                if e.get("normal_fluxes") or e.get("values"):
                    exports.append(q)
            except Exception:                           # noqa: BLE001
                continue
    hist = _history_files(work)
    if not (scripts or exports):
        return None                    # no sign of a partitioned coupling here
    fields = _field_files(work)
    ifaces = _interface_files(work, sided=True)
    logs = _level_logs(work)

    def step(n, what, brief, as_is=True):
        hand = ("HAND THIS STEP TO A SUB-AGENT AS IS" if as_is else
                "HAND THIS STEP TO A SUB-AGENT WITH THE <...> BLOCK REPLACED BY YOUR TASK'S OWN WORDS -- the worker "
                "sees nothing else (measured: a worker briefed without them wrote placeholder source terms)")
        return {"step": n, "text": (
            f"LADDER STEP {n} OF 6 -- {what}\n"
            f"{hand}: spawn_subagent(role='worker', "
            f"task=\"{brief}\"). The step ends when its check passes on disk; "
            f"the sub-agent reports the exact error otherwise. Judge nothing about "
            f"the whole task -- only this step."), "brief": brief}

    if len(scripts) < 2:
        have = (f"{len(scripts)} participant script(s) found"
                + (f" ({scripts[0].relative_to(work)})" if scripts else ""))
        return step(1, f"WRITE THE {'SECOND' if scripts else 'FIRST'} PARTICIPANT: {have}.",
                    "Write the participant script of ONE code for ONE subdomain in its own directory ./side_<x>: "
                    "call knowledge(topic='coupling', solver=<that code>) and copy the served CONTRACT into "
                    "the file unchanged (imports.json handshake, sign convention, flux recovery, exports "
                    "schema, export self-check); fill only its marked hole(s) with the mesh, form, material, "
                    "source and solve for this subdomain from THIS DATA, which is all you know of the task: "
                    "<THAT SUBDOMAIN, COPIED FROM YOUR TASK WORD FOR WORD: geometry and interface position; equations and "
                    "coefficients; source terms as written; boundary values; the level-1 mesh; the file names the task "
                    "prescribes for this side>. Write ./config.json for level 1 and a "
                    "synthetic ./imports.json; if the code takes an input deck, run check_input(solver=<that code>, "
                    "input_path=<the deck>) until it names no defect before the binary runs; run the script with "
                    "that code's own interpreter until ./exports.json appears with finite values. CHECK: "
                    "./side_<x>/exports.json exists and the script exited 0.", as_is=False)
    # A PARTICIPANT WRITTEN FROM SCRATCH IS THE NEXT STEP, NOT A DETAIL. Every served
    # contract carries at least one of these lines; a script with none of them was
    # not copied from the door. Measured on three cells: a Neumann side that put
    # zeros in every point load and reported the code could not apply the flux, a
    # Kratos side whose flux condition carried no nodal value and coupled as
    # unresponsive, a thermo-elastic side with no handshake at all. The served
    # self-checks would have refused each export. Fires only before the first
    # converged level: a coupling that already converged a level has working scripts.
    _served_marks = ("EXPORT SELF-CHECK ─ keep this block", "exports.json LAST",
                     "CONTRACT (do not change)", "openPASO DOES NOT SERVE THIS")
    _converged_any = False
    for q in hist:
        try:
            if sum(1 for _ in q.open(errors="ignore")) - 1 >= 3:
                _converged_any = True
                break
        except OSError:
            continue
    if not _converged_any:
        unserved = []
        for q in scripts:
            try:
                t = q.read_text(errors="ignore")
            except OSError:
                continue
            if not any(m in t for m in _served_marks):
                unserved.append(q)
        if unserved:
            who = ", ".join(str(q.relative_to(work)) for q in unserved[:2])
            return step(1, f"RESTORE THE SERVED CONTRACT IN {who}: the file carries none of the served "
                           f"contract's lines, so it was written from scratch or rewritten instead of copied "
                           f"(measured: such scripts put zeros in every interface load, or coupled as "
                           f"unresponsive, and reported the code could not do it).",
                        f"For {who}: call knowledge(topic='coupling', solver=<that code>) (add "
                        "physics='thermoelastic' when the interface carries temperature and displacement "
                        "together), copy the served CONTRACT for this side's role into the file UNCHANGED, and "
                        "move only your mesh, deck/form, material, source and solve into its marked hole(s); keep "
                        "every served line, including the EXPORT SELF-CHECK block. Then write ./config.json for "
                        "level 1 and a synthetic ./imports.json and run it with that code's own interpreter until "
                        "./exports.json appears. CHECK: the file contains the served EXPORT SELF-CHECK block "
                        "and the script exited 0 with ./exports.json beside it.")
    dirs_with_export = {q.parent for q in exports}
    unrun = [q for q in scripts if q.parent not in dirs_with_export]
    if len(dirs_with_export) < 2 and unrun:
        who = ", ".join(str(q.relative_to(work)) for q in unrun[:2])
        # A 4C SIDE THAT ALREADY RAN THE BINARY: the step is the deck 4C refused, not the
        # whole participant. The side directory holds 4C's own verdict -- its error block
        # in the captured console, the VTU folder of every run that finished, the reaction
        # monitor files -- and the deck defects openPASO can name from the deck text. Put
        # them in the brief, so the worker starts from the defect (measured: a worker
        # handed the whole step again rewrote the deck from scratch and hit the same
        # section a second time). Reads only the agent's own files; writes nothing.
        deck_state = _fourc_deck_state(unrun[0].parent, work)
        if deck_state:
            return step(2, f"MAKE THE 4C DECK RUN in {who}: {deck_state['what']}",
                        f"In the directory of {who}: {deck_state['brief']} Then run the participant again "
                        "with its interpreter. CHECK: ./exports.json appears beside the script with finite "
                        "values and the script exited 0.")
        return step(2, f"RUN EACH PARTICIPANT STANDALONE: {who} has not exported yet.",
                    f"In the directory of {who}: write a synthetic ./imports.json (the partner's field name, "
                    "coordinates along the interface, values and normal_fluxes -- plausible numbers) and "
                    "./config.json for level 1; run the script with that code's own interpreter and a "
                    "generous timeout (first runs compile); read the traceback from the top, fix the script, "
                    "repeat. CHECK: ./exports.json appears beside the script with finite values and the "
                    "script exited 0.")
    rows_by_level: dict[int, int] = {}
    for q in hist:
        k = _level_of(q)
        if k is None:
            continue
        try:
            n = sum(1 for _ in q.open(errors="ignore")) - 1
        except OSError:
            n = 0
        rows_by_level[k] = max(rows_by_level.get(k, 0), n)
    done_levels = sorted(k for k, n in rows_by_level.items() if n >= 3)
    for k in done_levels:
        sides_f = {_side_of(q) for q in fields if _level_of(q) == k and _side_of(q)}
        sides_i = {_side_of(q) for q in ifaces if _level_of(q) == k}
        if len(sides_f) < 2 or len(sides_i) < 2:
            # A COUPLED LEVEL WITHOUT ITS PER-LEVEL DUMPS IS RE-RUN, NOT INTERPOLATED FROM NOTHING. Measured
            # (round 41, C2 7123): level 1's scripts wrote no interface_level1.csv, the deliverables pass skipped
            # that level's interface files with a warning, and three second-order levels were read as no result.
            _gap = []
            for _sd in sorted({q.parent for q in scripts}):
                try:   # only a side whose script carries the served dump block is expected to leave dumps
                    if not any("field_level" in q.read_text(errors="ignore") for q in _sd.glob("*.py")):
                        continue
                except OSError:
                    continue
                for _stem in (f"field_level{k}", f"interface_level{k}"):    # the served name or the agent's suffixed variant
                    if not any(_sd.glob(f"{_stem}*.csv")):
                        _gap.append(f"{_sd.name}/{_stem}*.csv")
            if _gap:
                return step(3, f"LEVEL {k} WAS COUPLED BUT ITS PER-LEVEL DUMPS ARE MISSING ({', '.join(_gap[:4])}): the deliverables "
                               f"cannot be written from them; the participant that ran level {k} did not carry the served dump block.",
                            f"In {', '.join(sorted({str(q.relative_to(work)) for q in scripts}))}: make sure the served block that writes "
                            f"field_level<k>.csv and interface_level<k>.csv after exports.json is present and unchanged (it is part of the "
                            f"contract), then couple() level {k} again with the same participants so its dumps appear; CHECK: both sides' "
                            f"field_level{k}.csv and interface_level{k}.csv exist next to their exports.json.")
            # LEVELS FIRST, DELIVERABLES ONCE -- the ladder says the same as the couple() lead, or the
            # parent copies the ladder's ready-made spawn call and writes level 1's files (measured
            # 2026-09-12: 4 of 4 parents did exactly that while the lead above asked for couple_levels;
            # round 40's C3 6041 lost its third level to the same order). The step carries a <...>
            # block for the levels the task prescribes: the worker sees only the brief.
            nxt = max(done_levels) + 1
            # A SIDE WITHOUT LEVEL-TAGGED OUTPUT IS PRESERVED FIRST, INSIDE THE READY-MADE CALL. Measured
            # (round 46, C1 7191; parent test 2026-09-13, 2 of 4): a warning after the one-call text did
            # not move the parent; the spawn call it copies must carry the copy step itself.
            _untagged = []
            for _sd in sorted({q.parent for q in scripts}):
                try:
                    _has = [q.name for q in _sd.glob(f"*level{max(done_levels)}*") if not q.name.startswith("participant_output")]
                except OSError:
                    _has = ["?"]
                if not _has:
                    _untagged.append(_sd.name)
            _preserve = ("" if not _untagged else
                         f"FIRST, in {' and '.join(_untagged)}: this level's outputs carry no level tag (no file named with 'level{max(done_levels)}' "
                         f"beside exports.json), and the next level's run reuses the same names and overwrites them -- copy this level's field "
                         f"file(s), VTU folder and interface data to names containing 'level{max(done_levels)}' NOW (or add the served contract's dump "
                         f"block, which writes field_level<k>.csv and interface_level<k>.csv after exports.json); measured: a side whose runs shared "
                         f"one output prefix handed in identical solution files for every level and no interface file. THEN: ")
            return step(4, f"LEVEL {k}'S DELIVERABLES ARE NOT WRITTEN (coupled so far: levels {done_levels}; field files for "
                           f"{sorted(sides_f) or 'no'} side(s), interface files for {sorted(sides_i) or 'no'} side(s)). If your task "
                           f"prescribes levels beyond {max(done_levels)}, they come FIRST in ONE couple_levels call -- a level set with "
                           f"one level missing is no result at all -- and the deliverables of ALL levels follow in ONE pass.",
                        _preserve + f"<IF THE TASK PRESCRIBES LEVELS BEYOND {max(done_levels)}, START WITH THIS ONE CALL: "
                        f"couple_levels(participants=<the same list passed to couple>, levels=[{nxt}, ...every further level the "
                        f"task prescribes], history_pattern='<the task's per-level history file name with {{k}} in place of the level "
                        f"number>') -- every level keeps field_level<k>.csv, interface_level<k>.csv and participant_output_level<k>.log "
                        f"next to each exports.json; replace this block with the levels and file name, or delete it if level "
                        f"{max(done_levels)} is the last.> THEN, for EVERY coupled level k and EACH side, in ONE script that loops over "
                        "the levels: read that side's converged field (its per-level dumps "
                        "field_level<k>.csv, columns x,y,<field> nodal values, and interface_level<k>.csv, "
                        "columns x,y,<trace>,<flux> at its interface nodes -- NEVER exports.json, which the next "
                        "level overwrites: measured, a run that rebuilt level 1 from exports.json after level 2 "
                        "handed in two byte-identical levels), evaluate it at the probe points the task "
                        "prescribes by interpolating inside the element (never nearest node) and write the "
                        "per-level field file the task names; write the per-level interface file from that "
                        "side's own converged trace and its own outward flux at the prescribed interface "
                        "points -- BOTH from its interface_level<k>.csv (a Dirichlet side's trace there is the "
                        "value it imposed, identical to the partner's export; measured: a run that re-sampled "
                        "its field file at the boundary instead handed in a trace 2x off while its field was "
                        "right, so the result looked unphysical). The interpolation is four lines (measured on a served per-level dump): "
                        "a = numpy.loadtxt('field_level<k>.csv', delimiter=',', skiprows=1); "
                        "v = scipy.interpolate.griddata(a[:, :2], a[:, 2], P, method='linear') with P the "
                        "(n, 2) probe points -- one such call per value column (a[:, 2], a[:, 3], ... for a "
                        "field with several components, T then ux, uy for a thermo-elastic one); a NaN in v "
                        "is a probe outside this side's subdomain (leave it to the other side); write x, y "
                        "and the value columns in the task's order. Then each level's run log per side: a VERBATIM "
                        "copy of participant_output_level<k>.log next to that side's exports.json (never a summary). "
                        "CHECK: every coupled level has both sides' field files, interface files and run logs, and "
                        "audit_results(work_dir) reports no missing-fields and no run-log finding.", as_is=False)
        sides_l = set()
        for q in logs:
            if _level_of(q) != k or not _side_of(q):
                continue
            try:
                txt = strip_terminal_noise(q.read_text(errors="ignore"))
            except Exception:                           # noqa: BLE001
                continue
            if not sig_pats or any(_re.search(pp, txt, _re.IGNORECASE | _re.MULTILINE) for pp in sig_pats):
                sides_l.add(_side_of(q))
        if len(sides_l) < 2:
            return step(5, f"CAPTURE LEVEL {k}'S RUN LOGS: level {k} has a log holding the solver's own "
                           f"console output for {sorted(sides_l) or 'no'} side(s).",
                        f"For level {k}, for EACH side: the per-level run log the task names must hold that "
                        "solver's OWN console output (banner, iteration lines) plus the DOF-count line. The "
                        f"coupling tool kept each participant's captured console for THIS level as "
                        f"participant_output_level{k}.log next to its exports.json (participant_output.log is "
                        "only the latest level's) -- copy that file into the run log; never summarise or retype "
                        "it. CHECK: audit_results(work_dir) reports no run-log finding for this level.")
    next_level = (max(done_levels) + 1) if done_levels else 1
    short = (f" (its history so far has {rows_by_level.get(next_level, 0)} row(s))"
             if next_level in rows_by_level else "")
    if done_levels:
        what = (f"LEVELS {done_levels} ARE COMPLETE ON DISK. If your task prescribes more levels, run them ALL "
                f"in ONE call: couple_levels(participants=<the same list you passed to couple>, levels=[{next_level}, ...], "
                "history_pattern='<the task's per-level history file name with {k} in place of the level>'). "
                "Otherwise write the summary.")
        brief = (f"If the task prescribes a level {next_level}: call couple_levels(participants=<the same list you passed to "
                 f"couple>, levels=[{next_level}, ...every further level the task prescribes], history_pattern='<the task's "
                 "per-level history file name with {k} in place of the level number>') ONCE. It runs every remaining "
                 "level in that single call: each side gets its level's nx, ny in the environment (every cell count "
                 "doubled to halve h), each level warm-starts from the previous one, and each level's history file and "
                 "participant_output_level<k>.log are written. Measured: per-level couple() calls cost ten calls a level "
                 "and six proven couplings never reached level 3 that way. Then do steps 4 and 5 for every new level. "
                 "If the task prescribes no further level: write the summary file naming ONLY files that exist, then run "
                 "audit_results(work_dir). CHECK: the audit reports no missing level, no invented name and no missing "
                 "field file.")
        return step(6, what, brief)
    return step(3, f"COUPLE LEVEL {next_level}{short}.",
                f"Set level={next_level} in both ./config.json; call couple(participants=[{{name, command, "
                f"work_dir (absolute), imports_from}} for both sides], max_iter from the served rho guidance, "
                "tol from the task, history_path=<absolute path of this level's residual-history file>); "
                "read the reply's WHAT TO FIX NEXT and fix the named side until converged is true. Once this "
                "level converges, every further level is ONE couple_levels(participants=<same>, levels=[...], "
                "history_pattern='<per-level history file name with {k}>') call. CHECK: "
                "the residual-history file for this level exists with at least three rows and a falling "
                "residual, and the couple() reply says converged.")


def missing_fields_findings(work: Path) -> list[dict]:
    """A coupling that ran and wrote no field is not a result yet.

    Measured twice on the honest build: a run coupled three real levels
    (driver histories, eight or nine iterations to 1e-7 each), wrote its run
    logs and its residual histories, and handed in with NO per-level field
    file and NO interface file -- once listing invented names, once honestly
    listing only what existed. Nothing in the audit named the missing piece,
    because generic discovery cannot see a family that was never written.
    The agent's own histories and logs are the evidence that the solves ran,
    so the absence of any field file is a finding, not an unknown.
    """
    hist = _history_files(work)
    logs = _level_logs(work)
    if not (hist or logs):
        return []
    if _field_files(work) or _interface_files(work):
        return []
    levels = sorted({_level_of(q) for q in hist + logs if _level_of(q) is not None})
    return [{"sequence": "no field files", "priority": 15, "values": [], "finding": (
        f"NO PER-LEVEL FIELD FILE AND NO INTERFACE FILE EXISTS, while "
        f"{len(hist)} residual history file(s) and {len(logs)} per-level run "
        f"log(s) do (levels {levels}). The coupling ran and nothing was "
        f"written at the points your task prescribes, so there is no result "
        f"to check yet. For EACH level and EACH side: evaluate that side's "
        f"converged field at the prescribed probe points (interpolate inside "
        f"the element, never nearest node) and write the per-level field "
        f"file; write the per-level interface file from that side's own "
        f"converged trace and flux at the prescribed interface points. Every "
        f"number you already have is on disk in your participants' per-level "
        f"dumps (field_level<k>.csv, interface_level<k>.csv; exports.json holds "
        f"only the LAST level); no re-solve is needed.")}]


def summary_names_findings(work: Path) -> list[dict]:
    """Every file the summary names must exist. A summary that lists files it
    never wrote reads as invented, whatever the numbers beside it say.

    Measured: one coupled run listed 21 deliverables in its summary -- six
    per-level field files and six interface files among them -- and had
    written none of the twelve; its three residual histories were real. The
    audit at hand-in said nothing about the names, so the run handed in a
    list. No task knowledge is used: the names come from the agent's own
    summary text, and existence is checked by basename anywhere under the
    working directory outside openPASO's scratch.
    """
    import re as _re
    out: list[dict] = []
    sf = _summary_file(work)
    if sf is None or not sf.is_file():
        return out
    try:
        text = sf.read_text(errors="ignore")
    except OSError:
        return out
    text = _re.sub(r"\b(?:https?|ftp)://\S+", " ", text)      # links are not files
    names = sorted({m.group(0).strip("`'\"(),;")
                    for m in _re.finditer(r"[A-Za-z0-9_./-]+\.(?:csv|log|txt|json|vtu|vtk|pvd|yaml|yml|dat|npy|npz)\b", text)})
    names = [n for n in names if n and n != sf.name and not n.startswith(("http", "www."))]
    if not names:
        return out
    present: set = set()
    for q in work.rglob("*"):
        if not q.is_file():
            continue
        try:
            if _SCRATCH & set(q.relative_to(work).parts[:-1]):
                continue
        except ValueError:
            continue
        present.add(q.name)
    missing = [n for n in names if Path(n).name not in present]
    if not missing:
        return out
    out.append({"sequence": "summary names", "priority": 10, "values": [],
                "finding": (
        f"YOUR SUMMARY FILE NAMES {len(missing)} FILE(S) THAT DO NOT EXIST "
        f"anywhere under your working directory: {', '.join(missing[:8])}"
        + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        + ". A summary that lists files it never wrote is read as invented "
        "however real the rest of the work is. Write every file you name "
        "from the numbers you actually have -- or remove the name.")})
    return out


def run_log_identity_findings(work: Path) -> list[dict]:
    """A per-level run log must carry the named solver's OWN console output.

    A side is credited to a code only when its log holds a line that code
    emits (a solver-iteration line, a banner with a number). A log of the
    agent's own summary -- `NDOF = 54`, `solve completed`, `max|u| = ...` --
    is prose, and a side whose log is prose cannot be credited to that code
    however right its numbers are. Measured on one real coupled run: a side
    labelled DUNE-fem that solved with scipy and wrote a three-line summary
    passed every other check here. Reads only the agent's own files.
    """
    out: list[dict] = []
    try:
        from blind_eval.evidence import (            # noqa: PLC0415
            PER_CODE_SIGNATURES, strip_terminal_noise)
    except Exception:                                # signatures unavailable
        return out
    import re as _re
    pats = [p for plist in PER_CODE_SIGNATURES.values() for p in plist]
    # Per-SIDE logs only (run_level<k>_<side>.log): on a coupled problem the task
    # asks each side's log to carry that code's own console output, because
    # that is what says WHICH code ran which subdomain. A single-code log is
    # judged by the evidence gate's canonical lines and is not charged here.
    for f in _level_logs(work):
        if not _side_of(f):
            continue
        try:
            text = strip_terminal_noise(f.read_text(errors="ignore"))
        except Exception:
            continue
        if any(_re.search(p, text, _re.IGNORECASE | _re.MULTILINE)
               for p in pats):
            continue
        # THE COPY SOURCE, BY PATH. Measured (round 46, C1 7191): three such logs were named at
        # submission and the parent called them unfixable with 9 minutes left, while the driver's
        # per-level console copies sat in the side directories. Name the file to copy.
        _src = ""
        try:
            _k = _level_of(f); _sd = _side_of(f)
            if _k and _sd:
                for cand in sorted(work.rglob(f"participant_output_level{_k}.log")):
                    if cand.parent.name.lower().endswith(_sd.lower()) or cand.parent.name.lower().endswith(f"_{_sd.lower()}"):
                        _t = strip_terminal_noise(cand.read_text(errors="replace"))
                        if any(re.search(pp, _t, re.IGNORECASE | re.MULTILINE) for pp in pats) or re.search(r"^\s*NDOF\s*=\s*\d+\s*$", _t, re.M):
                            _src = (f" The coupling tool kept that side's console for level {_k} at {cand} ({len(_t)} bytes, with the "
                                    f"solver's own lines): copy that file over this one -- one command -- and keep its NDOF line.")
                        break
        except Exception:                                # noqa: BLE001
            _src = ""
        out.append({"sequence": f"run log {f.name}", "finding": (
            f"{f.name}: THIS LOG CARRIES NO LINE ANY SOLVER EMITS "
            f"({len(text)} bytes of your own summary). The task wants that "
            "subdomain's solver console output, captured verbatim (its "
            "iteration lines, its banner), because that is what establishes "
            "WHICH code ran on that side; a side whose log is your own words "
            "cannot be credited to that code however right its numbers are. "
            "If you ran it through subprocess you already have the bytes: "
            "write result.stdout (and stderr) into this file, plus the NDOF "
            "line." + _src)})
    return out



_AXIS_SUFFIX = re.compile(r"^(?P<base>.+?)[ _\-]?(?P<ax>[xyz])$", re.I)


def _vector_order_view(seqs: dict) -> dict:
    """Replace the components of one vector field by the vector itself.

    AN ORDER IS A PROPERTY OF A FIELD IN A NORM, NOT OF ONE CARTESIAN
    COMPONENT. The interface exclusion directly below already learned half of
    this lesson; the other half is the minor component of a vector. On the first
    vector-valued coupled run that was correct, at order 1.9 -- side
    A's ux self-difference improves at 1.92 and its uy, seven times smaller and
    sitting near the coupling iteration's own floor, improves at 0.96. The
    check fired on that component and told a correct agent its exchanged datum
    was O(h) and its coupling first-order. The vector formed from both
    components improves at 1.88, which is what the grader measures and what is
    true. Every scalar problem has one component per side, so its view is
    unchanged -- which is why this never showed on the 26 scalar CORRECT cells.

    Components join only when their own column names say so: identical after a
    trailing x/y/z is removed, with at least two distinct axes present. A field
    with no axis suffix always stands alone, so a temperature shipped beside a
    displacement (`T, ux, uy`, 43 files here) is never folded into it and can
    never be masked by it.
    """
    fam: dict = {}
    for label, seq in seqs.items():
        if not label.startswith("selfdiff_"):
            continue
        tag, _, field = label[len("selfdiff_"):].rpartition("_")
        if not tag:
            continue
        m = _AXIS_SUFFIX.match(field)
        if not m or not m.group("base"):
            continue
        fam.setdefault((tag, m.group("base")), {})[m.group("ax").lower()] = (
            label, seq)
    view = dict(seqs)
    for (tag, base), axes in fam.items():
        if len(axes) < 2:
            continue                      # one axis is not a vector
        members = [axes[a] for a in sorted(axes)]
        n = min(len(s) for _, s in members)
        if n < 2:
            continue
        for lab, _ in members:
            view.pop(lab, None)
        view[f"selfdiff_{tag}_{base}|vector"] = [
            math.sqrt(sum(s[i] ** 2 for _, s in members)) for i in range(n)]
    return view


def audit(work_dir: str, claimed_order: float | None = None,
          summary_path: str | None = None) -> dict:
    """The three questions, answered from the agent's own files.

    `summary_path` is the caller's hint for the agent's summary/answer file
    (a harness knows which file it just saw written); without it the file is
    discovered by name (result / summary / report / answer)."""
    work = Path(work_dir)
    if summary_path:
        _SUMMARY_HINT[str(work)] = summary_path
    findings: list[dict] = []
    findings.extend(residual_findings(work))
    findings.extend(completeness_findings(work))
    findings.extend(exact_ladder_findings(work))   # the provable one leads the band one
    findings.extend(ndof_ladder_findings(work))
    findings.extend(interface_ends_findings(work))
    findings.extend(unlaunched_participants_findings(work))
    findings.extend(run_log_identity_findings(work))
    findings.extend(summary_names_findings(work))
    findings.extend(missing_fields_findings(work))
    seqs = _sequences_from_workdir(work)
    csvs = _sequences_from_level_csvs(work)
    if "__ambiguous__" in csvs:
        # KEEP WHAT IS ALREADY KNOWN. This rebuilt the findings list from
        # scratch, so a residual history that had already been read and found
        # broken was dropped the moment two files collided on one per-level
        # slot. The two are independent — the residual check needs no per-level
        # field files at all, and the ambiguity is about which field file to
        # read — so reporting only the ambiguity told the agent to tidy its
        # filenames while saying nothing about a coupling that never converged.
        # THE INTERFACE FINDING SURVIVES AN AMBIGUOUS FIELD SET, for the
        # same reason the residual one does: it reads interface files, not
        # the per-level field slot that collided. So do the identity /
        # sampling / continuity checks — each reads its own files, none of
        # them the collided per-level field slot.
        findings = findings + interface_sign_findings(work)
        findings.extend(interface_continuity_findings(work))
        findings.extend(identical_solution_levels_findings(work))
        findings.extend(wrong_level_run_log_findings(work))
        findings.extend(solution_rows_grow_findings(work))
        findings.extend(equation_findings(work))
        findings = findings + [
            {"sequence": "level files", "values": [],
             "finding": (
                 "AMBIGUOUS INPUT: more than one file matches "
                 "the per-level pattern at the top level (" +
                 ", ".join(csvs["__ambiguous__"][:4]) +
                 "). I will not guess which is your answer — "
                 "name your per-level files uniquely, or "
                 "remove the stale ones, and re-run this "
                 "check.")}]
        return {"sequences_found": 0, "clean": False,
                "what_to_fix_next": what_to_fix_next(findings),
                "findings": findings,
                "note": ("the per-level field check did not run: input was "
                         "ambiguous. Any other finding above DID run and "
                         "stands.")}
    seqs.update(csvs)
    # near-zero field: the loads may never have been applied at all
    for label, seq in list(seqs.items()):
        # "< 1e-8" must INCLUDE exact zero — the three 4C runs that wired
        # VAL: [0.0], FUNCT: [0] delivered fields of literal 0.0 everywhere,
        # and "0 < x" excluded precisely them.
        if label.startswith("magnitude_") and seq and seq[0] < 1e-8:
            findings.append({"sequence": label, "values": seq, "finding": (
                "NEAR-ZERO FIELD: the finest-level field peaks below 1e-8. "
                "For a driven problem that usually means the load was never "
                "applied — check the deck actually contains your body "
                "force/source and that its load curve is active — not that "
                "the answer is a very small number.")})
    # A vector is one field; see _vector_order_view for why and for the
    # cell that proved it.
    for label, seq in _vector_order_view(seqs).items():
        if label.startswith("magnitude_"):
            continue
        if len(seq) < 3 and not label.startswith("selfdiff_"):
            continue
        if len(seq) < 2:
            continue
        entry = {"sequence": label, "values": seq}
        # 1. floor detection FIRST — a flat sequence is precisely the case the
        # old monotonicity filter threw away before this check could see it,
        # which is why runs with errors flat at ~6e-7 sailed through.
        rel = [abs(a - b) / max(abs(a), 1e-300) for a, b in zip(seq, seq[1:])]
        if all(r < 0.05 for r in rel):
            entry["finding"] = (
                "FLOOR: the levels are within 5% of each other, so refinement "
                "is changing nothing. Whatever limits this number, it is not "
                "the mesh — check solver tolerances (a nonlinear/iterative "
                "solver left at its default stops around 1e-6-1e-7 and that "
                "floor becomes your 'error'), or a fixed post-processing step.")
            findings.append(entry)
            continue
        drops = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
        if drops < len(seq) - 2:      # mostly non-decreasing: not error-like
            continue
        # 2/3. observed order vs the claim
        try:
            orders = [math.log2(a / b) for a, b in zip(seq, seq[1:]) if b > 0]
        except ValueError:
            continue
        if not orders:
            continue                     # nothing comparable; not a finding
        entry["observed_orders"] = [round(o, 2) for o in orders]
        med = sorted(orders)[len(orders) // 2]
        # THE CLAIMED ORDER IS THE FIELD'S, NOT THE INTERFACE TRACTION'S.
        #
        # Grouping by (kind, side) gave the audit interface_* sequences for the
        # first time, and the order check then compared them against the order
        # claimed in RESULT.txt. Those are different quantities: an interface
        # node that sits at the end of the interface has a one-sided boundary
        # weight and converges at order 1, measured 1.000/1.013/1.010 against a
        # known exact flux while the interior runs at 1.99. So a perfectly good
        # coupling shows interface self-differences improving at ~0.9-1.4.
        # It cost exactly one false alarm, and it was on the single coupled
        # run anyone had, at that point, verified correct against an
        # independent reference.
        _is_iface = label.startswith("selfdiff_interface") or \
            label.startswith("magnitude_interface")
        if _is_iface:
            continue
        _coupled_low = (0.5 <= med <= 1.45
                        and bool(_interface_files(work, sided=True)))
        if (claimed_order is None and _coupled_low
                and not label.startswith("magnitude_")):
            entry["finding"] = (
                f"YOUR OWN LEVELS IMPROVE AT ONLY ~{med:.2f}. On a COUPLED "
                "run an order stuck near 1 with a converged coupling is the "
                "FIRST-ORDER INTERFACE RECOVERY signature: the exchanged "
                "datum (flux or trace) is O(h) accurate and pollutes the "
                "field everywhere — the boundary trace of a P1 element "
                "gradient and a two-point nearest-node difference are both "
                "worth order ~1 there. Recover the exchanged quantity with "
                "the consistent residual (q = -(A u - b_vol)/w on the "
                "interface rows of YOUR OWN system) or a one-sided QUADRATIC "
                "through three points along the normal, then re-derive it at "
                "EVERY level and both sides. Measured: "
                "conversions from 0.85-0.97 to ~1.84 come from exactly this "
                "change and nothing else.")
            findings.append(entry)
        elif claimed_order is not None and med < claimed_order - 0.4:
            _msg = (
                f"ORDER MISMATCH: you are about to claim order "
                f"{claimed_order:g} but your own levels improve at ~{med:.2f}. "
                f"Common causes: an element degree lower than the task states; "
                f"a first-order time integrator behind a spatial study; "
                f"volumetric locking (near-incompressible material solved with "
                f"a pure displacement form — if your notes say 'mixed "
                f"formulation', check the assembled form actually uses it); "
                f"a low-order quadrature or projection in post-processing.")
            # THE COUPLED CAUSE WAS MISSING FROM THIS LIST. Measured three
            # times in one development day: couplings with converged
            # three-level evidence read as order 0.85, 0.96 and 0.97 against a
            # theoretical 2, each from a DIFFERENT first-order interface
            # recovery (a two-point nearest-node difference; a P1 element
            # gradient evaluated ON the boundary). Pattern checks cannot
            # enumerate the variants; the ORDER ITSELF is the signature, and
            # this is the one message every such run reads before delivering.
            if 0.5 <= med <= 1.45 and _interface_files(work, sided=True):
                _msg += (
                    " On a COUPLED run an order stuck near 1 with a converged "
                    "coupling is the FIRST-ORDER INTERFACE RECOVERY "
                    "signature: the exchanged datum (flux or trace) is O(h) "
                    "accurate and pollutes the field everywhere — the "
                    "boundary trace of a P1 element gradient and a two-point "
                    "nearest-node difference are both worth order ~1 there. "
                    "Recover the exchanged quantity with the consistent "
                    "residual (q = -(A u - b_vol)/w on the interface rows of "
                    "YOUR OWN system) or a one-sided QUADRATIC through three "
                    "points along the normal, then re-derive it at EVERY "
                    "level and both sides.")
            entry["finding"] = _msg
            findings.append(entry)
        elif any(o < -0.1 for o in orders):
            entry["finding"] = (
                "NON-MONOTONE: at least one refinement made the answer WORSE. "
                "A converging study never does that outside noise; check for a "
                "mesh-dependent bug (wrong BC on the finer mesh, probe points "
                "outside the domain, a tolerance floor).")
            findings.append(entry)
    findings.extend(contract_findings(Path(work_dir)))
    findings.extend(interface_sign_findings(work))
    findings.extend(export_findings(work))
    findings.extend(interface_continuity_findings(work))
    findings.extend(identical_solution_levels_findings(work))
    findings.extend(wrong_level_run_log_findings(work))
    findings.extend(solution_rows_grow_findings(work))
    # NOT VOLUNTARY: see equation_findings. The one check that separates a field
    # converging to the right function from one converging to a wrong one was
    # called in 1 of 1118 recorded coupled runs, so it is computed here from the
    # agent's own files and reported whether or not it was asked for.
    findings.extend(equation_findings(work))
    clean = not [f for f in findings if not f.get("informational")]
    # THE LADDER: the next unmet step of a coupled run, from the files. It
    # leads a clean reply and closes a dirty one, so the agent always knows
    # the one thing to do next.
    try:
        _ladder = coupled_ladder(work)
    except Exception:                                   # noqa: BLE001
        _ladder = None
    _lead = what_to_fix_next(findings, clean_msg=(_ladder["text"] if _ladder else None))
    if _ladder and _ladder["text"] not in _lead:
        _lead += "\n" + _ladder["text"]
    return {
        "sequences_found": len(seqs),
        "next_step": (_ladder or {}).get("text"),
        # THE SINGLE NEXT FIX, FIRST. A weak agent reads the top of the reply,
        # so the highest-priority finding leads with its full corrective text;
        # a clean funnel leads with the next ladder step.
        "what_to_fix_next": _lead,
        "findings": findings,
        "clean": clean,
        "note": ("This audit uses ONLY your own files — no reference "
                 "solution. 'clean' means self-consistent, not correct."),
    }


# ── the proof that is owed NOW, not at hand-in ─────────────────────────────
#
# MEASURED, and it is the largest recoverable loss in the record. Of 516
# graded coupled cells, 112 were malformed and 92 of those carry PROVEN
# execution evidence: the runs happened. 34 of them reached three coupled
# levels with both codes proven, the coupling proven and the interface
# satisfied, and scored nothing. The single commonest reason is one line --
# no captured run log carrying `NDOF = <n>` for some (level, side).
#
# The check for that already exists and already runs. It runs when the
# submission is written, and by then the median cell has TWO tool calls left;
# one in five has none. It names work costing five to fifteen actions. The
# same observation was made once before from file mtimes -- five of six runs
# wrote their summary at 93-99% of their file-activity span -- and two checks
# were moved early in response. These two were not, and they are the two that
# decide the malformed bucket.
#
# So this selects the subset that is ALREADY OWED for the levels the agent has
# actually coupled, to be delivered as each level finishes. What it must not
# do is ask for work that is not due yet: `contract_findings` at level 1
# reports ONLY 1 LEVEL(S) DELIVERED, which is true at hand-in and pure noise
# to an agent who is on level 1 of three (measured on seeds 5622 and 7262).
# A gate that names an absent defect costs an action for nothing.

_DUE_PER_LEVEL = ("run-log contract", "coupling evidence", "discretisation size")


def deliverable_proof_due(work: Path, levels_done) -> list[dict]:
    """Findings already owed for the levels that have actually been coupled.

    Self-consistency only: every one of these reads the agent's own files and
    compares them against each other. Nothing here knows what any task asked
    for.
    """
    try:
        lv = {int(k) for k in (levels_done or [])}
    except (TypeError, ValueError):
        return []
    if not lv:
        return []
    out: list[dict] = []
    try:
        out += [f for f in contract_findings(work, only_levels=lv)
                if f.get("sequence") in _DUE_PER_LEVEL]
    except Exception:                                    # noqa: BLE001
        pass
    try:
        out += completeness_findings(work, only_levels=lv)
    except Exception:                                    # noqa: BLE001
        pass
    # run_log_identity_findings IS DELIBERATELY NOT HERE, and it was in the
    # first draft. Measured against the recorded runs it speaks on 9 of the 32
    # CORRECT cells -- C3 runs whose logs are 76 to 242 bytes of the agent's
    # own words and which an independent check accepted anyway. A mid-run
    # surface that interrupts 28% of the work that goes on to be right is not
    # precise enough to be worth an action. It stays in the hand-in audit,
    # where it costs less and where it belongs.
    return out


# ── what your own files already say about the ANSWER, per level ────────────
#
# The deliverable proof above asks whether the run can be shown to have
# happened. This asks whether what it produced is worth handing in, and it is
# the family that decides a correct run against a completed but unphysical one.
#
# MEASURED on a live run that coupled three levels with both codes
# proven, wrote every deliverable, and came out completed but unphysical at order
# 0.16. Its own files said so three times over -- the field peaks below 1e-8,
# the levels sit within 5% of each other so refinement changes nothing, and
# the reported interface residual of 3.50e-15 is contradicted by its own two
# interface files, which differ by 127% of their own scale. Every one of those
# findings existed. NEAR-ZERO FIELD and FLOOR appear ZERO times in its
# trajectory, because they live only inside audit(), and audit() ran when the
# submission was written -- with two tool calls left.
#
# Same bodies, asked per level while the ladder is still running, and scoped
# to levels that have actually been coupled.

_MAG = "magnitude_"


def field_quality_due(work: Path, levels_done) -> list[dict]:
    """Answer-quality findings already visible in the agent's own files.

    Self-consistency only: a field measured against itself across levels, and
    the two sides' interface files measured against each other.
    """
    try:
        lv = {int(k) for k in (levels_done or [])}
    except (TypeError, ValueError):
        return []
    if not lv:
        return []
    out: list[dict] = []
    try:
        seqs = dict(_sequences_from_workdir(work))
        csvs = _sequences_from_level_csvs(work)
        if "__ambiguous__" not in csvs:
            seqs.update(csvs)
    except Exception:                                    # noqa: BLE001
        seqs = {}
    for label, seq in seqs.items():
        # SOLUTION FIELDS ONLY. An interface TRACE may legitimately be zero --
        # a Dirichlet seam held at zero, a flux that genuinely vanishes -- and
        # two cells that went on to grade CORRECT carry exactly that (an
        # interface u of 0 and an interface q of 1.9e-16). A solution field
        # that peaks below 1e-8 is a solve that produced nothing, and that is
        # never right.
        if not label.startswith(_MAG + "solution") or not seq:
            continue
        if seq[0] < 1e-8:
            out.append({"sequence": label, "values": seq, "finding": (
                "NEAR-ZERO FIELD: this field peaks below 1e-8. For a driven "
                "problem that almost always means the load never entered the "
                "assembled system -- not that the answer is a very small "
                "number. Check it now: a level that solves nothing costs the "
                "whole ladder built on top of it.")})
    # FLOOR needs two levels to mean anything, and only the self-difference
    # sequences carry it -- a magnitude that barely moves is often correct.
    for label, seq in _vector_order_view(seqs).items():
        if not label.startswith("selfdiff_") or len(seq) < 2:
            continue
        rel = [abs(a - b) / max(abs(a), 1e-300) for a, b in zip(seq, seq[1:])]
        if rel and all(r < 0.05 for r in rel):
            out.append({"sequence": label, "values": seq, "finding": (
                "FLOOR: successive levels are within 5% of each other, so "
                "refinement is changing nothing. Whatever limits this number "
                "it is not the mesh -- an iterative solver left at its default "
                "stops around 1e-6 to 1e-7 and that floor becomes your "
                "'error'. Refining further buys nothing until it moves.")})
            break                                        # one is the message
    # AND THE LADDER THAT IS GOING BACKWARDS. A refinement that makes the
    # answer worse is never noise at this size, and it is the opposite failure
    # to FLOOR: measured on a recorded cell whose self-differences grew about
    # fivefold per level while its interface residual read 1e-15 and every
    # other check reported clean.
    for label, seq in _vector_order_view(seqs).items():
        if not label.startswith("selfdiff_solution") or len(seq) < 2:
            continue
        if all(b > a * 1.5 for a, b in zip(seq, seq[1:])):
            out.append({"sequence": label, "values": seq, "finding": (
                "YOUR LADDER IS GOING BACKWARDS: each refinement made this "
                "quantity WORSE, not better. A converging study never does "
                "that outside noise, and a growing self-difference means the "
                "finer mesh is solving a different problem -- a boundary "
                "condition that moves with the mesh, an exchanged quantity "
                "that scales with the node count, or probe points outside the "
                "domain. Nothing built on top of this level will fix it.")})
            break
    # THE REPORTED-RESIDUAL-VS-FILES CHECK IS DELIBERATELY NOT HERE. It fires
    # on a run that was correct (one case: the two interface
    # files disagree at level 3 and the result was right anyway), so it is not
    # precise enough to interrupt a run with. It stays in the hand-in audit.
    try:
        out += [f for f in (interface_continuity_findings(work) or [])
                if any(f" level {k}" in str(f.get("sequence", "")) for k in lv)]
    except Exception:                                    # noqa: BLE001
        pass
    out += _interface_transmitted_nothing(work, lv)
    out += _imported_trace_not_held(work)
    out += _side_exported_nothing(work)
    return out


def _side_exported_nothing(work: Path) -> list[dict]:
    """One side's exports carry no field and no flux: all values exactly zero.

    THE UPSTREAM SUPPLY OF THE DEAD EXCHANGE. `_interface_transmitted_nothing`
    catches the pair once BOTH sides are silent and the per-level interface
    files exist; by then the iteration has already converged on nothing and the
    ladder is built on it. This reads one side's own exports.json, which exists
    the moment a participant runs once -- before any coupling, before any
    deliverable -- and says so while the run is still one side long.

    THE SERVED SELF-CHECK CANNOT SEE THIS. Its zero guard
    (coupling_knowledge.py, EXPORT SELF-CHECK) fires only for a Neumann side
    whose recovered flux is ~0 against a NONZERO imported flux. A side handed
    zeros -- the first coupling iteration, or a partner that is itself silent --
    skips that branch entirely and exports nothing without a word. Measured in
    single-side trials with the contract in hand and nothing to orchestrate:
    2 of 7 successful-looking exports were identically zero.

    Measured over several hundred runs: it flagged forty-one, none of which
    was correct. It fires on nothing that ever produced a sound number.

    Judged only where the side exported something at all: an absent or empty
    exports.json is a different defect with its own finding, and a side that
    exports values but no fluxes (or the reverse) is judged on what it wrote.
    """
    out: list[dict] = []
    try:
        import numpy as _np
    except Exception:                                    # noqa: BLE001
        return out
    for ep in sorted(work.rglob("exports.json")):
        try:
            e = json.loads(ep.read_text() or "{}")
        except (OSError, ValueError):
            continue
        if not isinstance(e, dict):
            continue
        v = _np.asarray(e.get("values") or [], float).ravel()
        q = _np.asarray(e.get("normal_fluxes") or [], float).ravel()
        if v.size == 0 and q.size == 0:
            continue                      # exported nothing at all: not this
        vmax = float(abs(v).max()) if v.size else 0.0
        qmax = float(abs(q).max()) if q.size else 0.0
        if vmax != 0.0 or qmax != 0.0:
            continue
        side = ep.parent.name or str(ep.parent)
        out.append({"sequence": f"side exported nothing {side}",
                    "values": [vmax, qmax], "finding": (
            f"{side} EXPORTED A FIELD AND A FLUX THAT ARE IDENTICALLY ZERO "
            f"({ep}). Every value and every flux is exactly 0.0, so this side "
            f"handed its partner nothing. On a driven problem that is the "
            f"answer a solve returns when the load never entered the assembled "
            f"system -- not a small number, the no-load answer. The export "
            f"self-check in the served contract cannot see it: its zero guard "
            f"only fires against a NONZERO imported flux, and a side handed "
            f"zeros skips that branch. Two sides in this state agree perfectly, "
            f"so the iteration converges at once and every self-consistency "
            f"number the run reports about itself reads as success. Check that "
            f"your source term and your boundary data actually reach the "
            f"assembled system before coupling on this.")})
        break                              # one side is the message
    return out


def _imported_trace_not_held(work: Path) -> list[dict]:
    """A side's exported interface values differ from the ones it imported.

    THE SAME DEFECT AS THE NGSOLVE-SPECIFIC WRITE GATE, SEEN FROM THE DATA AND
    THEREFORE TRUE OF EVERY BACKEND. On a Dirichlet-Neumann pair the side that
    imports a trace imposes it, so its own exported trace must come back
    unchanged. Where it does not, the solve overwrote what was handed to it and
    the two subdomains are answering different problems -- while the iteration
    still converges, because each side is self-consistent.

    FOUND BY ASKING WHAT SEPARATES C9 8631 (CORRECT) FROM 8632 (UNPHYSICAL).
    Same problem, same two codes, same recovery method, the same 44 interface
    points in the same order, three coupled levels each. 8631's side B exports
    exactly what it imported (0.00%); 8632's differs by 127%. Five other
    hypotheses were ruled out first -- point correspondence, corner handling,
    the constrained dof set, the values array indexing, and whether
    skfem's solve(*condense(...)) restores constrained values (it does).

    Measured over every run: **none of the judgeable correct sides is
    flagged**, against many sides in failing ones.

    Judgeable only where a side imports and exports the same shape of values,
    which is what makes it the trace-holding side; anything else returns
    nothing rather than a guess.
    """
    import json as _json
    out: list[dict] = []
    try:
        import numpy as _np
    except Exception:                                    # noqa: BLE001
        return out
    for sd in sorted(work.glob("side_*")):
        try:
            ip, ep = sd / "imports.json", sd / "exports.json"
            if not (ip.is_file() and ep.is_file()):
                continue
            imp = _json.loads(ip.read_text() or "{}")
            exp = _json.loads(ep.read_text() or "{}")
            src = next(iter(imp.values()), {}) if imp else {}
            iv = _np.array(src.get("values") or [], float)
            ev = _np.array(exp.get("values") or [], float)
            ic = _np.array(src.get("coordinates") or [], float)
            ec = _np.array(exp.get("coordinates") or [], float)
            if iv.size == 0 or ev.size == 0 or iv.shape != ev.shape:
                continue                                 # not the trace-holding side
            if ic.shape[0] != iv.shape[0] or ec.shape[0] != ev.shape[0]:
                continue
            scale = float(abs(iv).max())
            if scale == 0.0:
                continue
            oi = _np.argsort(ic[:, 0]); oe = _np.argsort(ec[:, 0])
            rel = float(abs(iv[oi] - ev[oe]).max() / scale)
            if rel < 0.01:
                continue
            out.append({"sequence": f"imported trace {sd.name}", "values": [rel],
                        "finding": (
                f"{sd.name} EXPORTS A DIFFERENT TRACE FROM THE ONE IT IMPORTED: "
                f"the two differ by {rel:.0%} of the imported scale at the same "
                f"interface points. This side is given the partner's trace and "
                f"imposes it, so what it exports back must be that trace "
                f"unchanged; a difference means the solve moved values it was "
                f"told to hold. The iteration converges anyway -- each side is "
                f"self-consistent -- and the two subdomains end up answering "
                f"different problems. Check that the interface degrees of "
                f"freedom are in the constrained set the solve honours, and "
                f"that nothing overwrites them afterwards. Measured across "
                f"recorded runs, no correct run has ever exported a trace that "
                f"differs from the one it was given.")})
        except Exception:                                # noqa: BLE001
            continue
    return out


def _interface_transmitted_nothing(work: Path, lv) -> list[dict]:
    """The same exchanged column identically zero on BOTH sides.

    NEITHER SIDE'S OWN SELF-CHECK CAN SEE THIS. Each participant guards
    "my recovered flux is ~0 against a NONZERO imported one" -- so when the
    exchange is zero in both directions, both guards are satisfied and both
    stay silent. The iteration then converges immediately, because a pair that
    exchanges nothing cannot disagree, and each side returns the answer it
    would have returned uncoupled.

    MEASURED on a live run that coupled three levels with both codes
    proven and its two sides agreeing on displacement to the digit, while the
    traction columns read 9.5e-18 and 0.0. It came out completed but unphysical.
    Across the record the rule hits fourteen runs and none of them is
    correct. A seam
    legitimately at zero does not trip it: the test needs the SAME column dead
    on BOTH sides while the interface carries a nonzero scale elsewhere.
    """
    out: list[dict] = []
    try:
        import csv as _csv
        for fa in sorted(work.rglob("interface_level*_A.csv")):
            k = _level_of(fa)
            if k is None or k not in lv:
                continue
            fb = Path(str(fa).replace("_A.csv", "_B.csv"))
            if not fb.is_file():
                continue
            def _read(q):
                with q.open() as fh:
                    rd = _csv.reader(fh)
                    hdr = next(rd, None)
                    rows = []
                    for r in rd:
                        try:
                            rows.append([float(x) for x in r])
                        except ValueError:
                            pass
                return hdr, rows
            ha, A = _read(fa)
            hb, B = _read(fb)
            if not A or not B or len(A[0]) != len(B[0]):
                continue
            ncol = len(A[0])
            scale = max([abs(v) for r in A for v in r[1:]]
                        + [abs(v) for r in B for v in r[1:]] + [0.0])
            if scale <= 1e-12:
                continue
            for c in range(1, ncol):
                za = max(abs(r[c]) for r in A)
                zb = max(abs(r[c]) for r in B)
                if za < 1e-12 * scale and zb < 1e-12 * scale:
                    name = (ha[c] if ha and c < len(ha) else f"column {c}")
                    out.append({"sequence": f"interface transmitted nothing level {k}",
                                "values": [za, zb], "finding": (
                        f"YOUR TWO SIDES EXCHANGED NOTHING at level {k}: the "
                        f"`{name}` column is identically zero in BOTH "
                        f"{fa.name} and {fb.name} ({za:.2e} and {zb:.2e}) while "
                        f"the interface carries values of order {scale:.2e} "
                        f"elsewhere. Neither participant's export self-check can "
                        f"see this -- each only fires when its OWN recovery is ~0 "
                        f"against a NONZERO partner, and here both are zero, so "
                        f"both stay quiet. The iteration then converges at once, "
                        f"because two sides exchanging nothing cannot disagree, "
                        f"and each returns the answer it would have returned with "
                        f"no partner at all. Find where that quantity is recovered "
                        f"and why it comes out zero before running another level.")})
                    break
    except Exception:                                    # noqa: BLE001
        return out
    return out
