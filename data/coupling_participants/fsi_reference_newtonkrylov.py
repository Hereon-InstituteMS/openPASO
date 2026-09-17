"""Independent re-solve of a two-participant coupling, by Newton-Krylov.

Written for `couple(monolithic=...)`: it runs in its own work_dir, takes no
arguments, and writes `monolithic.json` in InterfaceData shape on the SECOND
participant's interface points.

WHAT IT IS. A partitioned coupling iterates the relaxed fixed point and stops
when the exchanged data stops changing. Whether it stopped AT THE ROOT of the
coupled system, or merely stopped moving, is not a question that iteration can
answer about itself: an under-relaxed iteration on a slightly wrong map still
converges, smoothly, to the wrong place. So this script solves the same coupled
system a second time by a different outer algorithm —

      R(y) = B(A(y)) - y = 0      solved with scipy.optimize.root(method="krylov")

a matrix-free Newton method with GMRES and finite-difference Jacobian-vector
products, where y is the interface state participant B exports. Newton has no
relaxation parameter, cannot converge to a non-root, and reaches the root in a
handful of evaluations regardless of the fixed-point iteration's contraction
factor. It drives the SAME participant scripts as the same subprocesses over the
same imports.json / exports.json files, so the discrete equations it solves are
identical to the ones the coupling iterates.

WHAT IT IS NOT, and this matters for how far the agreement can be pushed:

  * it is NOT a monolithic solver. It does not assemble one system over both
    fields. It solves the coupled INTERFACE equation to a Newton tolerance.
  * it shares the participant scripts with the coupling, so it CANNOT detect a
    bug inside a participant — a wrong material number, a sign error in an
    exported traction, a mis-stated boundary condition. Both would be wrong
    identically and agree perfectly. It establishes that the ITERATION found the
    root of the system it was given, and nothing more.
  * the check that covers what this one cannot is an independent solve of the
    same problem in a DIFFERENT code (for FSI: a native monolithic FSI solver).
    Run both. Neither substitutes for the other.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every path and number below is a PLACEHOLDER ─────────
#    A_* is the participant driven FIRST from the current interface state,
#    B_* the one whose export IS the interface state being solved for.
A_NAME    = "fluid"
A_SCRIPT  = "/absolute/path/to/participant_a.py"
A_CMD     = ["python", "participant_a.py"]
B_NAME    = "solid"
B_SCRIPT  = "/absolute/path/to/participant_b.py"
B_CMD     = ["python", "participant_b.py"]
FTOL      = 1e-11        # Newton residual tolerance (absolute, on the ravelled state)
MAXITER   = 60
RDIFF     = 1e-6         # relative step for the finite-difference Jacobian-vector product
TIMEOUT   = 900          # per participant invocation
# ─────────────────────────────────────────────────────────────────────────


def _run(wd: Path, cmd, imports: dict) -> dict:
    (wd / "imports.json").write_text(json.dumps(imports, indent=2, sort_keys=True))
    ep = wd / "exports.json"
    if ep.exists():
        ep.unlink()
    r = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True,
                       timeout=TIMEOUT)
    if r.returncode != 0 or not ep.exists():
        raise RuntimeError(f"{cmd} rc={r.returncode}: {r.stderr[-800:]}")
    return json.loads(ep.read_text())


def main() -> int:
    here = Path.cwd()
    wa, wb = here / "_ref_a", here / "_ref_b"
    for w, src in ((wa, A_SCRIPT), (wb, B_SCRIPT)):
        if w.exists():
            shutil.rmtree(w)
        w.mkdir(parents=True)
        shutil.copy(src, w / Path(src).name)

    calls = [0]

    # iteration-1 pass: fixes the interface point layout and gives a start value
    ea = _run(wa, A_CMD, {})
    eb = _run(wb, B_CMD, {A_NAME: ea})
    coords = np.asarray(eb["coordinates"], float)
    y0 = np.asarray(eb["values"], float)
    shape = y0.shape

    def G(y):
        calls[0] += 1
        y = np.asarray(y, float).reshape(shape)
        payload = {B_NAME: {"field_name": eb["field_name"],
                            "n_points": int(len(coords)),
                            "coordinates": coords.tolist(),
                            "values": y.tolist()}}
        a = _run(wa, A_CMD, payload)
        b = _run(wb, B_CMD, {A_NAME: a})
        return np.asarray(b["values"], float)

    from scipy.optimize import root as sroot

    def R(yflat):
        return (G(yflat) - yflat.reshape(shape)).ravel()

    sol = sroot(R, y0.ravel(), method="krylov",
                options={"fatol": FTOL, "disp": False, "maxiter": MAXITER,
                         "jac_options": {"rdiff": RDIFF}})
    y = sol.x.reshape(shape)
    res = float(np.linalg.norm(R(sol.x)))
    rel = res / max(float(np.linalg.norm(y)), 1e-300)

    Path("monolithic.json").write_text(json.dumps({
        "field_name": eb["field_name"],
        "n_points": int(len(coords)),
        "coordinates": coords.tolist(),
        "values": y.tolist(),
        "meta": {"method": "Newton-Krylov on the coupled interface residual",
                 "success": bool(sol.success), "message": str(sol.message),
                 "evaluations": int(calls[0]),
                 "residual_norm": res, "residual_relative": rel},
    }, indent=2))
    print(f"[reference] success={sol.success} evals={calls[0]} "
          f"|R|={res:.3e} rel={rel:.3e}", flush=True)
    # A reference that did not reach its own root is not a reference. Exit
    # non-zero so `couple` reports NOT CHECKED rather than comparing against it.
    return 0 if (sol.success and rel < 1e-6) else 1


if __name__ == "__main__":
    sys.exit(main())
