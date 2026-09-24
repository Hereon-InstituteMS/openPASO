"""deal.II VECTOR participant for the openPASO `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X. The
exchanged interface state is a VECTOR on BOTH channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export, q_out = -(sigma . n_own)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Pure glue: the PDE solve is done by a compiled deal.II executable THAT YOU
WRITE AND BUILD YOURSELF. No C++ source ships with this contract and none is on
this install, so do not search for one: write a program that reads the side,
the geometry, the material, the mesh size, the source samples and the imported
interface samples from one plain-text file your solve region writes, and
writes the interface trace and the CONSISTENT flux (the residual of the
assembled system with no boundary condition applied, divided by the nodal
interface weight) to a plain-text file your solve region reads back. That pair
of files is private to you. This wrapper is the contract around it: the
imports.json handshake, the sign convention, the exports schema and the
self-check. deal.II has no Python API, so unlike every other backend there is
a BUILD STEP before this can run at all:

    cmake -S <dir with YOUR .cc and a 6-line CMakeLists> -B <build> \
          -DDEAL_II_DIR=<deal.II BUILD or INSTALL tree>
    make -C <build>

and DEALII_EXE below must point at YOUR binary.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.55
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.55
E_MOD     = 1000.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)


def B_SRC(x, y):
    """Body force per unit volume, (b_x, b_y), as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants: with displacement prescribed
    on the whole outer boundary and no body force, the only solution is
    u = 0 everywhere, and the coupling will converge beautifully to it.

    If your problem states a body force, or gives you a manufactured solution
    whose source term you derived, put it here. `x` and `y` are NumPy arrays,
    so build the answer with NumPy and return two arrays of the same shape:

        return (2.0 * MU * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y),
                np.zeros_like(x))
    """
    return np.zeros_like(x), np.zeros_like(y)
NX, NY    = 24, 16
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
DEALII_EXE = "./dealii_side"   # YOUR compiled solver; you write and build it (docstring)
# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                # FE_Q degree used inside the FESystem
LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))



def _partner_block(imp):
    """The partner's block from imports.json, by the name the driver actually used."""
    if not isinstance(imp, dict) or not imp:
        return None
    if PARTNER in imp:
        return imp[PARTNER] or None
    others = [k for k in imp if isinstance(imp.get(k), dict)]
    if len(others) == 1:     # named differently in couple(...) than here: read it, say so
        import sys as _sys
        print(f"NOTE: PARTNER is {PARTNER!r} but imports.json is keyed {others[0]!r} "
              f"-- reading that block; align PARTNER with the name in your "
              f"couple(...) call.", file=_sys.stderr)
        return imp[others[0]] or None
    if others:
        raise SystemExit(f"imports.json holds blocks named {others} and none is "
                         f"PARTNER={PARTNER!r}: set PARTNER to the partner's name "
                         f"in your couple(...) call.")
    return None


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return _partner_block(d)


def sample(imp, key, fallback):
    """Partner samples as sorted (y, vx, vy) triples; constant fallback if
    absent. The deal.II solver interpolates them piecewise-linearly PER
    COMPONENT (clamped outside the range) onto its own interface nodes — the
    same semantics as numpy.interp applied to each component separately."""
    if imp and imp.get("coordinates"):
        ys = [float(c[1]) for c in imp["coordinates"]]
        raw = imp.get(key) or []
        vs = []
        for v in raw:
            if isinstance(v, (list, tuple)):
                vs.append((float(v[0]), float(v[1])))
            else:                       # a scalar partner on a vector interface
                vs = []
                break
        if len(vs) == len(ys) and len(ys) > 0:
            return sorted((y, v[0], v[1]) for y, v in zip(ys, vs))
    fx, fy = float(fallback[0]), float(fallback[1])
    return [(float(Y0), fx, fy), (float(Y1), fx, fy)]


imp = read_imports()
if SIDE == "dirichlet":
    side_flag, triples = 0, sample(imp, "values", (UI_X, UI_Y))
else:
    side_flag, triples = 1, sample(imp, "normal_fluxes", (TI_X, TI_Y))

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
lines = [f"{side_flag} {E_MOD!r} {NU!r} {X0!r} {X1!r} {Y0!r} {Y1!r} "
         f"{IFACE_X!r} {NX} {NY} {DEGREE}",
         " ".join(repr(float(c)) for c in UDX),
         " ".join(repr(float(c)) for c in UDY),
         str(len(triples))]
lines += [f"{y:.16g} {vx:.16g} {vy:.16g}" for y, vx, vy in triples]

# BODY FORCE. The compiled solver assembles + \int b . v dx from samples of
# B_SRC on a UNIFORM tensor grid and interpolates them bilinearly, so the grid
# is put exactly on the FE nodes (NX*DEGREE+1 by NY*DEGREE+1): the interpolant
# the solver integrates is then the Q1 interpolant of the source, the same
# semantics as the Python participants. The block is appended LAST because the
# solver reads it optionally -- but note that a solver binary built before this
# block existed reads the file up to the samples and STOPS, so it would ignore
# the body force silently. Rebuild elast_iface_dealii after changing B_SRC for
# the first time.
nbx, nby = NX * DEGREE + 1, NY * DEGREE + 1
gx, gy = np.meshgrid(np.linspace(X0, X1, nbx), np.linspace(Y0, Y1, nby),
                     indexing="ij")
bx = np.broadcast_to(np.asarray(B_SRC(gx, gy)[0], float), gx.shape)
by = np.broadcast_to(np.asarray(B_SRC(gx, gy)[1], float), gx.shape)
lines.append(f"{nbx} {nby}")
flat = np.empty(2 * bx.size)
flat[0::2] = bx.ravel(order="F")       # x index fastest, as the solver reads it
flat[1::2] = by.ravel(order="F")
lines += [" ".join(f"{val:.16g}" for val in flat[k:k + 2 * nbx])
          for k in range(0, flat.size, 2 * nbx)]
Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

out_txt = Path("dealii_output.txt")
if out_txt.exists():
    out_txt.unlink()
r = subprocess.run([DEALII_EXE, "dealii_input.txt", "dealii_output.txt"],
                   capture_output=True, text=True)
if r.returncode != 0 or not out_txt.is_file():
    sys.stderr.write("deal.II elasticity solver failed (rc=%s)\n%s\n%s\n"
                     % (r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
    sys.exit(1)

# PASS THE SOLVER'S OWN CONSOLE THROUGH. capture_output keeps it out of this
# script's stdout, and the per-level run log your task asks for is exactly that
# console -- a log carrying only this wrapper's prose cannot establish which
# code ran on this side. Re-emitting it costs nothing and is the difference
# between a log that counts and one that does not. The `NDOF = <integer>` line
# the log contract needs comes from YOUR program: print it there, on a line of
# its own, and it arrives here.
if r.stdout:
    print(r.stdout, end="")
if r.stderr:
    sys.stderr.write(r.stderr)

# Unlike the pure-Python participants, this one talks to a COMPILED binary, so
# the script and the solver can disagree about what the input file contains. A
# binary built before the body-force block existed stops reading at the samples
# and drops the body force without a word — and a dropped body force returns
# u = 0, which is the failure this block was added to prevent, now wearing a
# participant that looks correct. The solver therefore announces what it read,
# and a non-zero B_SRC that produced no announcement is a hard error, never a
# quiet zero.
if not any(ln.startswith("BODY_FORCE on") for ln in r.stdout.splitlines()):
    if float(np.abs(bx).max()) > 0.0 or float(np.abs(by).max()) > 0.0:
        sys.stderr.write(
            "B_SRC is non-zero but the solver did not report reading a body "
            "force. The binary at %s is older than the input this script "
            "writes: rebuild your solver. Refusing to return a "
            "result that silently ignores the source term.\n" % DEALII_EXE)
        sys.exit(1)

coords, disp, trac = [], [], []
for line in out_txt.read_text().splitlines():
    tok = line.split()
    if len(tok) != 5:
        continue
    coords.append([float(IFACE_X), float(tok[0])])
    disp.append([float(tok[1]), float(tok[2])])
    trac.append([float(tok[3]), float(tok[4])])
if not coords:
    sys.stderr.write("deal.II solver produced no interface points\n")
    sys.exit(1)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a traction of ~0 against a nonzero partner); and a traction that is the
#    partner's array negated instead of a recovery from THIS side's own system.
#    A VECTOR side needs it more, not less: a displacement field that came out
#    ~0 because the load never arrived still couples, still converges and still
#    hands in three tidy levels.
_chk_vals = np.asarray(disp, float).ravel()
_chk_flux = np.asarray(trac, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or "
                     "tractions; the solve did not produce a usable field, so "
                     "nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface traction is ~0 "
                     "against a nonzero imported traction: the imported load "
                     "never entered the assembled system (the facet term / "
                     "boundary condition that integrates it is missing). Fix "
                     "the application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied traction can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported traction is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": len(coords),
    "coordinates": coords,
    "values": disp,
    "normal_fluxes": trac,
}, indent=2))
