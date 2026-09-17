"""deal.II participant for the openPASO `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
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
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.6
K         = 0.8


def F_SRC(x, y):
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)

    THIS PARTICIPANT DRIVES A COMPILED SOLVER: your solve region below has to
    carry the samples of F_SRC to your binary (for example on a uniform grid in
    the input file) and your binary has to assemble the volume integral of f times v from them. A
    solver that never reads the source returns the boundary-data-only solution
    with no error anywhere, so have the binary announce what it read and make
    your wrapper refuse a non-zero F_SRC that produced no announcement.
    """
    return np.zeros_like(x)
T_OUTER   = 320.0
NX, NY    = 24, 16
T_INIT    = 310.0
Q_INIT    = 0.0           # iteration-1 fallback interface flux
DEALII_EXE = "./dealii_side"   # YOUR compiled solver; you write and build it (docstring)
# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                # FE_Q degree used by the deal.II solver


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback):
    """Partner samples as sorted (y, value) pairs; constant fallback if absent.

    The deal.II solver interpolates them piecewise-linearly (clamped outside
    the range) onto its own interface nodes — same semantics as numpy.interp.
    """
    if imp and imp.get("coordinates"):
        ys = [float(c[1]) for c in imp["coordinates"]]
        vs = [float(v) for v in (imp.get(key) or [])]
        if len(vs) == len(ys) and len(ys) > 0:
            return sorted(zip(ys, vs))
    return [(float(Y0), float(fallback)), (float(Y1), float(fallback))]


imp = read_imports()
if SIDE == "dirichlet":
    side_flag, pairs = 0, sample(imp, "values", T_INIT)
else:
    side_flag, pairs = 1, sample(imp, "normal_fluxes", Q_INIT)

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides the mesh knobs and names the level. The dumps
#    at the foot of this file carry that level in their NAME, so a mesh study
#    leaves one file per level instead of the fine mesh overwriting the coarse.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

FIELD_OUT = "field_out.txt"   # your .cc writes "x y u" per support point here

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# The ninth header field is the solver's LEGACY CONSTANT source. It is kept in
# the file format so a solver binary built before the sampled block below still
# parses this header, and it is always written as 0.0: the real source is the
# sampled grid appended at the end, which the solver ADDS to it.
header = (f"{side_flag} {K!r} {X0!r} {X1!r} {Y0!r} {Y1!r} {IFACE_X!r} "
          f"{T_OUTER!r} 0.0 {NX} {NY} {DEGREE}")
lines = [header, str(len(pairs))]
lines += [f"{y:.16g} {v:.16g}" for y, v in pairs]

# VOLUMETRIC SOURCE. The compiled solver assembles + \int f v dx from samples of
# F_SRC on a UNIFORM tensor grid and interpolates them bilinearly, so the grid is
# put exactly on the FE nodes (NX*DEGREE+1 by NY*DEGREE+1): the interpolant the
# solver integrates is then the Q1 interpolant of the source, the same semantics
# as the Python participants. The block is appended LAST because the solver reads
# it optionally -- but note that a solver binary built before this block existed
# reads the file up to the samples and STOPS, so it would ignore the source
# silently. Rebuild heat_iface_dealii after changing F_SRC for the first time.
nfx, nfy = NX * DEGREE + 1, NY * DEGREE + 1
gx, gy = np.meshgrid(np.linspace(X0, X1, nfx), np.linspace(Y0, Y1, nfy),
                     indexing="ij")
fsrc = np.broadcast_to(np.asarray(F_SRC(gx, gy), float), gx.shape)
lines.append(f"{nfx} {nfy}")
flat = fsrc.ravel(order="F")           # x index fastest, as the solver reads it
lines += [" ".join(f"{val:.16g}" for val in flat[k:k + nfx])
          for k in range(0, flat.size, nfx)]
Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

out_txt = Path("dealii_output.txt")
if out_txt.exists():
    out_txt.unlink()
r = subprocess.run([DEALII_EXE, "dealii_input.txt", "dealii_output.txt"],
                   capture_output=True, text=True)
if r.returncode != 0 or not out_txt.is_file():
    sys.stderr.write("deal.II solver failed (rc=%s)\n%s\n%s\n"
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
# binary built before the volume-source block existed stops reading at the
# samples and drops the source without a word — and a dropped source returns the
# boundary-data-only solution, which is the failure this block was added to
# prevent, now wearing a participant that looks correct. The solver therefore
# announces what it read, and a non-zero F_SRC that produced no announcement is
# a hard error, never a quiet zero.
if not any(ln.startswith("VOLUME_SOURCE on") for ln in r.stdout.splitlines()):
    if float(np.abs(fsrc).max()) > 0.0:
        sys.stderr.write(
            "F_SRC is non-zero but the solver did not report reading a volume "
            "source. The binary at %s is older than the input this script "
            "writes: rebuild your solver. Refusing to return a result "
            "that silently ignores the source term.\n" % DEALII_EXE)
        sys.exit(1)

coords, temps, fluxes = [], [], []
for line in out_txt.read_text().splitlines():
    tok = line.split()
    if len(tok) != 3:
        continue
    coords.append([float(IFACE_X), float(tok[0])])
    temps.append(float(tok[1]))
    fluxes.append(float(tok[2]))
if not coords:
    sys.stderr.write("deal.II solver produced no interface points\n")
    sys.exit(1)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(temps, float).ravel()
_chk_flux = np.asarray(fluxes, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was "
                     "exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load never "
                     "entered the assembled system (the facet term / boundary "
                     "condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

# PER-LEVEL PERSISTENCE: this level's interface trace and flux, and -- when the
# solver wrote one -- this level's field, named by LEVEL. exports.json is
# overwritten by the next level; these files are not.
# Interpolate THESE onto the probe points your task names. A file the next
# level overwrites cannot carry a mesh study.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u,qn\n")
        for (_px, _py), _t, _q in zip(coords, temps, fluxes):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},"
                     f"{float(_t):.11e},{float(_q):.11e}\n")
    # THE FIELD COMES FROM YOUR OWN PROGRAM. deal.II is C++: this wrapper only runs
    # your binary and reads what it printed, so the whole-domain field exists only
    # if your .cc writes it. Have the program write FIELD_OUT -- one "x y u" line
    # per support point, full precision -- next to its interface output, and this
    # block turns it into the per-level file. Without it there is no field to hand
    # in at any level, whatever the solve did.
    if Path(FIELD_OUT).is_file():
        _rows = []
        for _ln in Path(FIELD_OUT).read_text().splitlines():
            _tok = _ln.split()
            if len(_tok) != 3:
                continue
            try:
                _rows.append([float(_t) for _t in _tok])
            except ValueError:
                continue
        if _rows:
            with open(f"field_level{LEVEL}.csv", "w") as _f:
                _f.write("x,y,u\n")
                for _px, _py, _u in _rows:
                    _f.write(f"{_px:.11e},{_py:.11e},{_u:.11e}\n")
            print(f"[dealii {SIDE}] field_level{LEVEL}.csv: {len(_rows)} points")
    else:
        print(f"[dealii {SIDE}] NO {FIELD_OUT}: your program printed the interface "
              f"only, so this level has no field to hand in. Write one "
              f"'x y u' line per support point to {FIELD_OUT} and run it again.")
except Exception as _dump_exc:
    # AND LEAVE NO HALF-WRITTEN FILE BEHIND. `open(..., "w")` truncates
    # before it fails, so a dump that died mid-way leaves a header-only
    # CSV -- a file that looks like a submission and carries no rows.
    for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
        try:
            if Path(_partial).is_file() and len(
                    Path(_partial).read_text().splitlines()) <= 1:
                Path(_partial).unlink()
        except OSError:
            pass
    print(f"[dealii per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(coords),
    "coordinates": coords,
    "values": temps,
    "normal_fluxes": fluxes,
}, indent=2))
