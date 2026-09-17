"""deal.II TRANSIENT participant for the openPASO `couple` driver.

Transient conduction  rho_c dT/dt - div(k grad T) = f(x, y, t)  on ONE
rectangular subdomain of a domain split by a straight interface at x = IFACE_X,
integrated with the theta-scheme (THETA = 0.5 is Crank-Nicolson, second order).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST, exits 0.

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
Check cmake's "Using the deal.II-X found at" line: pointing DEAL_II_DIR at a
deal.II SOURCE tree silently falls back to whatever old deal.II is installed
system-wide, and the build then fails or misbehaves for reasons that look like
your code.

TIME-COUPLING STRATEGY: WAVEFORM — one run marches the WHOLE window
T_START -> T_END and exchanges the ENTIRE interface trace,

    values[i][n]        = T at interface point i at time level t^(n+1)
    normal_fluxes[i][n] = THETA-AVERAGED outward normal flux density over step n

both (n_points, N_STEPS). The full argument for that choice, what coupling
once per time step would cost instead, and why a participant may NOT keep
hidden time state between driver iterations, are in the module docstring of
`participant_fenics_transient.py` — this file implements the same protocol and
interoperates with it directly (measured: FEniCSx Dirichlet + deal.II Neumann
on a manufactured two-material problem, order 2.0 in space-time at the final
time).

THE TWO PARTICIPANTS MUST BE GIVEN THE SAME T_START, T_END, N_STEPS AND THETA.
The payload has no time axis, so a mismatched window is invisible unless the
trace LENGTH differs — which the solver checks and refuses on.

THE THREE PROBLEM FUNCTIONS ARE muparser EXPRESSION STRINGS in the variables
x, y, t, not Python callables: a compiled backend cannot be handed a lambda.
muparser syntax is close to C, NOT to Python — `^` is the power operator,
`exp/sin/cos/log/sqrt/abs/if` exist, and there is no `**`.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material, BCs and time
#    window. As shipped this is the LEFT / Dirichlet side; the payload that
#    served this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.6      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 0.8           # conductivity
RHO_C     = 1.0           # VOLUMETRIC heat capacity rho*c  (NOT c alone)
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
T_START   = 0.0           # coupling window start  ─┐ BOTH participants must be
T_END     = 1.0           # coupling window end     ├─ given the SAME three
N_STEPS   = 20            # steps in the window     ─┘ numbers AND the same THETA
THETA     = 0.5           # 0.5 = Crank-Nicolson (2nd order) | 1.0 = backward
                          # Euler (L-stable, 1st order — caps the space-time
                          # order at 1 when you refine dt with h)
OUTER_FACES = "x"         # "x"  : only the non-interface x-face is Dirichlet,
                          #        the two y-faces are natural (zero flux)
                          # "all": the WHOLE non-interface boundary is Dirichlet
# muparser expressions in x, y, t (see the docstring: C syntax, `^` for powers).
T_INITIAL = "300.0"       # T(x, y, T_START); must agree with T_OUTER at T_START
                          # on the Dirichlet faces or the first step carries an
                          # initial layer that Crank-Nicolson answers with
                          # oscillations rather than with second order
T_OUTER   = "320.0"       # Dirichlet datum on the faces selected above
F_SRC     = "0.0"         # volumetric source
T_GUESS   = 300.0         # iteration-1 fallback interface temperature, and
Q_GUESS   = 0.0           # iteration-1 fallback interface flux. Unlike the
                          # FEniCSx participant, which uses its initial
                          # condition as the temperature fallback, this wrapper
                          # never evaluates T_INITIAL (it is a string for the
                          # C++ side), so set T_GUESS to T_INITIAL's interface
                          # value yourself.
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


def sample_trace(imp, key, fallback):
    """Partner samples as (y, [v_1 ... v_N]) rows sorted by y; a constant trace
    if there is nothing to read (iteration 1).

    The deal.II solver interpolates each COLUMN piecewise-linearly (clamped
    outside the range) onto its own interface nodes — same semantics as
    numpy.interp, applied per time level. Interpolating a flattened trace
    instead interleaves the time levels: the length is still right, the
    coupling still converges, and every number is wrong.

    A trace of the WRONG NUMBER OF TIME LEVELS EXITS LOUDLY and does not fall
    back. That distinction is the whole guard: the fallback is what iteration 1
    looks like, so quietly using it for a partner whose window disagrees turns
    a mismatched window into a run that converges to the initial guess at the
    interface. Measured before this was split out: a 7-step partner against a
    10-step participant produced rc=0, an exports.json, and a final interface
    temperature of exactly the fallback constant, with nothing in the run
    saying the partner's data had been dropped. The solver's own check never
    fired, because the wrapper had already sanitised the input away.
    """
    if not imp or not imp.get("coordinates"):
        return [(float(Y0), [float(fallback)] * N_STEPS),      # iteration 1
                (float(Y1), [float(fallback)] * N_STEPS)]
    ys = [float(c[1]) for c in imp["coordinates"]]
    vs = imp.get(key) or []
    if len(vs) != len(ys) or not ys:
        return [(float(Y0), [float(fallback)] * N_STEPS),
                (float(Y1), [float(fallback)] * N_STEPS)]
    rows = [(y, [float(v)] if not isinstance(v, (list, tuple))
             else [float(z) for z in v]) for y, v in zip(ys, vs)]
    widths = {len(r[1]) for r in rows}
    if widths != {N_STEPS}:
        sys.exit(f"partner '{PARTNER}' exported a trace with {sorted(widths)} "
                 f"time levels, this participant's window has "
                 f"N_STEPS={N_STEPS}. The exchange carries no time axis, so a "
                 f"trace of the WRONG LENGTH is the only symptom a mismatched "
                 f"window shows. Give both participants the same "
                 f"T_START/T_END/N_STEPS/THETA.")
    return sorted(rows)


imp = read_imports()
if SIDE == "dirichlet":
    side_flag, pairs = 0, sample_trace(imp, "values", T_GUESS)
else:
    side_flag, pairs = 1, sample_trace(imp, "normal_fluxes", Q_GUESS)

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

FIELD_OUT = "field_out.txt"   # your .cc writes "x y u" per support point here,
                              # at the END of the time window

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
header = (f"{side_flag} {K!r} {RHO_C!r} {X0!r} {X1!r} {Y0!r} {Y1!r} {IFACE_X!r} "
          f"{NX} {NY} {DEGREE} {THETA!r} {T_START!r} {T_END!r} {N_STEPS} "
          f"{1 if OUTER_FACES == 'all' else 0}")
lines = [header, T_INITIAL.strip(), T_OUTER.strip(), F_SRC.strip(),
         f"{len(pairs)} {N_STEPS}"]
lines += [f"{y:.16g} " + " ".join(f"{v:.16g}" for v in row) for y, row in pairs]
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

raw = out_txt.read_text().split()
if len(raw) < 2:
    sys.stderr.write("deal.II solver produced no interface points\n")
    sys.exit(1)
n_nodes, n_steps_out = int(raw[0]), int(raw[1])
vals = [float(v) for v in raw[2:]]
stride = 1 + 2 * n_steps_out
if n_steps_out != N_STEPS or len(vals) != n_nodes * stride:
    sys.stderr.write(f"deal.II output is malformed: {n_nodes} nodes x "
                     f"{n_steps_out} steps needs {n_nodes * stride} numbers, "
                     f"got {len(vals)}\n")
    sys.exit(1)

coords, temps, fluxes = [], [], []
for i in range(n_nodes):
    row = vals[i * stride:(i + 1) * stride]
    coords.append([float(IFACE_X), row[0]])
    temps.append(row[1:1 + n_steps_out])
    fluxes.append(row[1 + n_steps_out:])
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

print(f"[dealii-transient {SIDE}] iface n={n_nodes} steps={N_STEPS} "
      f"dt={(T_END - T_START) / N_STEPS:.6g} theta={THETA} "
      f"T(t_end)=[{min(r[-1] for r in temps):.6g},"
      f"{max(r[-1] for r in temps):.6g}] "
      f"q(last step)=[{min(r[-1] for r in fluxes):.6g},"
      f"{max(r[-1] for r in fluxes):.6g}]")

# PER-LEVEL PERSISTENCE: the interface trace and flux at the LAST step, and --
# when the program wrote one -- the end-of-window field, named by LEVEL.
# exports.json is overwritten by the next level; these files are not.
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
                     f"{float(_t[-1]):.11e},{float(_q[-1]):.11e}\n")
    # THE FIELD COMES FROM YOUR OWN PROGRAM. deal.II is C++: this wrapper only runs
    # your binary and reads what it printed, so the whole-domain field exists only
    # if your .cc writes it. Have the program write FIELD_OUT -- one "x y u" line
    # per support point at the end of the window -- and this block turns it into the
    # per-level file. Without it there is no field to hand in at any level.
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
            print(f"[dealii-transient {SIDE}] field_level{LEVEL}.csv: {len(_rows)} points")
    else:
        print(f"[dealii-transient {SIDE}] NO {FIELD_OUT}: your program printed the "
              f"interface only, so this level has no field to hand in. Write one "
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
    print(f"[dealii_transient per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

# exports.json LAST: the driver takes its existence as proof of success.
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the
#    partner's array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(temps, float).ravel()
_chk_flux = np.asarray(fluxes, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or "
                     "fluxs; the solve did not produce a usable field, so "
                     "nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load "
                     "never entered the assembled system (the facet term / "
                     "boundary condition that integrates it is missing). Fix "
                     "the application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied load can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": n_nodes,
    "coordinates": coords,
    "values": temps,
    "normal_fluxes": fluxes,
}, indent=2))
