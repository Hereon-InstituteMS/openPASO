"""SPARTA (DSMC) participant for the openPASO `couple` driver.

SPARTA is a rarefied-gas DSMC particle code, not a FEM code, and has no
scripting API: it reads a text input deck and writes text dump/log files.
This module is therefore a WRAPPER that, each coupling iteration:
  1. reads imports.json  -> per-surface-element WALL TEMPERATURE,
  2. writes `tsurf.in` in SPARTA's `custom surf ... file` format,
  3. writes the deck, which does
         custom      surf create tsurf float 0 file tsurf.in 1 tsurf
         surf_collide 1 diffuse s_tsurf 1.0      <- per-element Dirichlet T
         compute     1 surf all all etot         <- per-element heat flux
         fix         1 ave/surf all 1 N N c_1[1] ave one
         dump        1 surf all N flux.out id v1x v1y v2x v2y f_1 s_tsurf
     NOTE the dump must reference `f_1`, NOT `f_1[1]`: a fix ave/surf with a
     single input column is a per-surf VECTOR, and the [1] form aborts with
     "Dump surf fix does not compute per-surf array" (dump_surf.cpp:609).
  4. runs spa_serial,
  5. parses the ASCII surf dump and writes exports.json.

ROLE.  SPARTA is a DIRICHLET-type participant for conjugate heat transfer:
it IMPORTS a field value (wall temperature) and EXPORTS a flux (the net
energy flux the gas deposits on the wall, `etot`, positive = wall gains
energy = energy leaving the gas subdomain through the interface, i.e.
exactly SPEC.md's outward-normal q_out for the gas side).
SPARTA HAS NO NATIVE FLUX BC, so this script is the Dirichlet side: none of the
nine `surf_collide` styles takes a prescribed heat flux, and the four thermal
ones (`diffuse`, `cll`, `td`, `impulsive`) all take a temperature.
A flux can be imposed only INDIRECTLY: `fix surf/temp` turns a per-surf flux
into a per-surf temperature via the gray-body law q = sigma*emisurf*T^4, and it
accepts that flux from ANY per-surf compute or fix — SPARTA's own doc states it
"does not check that the specified compute/fix calculates an energy flux" — so
an imported flux does reach it through `custom surf ... file` plus
`fix ave/surf s_<name>`.  That route prescribes the wall temperature which would
RADIATE the imported flux; it does not constrain the gas-side flux, and it adds
an emissivity unrelated to the coupling.  It is NOT implemented below and was
NOT run.  See knowledge(topic='coupling', solver='sparta').

STOCHASTICITY.  DSMC output is a Monte-Carlo estimate.  With a fixed RNG seed
and identical input the run is bit-reproducible (so a fixed-point iteration
can appear to "converge" even when the physics has not); with a varying seed
the exported flux carries sampling noise that does NOT shrink with coupling
iterations, so the driver's relative-residual tolerance cannot be driven below
that noise floor.  SEED_MODE below selects which regime you are in.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "solid"      # partner participant name in couple(...)
SPARTA     = "spa_serial"    # the SPARTA binary path `discover(query='list')` prints
DATA_DIR   = ""              # dir holding the surf/species/vss files, "" = work_dir
SURF_FILE  = "circle.surf"     # 50 line elements (2D cylinder)
SPECIES    = "ar.species"
VSS        = "ar.vss"
T_INIT     = 500.0        # iteration-1 fallback wall temperature [K]
NRUN       = 4000         # DSMC timesteps
NAVE       = 2000         # sample the flux over the last NAVE steps
SEED_MODE  = "fixed"      # "fixed" -> same seed every iteration (deterministic)
                          # "vary"  -> new seed every iteration (honest noise)
SEED       = 12345
# ─────────────────────────────────────────────────────────────────────────

DECK = "cpl.sparta"
TSURF_IN = "tsurf.in"
FLUX_OUT = "flux.out"


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text() or "{}")
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def read_surf_elements():
    """Parse SPARTA's surf file -> (n_elem, centroids[n,2])."""
    txt = Path(SURF_FILE).read_text().splitlines()
    npts = nlines = 0
    for ln in txt:
        s = ln.split("#")[0].strip()
        if s.endswith("points"):
            npts = int(s.split()[0])
        elif s.endswith("lines"):
            nlines = int(s.split()[0])
    pts, lines, sec = {}, [], None
    for ln in txt:
        s = ln.split("#")[0].strip()
        if not s:
            continue
        low = s.lower()
        if low == "points":
            sec = "p"; continue
        if low == "lines":
            sec = "l"; continue
        if low.endswith("points") or low.endswith("lines"):
            continue
        f = s.split()
        if sec == "p" and len(f) >= 3:
            pts[int(f[0])] = (float(f[1]), float(f[2]))
        elif sec == "l" and len(f) >= 3:
            lines.append((int(f[0]), int(f[-2]), int(f[-1])))
    lines.sort()
    cen = np.array([[0.5 * (pts[a][0] + pts[b][0]),
                     0.5 * (pts[a][1] + pts[b][1])] for (_, a, b) in lines])
    assert len(lines) == nlines and len(pts) == npts, "surf file parse mismatch"
    return len(lines), cen
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def _arclen(p):
    """Normalised cumulative chord length along an ordered point list."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else d


def sample_on(imp, key, fallback, cen):
    """Map partner samples onto SPARTA's element centroids using normalised
    arc length along the interface curve as the 1D interface parameter
    (the curved-interface analogue of SPEC.md's tangential coordinate y).
    Both sides must list their interface points ordered along the curve."""
    n = len(cen)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    src = np.asarray(imp["coordinates"], float)[:, :2]
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != len(src) or len(src) < 2:
        return np.full(n, float(fallback))
    return np.interp(_arclen(cen), _arclen(src), vs)


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def write_tsurf(t):
    """SPARTA `custom surf ... file` format: comment, blank, 'N M', 'id v'."""
    L = ["# per-surf wall temperature written by the openPASO coupling driver", ""]
    L.append(f"{len(t)} 1")
    for i, v in enumerate(t, 1):
        L.append(f"{i} {float(v):.10g}")
    Path(TSURF_IN).write_text("\n".join(L) + "\n")


def write_deck(seed):
    Path(DECK).write_text(f"""\
seed                {seed}
dimension           2
global              nrho 4.247e19 fnum 7e14 gridcut 0.01 comm/style all comm/sort yes
timestep            3.5e-7
boundary            o ro p
create_box          -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid         30 15 1 block * * *
species             {SPECIES} Ar
mixture             all vstream 2634.1 0 0 temp 200.0
collide             vss all ar.vss
collide_modify      vremax 1000 yes
read_surf           {SURF_FILE} group 1

custom              surf create tsurf float 0 file {TSURF_IN} 1 tsurf
surf_collide        1 diffuse s_tsurf 1.0
surf_modify         1 collide 1

fix                 in emit/face all xlo twopass
create_particles    all n 0 twopass

compute             1 surf all all etot
fix                 1 ave/surf all 1 {NAVE} {NAVE} c_1[1] ave one
dump                1 surf all {NRUN} {FLUX_OUT} id v1x v1y v2x v2y f_1 s_tsurf
dump_modify         1 pad 0

stats               {NRUN}
stats_style         step cpu np nscoll
run                 {NRUN}
""")


def parse_dump(path, n):
    """Read the LAST snapshot of a SPARTA ASCII surf dump."""
    txt = Path(path).read_text().splitlines()
    starts = [i for i, l in enumerate(txt) if l.startswith("ITEM: SURFS")]
    if not starts:
        raise SystemExit(f"no 'ITEM: SURFS' block in {path}")
    i0 = starts[-1] + 1
    rows = []
    for l in txt[i0:]:
        if l.startswith("ITEM:"):
            break
        f = l.split()
        if len(f) >= 7:
            rows.append([float(x) for x in f[:7]])
    a = np.asarray(rows, float)
    a = a[np.argsort(a[:, 0])]
    if len(a) != n:
        raise SystemExit(f"dump has {len(a)} rows, expected {n}")
    return a


# --------------------------------------------------------------------- run
for f in (SURF_FILE, SPECIES, VSS):
    if not Path(f).is_file():
        src = Path(DATA_DIR) / f
        if not src.is_file():
            sys.stderr.write(f"missing SPARTA data file {f} (not in cwd, not in "
                             f"{DATA_DIR}). couple() DOES stage files a deck in work_dir "
                             f"references -- searching data_dir and data_files first, then "
                             f"SPARTA_DATA_DIR, then the distribution -- so this is a STANDALONE "
                             f"run: put the file beside the deck yourself, or pass data_dir.\n")
            sys.exit(3)
        Path(f).write_bytes(src.read_bytes())
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

n_elem, cen = read_surf_elements()
imp = read_imports()
t_wall = sample_on(imp, "values", T_INIT, cen)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
write_tsurf(t_wall)

it = 0
ctr = Path(".iter")
if ctr.is_file():
    it = int(ctr.read_text().strip() or 0)
ctr.write_text(str(it + 1))
seed = SEED if SEED_MODE == "fixed" else SEED + 1000 * (it + 1)
write_deck(seed)

Path(FLUX_OUT).unlink(missing_ok=True)
r = subprocess.run([SPARTA, "-in", DECK], capture_output=True, text=True, timeout=3600)
if r.returncode != 0 or not Path(FLUX_OUT).is_file():
    sys.stderr.write(f"SPARTA failed rc={r.returncode}\n{(r.stdout or '')[-1500:]}\n"
                     f"{(r.stderr or '')[-800:]}\n")
    sys.exit(1)

# PASS THE SOLVER'S OWN CONSOLE THROUGH. capture_output keeps SPARTA's banner,
# its step table and its particle counts out of this script's stdout, and the
# per-level run log your task asks for is exactly that console -- a log carrying
# only this wrapper's prose cannot establish which code ran on this side.
if r.stdout:
    print(r.stdout, end="")
if r.stderr:
    sys.stderr.write(r.stderr)
# THE RUN-LOG CONTRACT LINE, in this code's own currency. A DSMC cell refines by
# PARTICLE COUNT on a fixed grid, not by adding cells, so the `NDOF = <integer>`
# line the log contract asks for carries the particle count -- SPARTA reports it
# in the console above, and it is the number that grows between your levels.
# Print it on a line of its own from whatever your deck actually ran.

a = parse_dump(FLUX_OUT, n_elem)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
cx = 0.5 * (a[:, 1] + a[:, 3])
cy = 0.5 * (a[:, 2] + a[:, 4])
q_out = a[:, 5]          # etot: net energy flux INTO the wall = OUT of the gas
t_used = a[:, 6]

# ── EXPORT SELF-CHECK ─ keep this block. An export that is not finite is
#    worthless however well the iteration behaved, and a field that came out
#    identically zero still couples, still converges and still hands in tidy
#    levels. The DSMC tally is the load the solid sees.
_chk_vals = np.asarray(q_out, float).ravel()
if not np.isfinite(_chk_vals).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values; the "
                     "solve did not produce a usable field, so nothing was "
                     "exported")
if _chk_vals.size and np.abs(_chk_vals).max() == 0.0:
    raise SystemExit("EXPORT SELF-CHECK: every exported interface value is "
                     "exactly zero. That is the no-response answer, not a "
                     "solve: the partner's data never reached the assembled "
                     "system. Fix the application; do not couple on")

Path("exports.json").write_text(json.dumps({
    "field_name": "wall_temperature",
    "n_points": int(n_elem),
    "coordinates": [[float(x), float(y)] for x, y in zip(cx, cy)],
    "values": [float(v) for v in t_used],
    "normal_fluxes": [float(v) for v in q_out],
}, indent=2))
print(f"SPARTA it={it+1} seed={seed} n={n_elem} T_wall=[{t_wall.min():.2f},"
      f"{t_wall.max():.2f}] sum(etot)={q_out.sum():.6e} mean={q_out.mean():.6e}")
