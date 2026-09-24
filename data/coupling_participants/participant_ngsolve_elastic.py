"""NGSolve VECTOR participant for the openPASO `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X (or
y = IFACE_X when IFACE_AXIS is "y"). Unlike the
scalar (heat) participants, the exchanged interface state is a VECTOR on BOTH
channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

SIGN CONVENTION — the thing a vector coupling gets wrong silently.
`normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_x

the SAME convention the shipped scalar participants use for heat
(q_out = -k dT/dn_own). The two sides' exports therefore CANCEL componentwise,
and the NEUMANN side applies the partner's numbers UNCHANGED
(`f += InnerProduct(g, v) * ds("interface")`), because the natural boundary
term of the elasticity weak form is +(sigma . n_own) . v = +q_out_partner . v.

NGSolve's VectorH1 is BLOCKED BY COMPONENT: GetDofNrs(NodeId(VERTEX, i))
returns (dof of u_x, dof of u_y) and those two indices are nv apart, not
adjacent. Writing nodal values with an adjacency assumption silently scatters
u_y into the u_x block.
"""
import json
import os
from pathlib import Path

import ngsolve                # the MODULE, so ngsolve.ngsglobals.msg_level = 3 resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from ngsolve import ...` alone leaves `ngsolve` undefined
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, CF, GridFunction, InnerProduct,
                     LinearForm, Mesh, NodeId, TaskManager, VectorH1, ds, dx,
                     grad)


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.55     # this subdomain
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.55          # WHERE that line sits: equal to X0 or X1 for axis "x",
                          # to Y0 or Y1 for axis "y"
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

    HOW THIS ENTERS NGSolve (your solve below has to do it): NGSolve's symbolic
    `x`, `y` are CoefficientFunctions and carry NO NumPy ufuncs, so do NOT call
    this function on them (np.sin(x) raises; np.zeros_like(x) silently returns
    a 0-d object array and the source collapses to a constant), and do NOT wrap
    the function -- CoefficientFunction(B_SRC) is a TypeError ("incompatible
    constructor arguments", measured). Sample it at the mesh vertices instead,
    np.array([v.point for v in mesh.vertices]), into one P1 GridFunction per component on your
    space (gf.vec.FV().NumPy()[vertex_dofs] = B_SRC(vx, vy)[i]); a GridFunction IS
    a CoefficientFunction and integrates as gf_x * v[0] * dx + gf_y * v[1] * dx -- the P1 interpolant
    of the source, quadrature error O(h^2), the order of the discretisation.
    """
    return np.zeros_like(x), np.zeros_like(y)
NX, NY    = 24, 16        # this subdomain's own mesh (netgen maxh derived below)

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

UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
# ─────────────────────────────────────────────────────────────────────────

MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)    # netgen's scalar mesh size
ORDER = 1                                     # nodal == vertex dofs

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
ON_RIGHT = abs(IFACE_X - HI) < abs(IFACE_X - LO)   # interface at this side's MAX of that axis?
OUTER_X = LO if ON_RIGHT else HI           # the opposite face, on the same axis
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_AX
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)



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


def sample(imp, key, fallback, y):
    """Map the partner's VECTOR samples onto this participant's y-coordinates,
    COMPONENT BY COMPONENT. One np.interp over a flattened (N, 2) array
    interleaves the components: right length, converges, every number wrong.

    Returns (len(y), ncomp)."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[AL] for c in imp["coordinates"]], float)   # the coordinate ALONG the interface
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, c])
                            for c in range(vs.shape[1])])


imp = read_imports()

# MAKE THIS CODE SPEAK, BEFORE THE SOLVE RUNS. It is silent by default, and a
# per-level run log carrying no line the solver itself emitted cannot
# establish which code ran on this side, however right its numbers are.
# It sits HERE, beside the level rule, and not up with the imports:
# measured over agent-written participants, a line placed in the import
# block survived in about half of them because that block gets rewritten,
# while everything beside the level rule survived in all of them.
ngsolve.ngsglobals.msg_level = 3

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# ── mesh: SplineGeometry.AddRectangle edge order is bottom, right, top, left ──
# Everything that is not the interface carries the prescribed displacement, so
# it all gets the same boundary name.
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("outer", "interface", "outer", "outer") if ON_RIGHT else
                      ("outer", "outer", "outer", "interface")))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = VectorH1(mesh, order=ORDER,
               dirichlet=("outer|interface" if SIDE == "dirichlet" else "outer"))
u, v = fes.TnT()

# vertex -> (x-dof, y-dof). VectorH1 blocks by COMPONENT, so these are nv apart.
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[:2] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]           # sorted by y
y_if = vxy[iface_v, 1]
outer_v = np.where((np.abs(vxy[:, 0] - OUTER_X) < TOL) |
                   (np.abs(vxy[:, 1] - Y0) < TOL) |
                   (np.abs(vxy[:, 1] - Y1) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES: they
# sit on a y-face, which carries a prescribed displacement in the un-split
# problem. Leaving them to the interface leaves them unconstrained on the
# Neumann side — the subproblem is still well posed, still converges, and lands
# a few percent off. They are still EXPORTED, just not interface-imposed.
corner = (np.abs(y_if - Y0) < TOL) | (np.abs(y_if - Y1) < TOL)


def eps_of(g):
    exx, eyy = g[0, 0], g[1, 1]
    return exx, eyy, 0.5 * (g[0, 1] + g[1, 0])


a = BilinearForm(fes)
gu, gv = grad(u), grad(v)
eu = eps_of(gu)
ev = eps_of(gv)
a += (2.0 * MU * (eu[0] * ev[0] + eu[1] * ev[1] + 2.0 * eu[2] * ev[2])
      + LAM * (eu[0] + eu[1]) * (ev[0] + ev[1])) * dx
f = LinearForm(fes)
# BODY FORCE. B_SRC is sampled at the vertices and carried by a GridFunction,
# which IS a CoefficientFunction, so the linear form's source varies in space
# instead of being the constant it used to be. ORDER = 1, so this is the P1
# interpolant of the source; its quadrature error is O(h^2), the same order as
# the P1 discretization error itself. Do NOT hand B_SRC ngsolve's symbolic x, y
# instead: a CoefficientFunction carries no NumPy ufuncs, so np.sin(x) raises
# and np.zeros_like(x) returns a 0-d OBJECT array — the source collapses to a
# constant and the subdomain solves the wrong problem with no error raised.
# Nodal writes go through vdof, because VectorH1 blocks BY COMPONENT.
gfb = GridFunction(fes)
gfb.vec[:] = 0.0
bx, by = B_SRC(vxy[:, 0], vxy[:, 1])
bvals = gfb.vec.FV().NumPy()
bvals[vdof[:, 0]] = np.broadcast_to(np.asarray(bx, float), (mesh.nv,))
bvals[vdof[:, 1]] = np.broadcast_to(np.asarray(by, float), (mesh.nv,))
f += InnerProduct(gfb, v) * dx
# THE VOLUME LOAD ALONE, in its own form. The Neumann branch adds the partner's
# interface term into `f`; the traction recovery at the bottom must subtract
# the volume load WITHOUT it, on both sides — subtracting the combined vector
# is what made the reaction look like zero on the Neumann side.
f_vol = LinearForm(fes)
f_vol += InnerProduct(gfb, v) * dx

gfu = GridFunction(fes)                    # also carries the Dirichlet data
gfu.vec[:] = 0.0
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    for k, vtx in enumerate(iface_v):
        if corner[k]:
            continue
        gfu.vec[int(vdof[vtx, 0])] = float(u_if[k, 0])
        gfu.vec[int(vdof[vtx, 1])] = float(u_if[k, 1])
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for k, vtx in enumerate(iface_v):
        gfun.vec[int(vdof[vtx, 0])] = float(t_if[k, 0])
        gfun.vec[int(vdof[vtx, 1])] = float(t_if[k, 1])
    # APPLY the partner's numbers UNCHANGED
    f += InnerProduct(gfun, v) * ds("interface")

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# The outer boundary is written LAST so it wins at the two interface corners.
ox, oy = vxy[outer_v, 0], vxy[outer_v, 1]
oux = UDX[0] + UDX[1] * ox + UDX[2] * oy + UDX[3] * oy * oy
ouy = UDY[0] + UDY[1] * ox + UDY[2] * oy + UDY[3] * oy * oy
for k, vtx in enumerate(outer_v):
    gfu.vec[int(vdof[vtx, 0])] = float(oux[k])
    gfu.vec[int(vdof[vtx, 1])] = float(ouy[k])
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

with TaskManager():
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    a.Assemble()
    f.Assemble()
    f_vol.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(),
                                  inverse="sparsecholesky") * res
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # Interface traction export q_out = -(sigma . n_own).
    #
    # WHY NOT AN L2 PROJECTION OF THE STRESS. That is what this file used to do
    # on the Neumann side: project -(sigma(u_h) . n_own) over the whole
    # subdomain and sample it at the interface. The gradient of a P1 solution —
    # and therefore the stress — is only O(h) accurate ON the boundary; the
    # superconvergence points are interior, and the boundary trace is exactly
    # what the coupling reads.
    #
    # THE CONSISTENT (REACTION) TRACTION. From
    #     a(u,v) - (f,v) = int_dOmega (sigma(u).n).v ds = -int_Gamma q_out.v ds
    # (the second equality is this file's sign convention) it follows that for
    # every vector basis function phi_i on the interface
    #     int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b_vol
    # with r the UNCONSTRAINED residual: NGSolve's a.mat and the load vectors
    # are exactly that — the Dirichlet condition lives in fes.FreeDofs() at
    # solve time and never touches the assembled operator, so the constrained
    # rows still carry the reaction.
    #
    # ONE FORMULA, BOTH SIDES. An earlier version used that reaction on the
    # Dirichlet side and the projection on the Neumann side, reasoning that the
    # Neumann interface dofs are free so r comes out ~0 there. That holds only
    # when the residual is taken against a load that ALREADY CONTAINS the
    # interface term. Subtract the VOLUME load alone and those same rows carry
    # exactly the interface functional the partner applied:
    #     (A u - b_vol)_i = int_Gamma g . phi_i ds
    # On the Dirichlet side there is no interface term, so f_vol == f and the
    # two cases are one expression.
    #
    # WHAT IS MEASURED, AND WHAT IS ONLY ALGEBRA. Handing the NEUMANN side a
    # traction and asking for it back is an ASSEMBLY IDENTITY, not a
    # convergence test: on free interface rows r = A u - b_vol IS M_Gamma g, so
    # the export is -(M_Gamma g)/(M_Gamma 1) and its offset from -g is
    # -(h^2/6) g''(y) for ANY correct assembly of ANY equation. The "order
    # 2.00" that used to stand here was read off that fixture; it is a property
    # of the P1 boundary mass matrix, not of this code — a bare NumPy mass
    # matrix reproduces the same numbers with no PDE, no solver and no material
    # in it. That fixture is kept (tests/test_interface_flux_recovery.py) for
    # what it really tests, and for a VECTOR participant the blocked-dof
    # mapping below is exactly the kind of defect it catches.
    #
    # THE ORDER is measured on the DIRICHLET side against an ANALYTIC interface
    # flux the participant is never handed
    # (tests/test_interface_flux_converges_to_a_known_exact_flux.py). Scalar
    # conduction, FEniCSx, the same recovery, 8/16/32/64 uniform triangle
    # meshes, max error over interior interface nodes:
    #   2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996 1.998 1.993
    # and only first order (1.10, 1.06, 1.04) at the two nodes where the
    # interface meets the outer boundary, handled apart below. There is no
    # VECTOR measurement against an analytic traction, and none is claimed.
    #
    # THE RETIRED L2-PROJECTED GRADIENT, in the norms it was measured in: order
    # ~1 in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and
    # non-convergent in the max norm that includes the near-end nodes, where it
    # stalls at 2.6 against a true flux of size 2 to 5. It was written up as a
    # flat "order 0.00, it never converges", which was true of one norm only.
    # Not re-measured since the branch was deleted.
    #
    # THE WEIGHT IS ONE SCALAR PER NODE, NOT ONE PER DOF. w_i = int_Gamma phi_i
    # ds belongs to the NODE, while VectorH1 blocks the dofs BY COMPONENT; the
    # test vector (1,1) in the weight form puts that same number on both of a
    # node's dofs, so the loop below divides every component by its own node's
    # weight and reaches it through vdof, never by assuming adjacency.
    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec   # r = A u_h - b_vol, no bc here
    fw = LinearForm(fes)
    fw += InnerProduct(CF((1.0, 1.0)), v) * ds("interface")
    fw.Assemble()

    Q = np.zeros((len(iface_v), 2))
    ok = np.ones((len(iface_v), 2), bool)
    for k, vtx in enumerate(iface_v):
        for c in (0, 1):
            d = int(vdof[vtx, c])
            wi = float(fw.vec[d])
            if abs(wi) > 1e-14:
                Q[k, c] = -float(rvec[d]) / wi
            else:
                ok[k, c] = False

    # THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (a y-face),
    # so their rows carry the OUTER reaction too and their residual is not this
    # interface's traction. Take the nearest interior interface node rather
    # than exporting a corner value that is physically a different quantity.
    # This holds on BOTH sides: the corners are outer-Dirichlet either way.
    suspect = np.isin(iface_v, outer_v) | ~ok.all(axis=1)
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN.
# The audit and the hand-in read that exact shape, and they read it PER
# LEVEL: it is how a grader tells a refined mesh from the same mesh run
# three times. The LEADING NEWLINE is deliberate -- a program that writes
# without a trailing newline glues its text onto the front of the next
# line, and an X11 warning has done exactly that here, turning a correct
# line into 'Invalid MIT-MAGIC-COOKIE-1 keyNDOF = 54'.
# A number inside a prose sentence does not count either, and a
# wrong number is worse than none -- one coupled run that was right in
# every other respect reported NDOF = 1 at all three levels, and its
# refined mesh could not be told from an unrefined one.
try:
    print(f"\nNDOF = {int(fes.ndof)}")
except Exception as _ndof_exc:
    print(f"[ngsolve] could not report NDOF: {_ndof_exc!r}. Your task's"
          f" execution log needs `NDOF = <integer>` on a line of its own,"
          f" so print your own degree-of-freedom count here.")

# PER-LEVEL PERSISTENCE: this level's whole field, and its interface trace and
# traction, named by LEVEL. exports.json is overwritten by the next level;
# these files are not.
# the probe points your task names -- never a file the next level overwrites.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    with open(f"field_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy\n")
        for _i in range(mesh.nv):
            _p = mesh.vertices[_i].point
            _f.write(f"{float(_p[0]):.11e},{float(_p[1]):.11e},"
                     f"{float(gfu.vec[int(vdof[_i, 0])]):.11e},"
                     f"{float(gfu.vec[int(vdof[_i, 1])]):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy,qx,qy\n")
        for _yy, (_ux, _uy), (_qx, _qy) in zip(y_if, [(gfu.vec[int(vdof[_n, 0])], gfu.vec[int(vdof[_n, 1])]) for _n in iface_v], Q):
            _px, _py = ((float(IFACE_X), float(_yy)) if AX == 0
                        else (float(_yy), float(IFACE_X)))
            _f.write(f"{_px:.11e},{_py:.11e},{float(_ux):.11e},"
                     f"{float(_uy):.11e},{float(_qx):.11e},{float(_qy):.11e}\n")
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
    print(f"[ngsolve_elastic per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a traction of ~0 against a nonzero partner); and a traction that is the
#    partner's array negated instead of a recovery from THIS side's own system.
#    A VECTOR side needs it more, not less: a displacement field that came out
#    ~0 because the load never arrived still couples, still converges and still
#    hands in three tidy levels.
_chk_vals = np.asarray([[gfu.vec[int(vdof[i, 0])], gfu.vec[int(vdof[i, 1])]] for i in iface_v], float).ravel()
_chk_flux = np.asarray(Q, float).ravel()
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
                     "boundary condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied traction can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported traction is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_v)),
    "coordinates": [([float(IFACE_X), float(yy)] if AX == 0 else [float(yy), float(IFACE_X)])
                    for yy in y_if],
    "values": [[float(gfu.vec[int(vdof[i, 0])]), float(gfu.vec[int(vdof[i, 1])])]
               for i in iface_v],
    "normal_fluxes": [[float(q0), float(q1)] for q0, q1 in Q],
}, indent=2))
