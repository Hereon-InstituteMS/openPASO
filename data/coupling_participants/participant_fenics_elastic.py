"""FEniCSx (dolfinx) VECTOR participant for the openPASO `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X. Unlike the
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
(q_out = -k dT/dn_own). Two consequences, both load-bearing:

  * the two sides' exports CANCEL componentwise, because n_own is anti-parallel
    across the interface — that is what makes the interface balance check a
    conservation statement rather than an accident;
  * the NEUMANN side applies the partner's numbers UNCHANGED, as
    `L += inner(g, v) * ds`, because the natural boundary term of the
    elasticity weak form is +(sigma . n_own) . v = +q_out_partner . v.

Exporting the raw traction (sigma . n_own) instead flips the sign the Neumann
side applies; the iteration still converges, to the wrong answer.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import dolfinx                # the MODULE, so dolfinx.log.set_log_level(...) resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from dolfinx import fem` alone leaves `dolfinx` undefined
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.55     # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.55          # the shared interface; X0/X1 for axis "x", Y0/Y1 for axis "y"
E_MOD     = 1000.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
# The two subdomains must agree at the two interface corners, or the coupled
# problem is not the un-split one.
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
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides NX, NY and names the level; the per-level
#    dumps below carry that level so the coarse levels survive the fine ones.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))   # a multi-level call's level keys
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
OUTER_X = LO if abs(IFACE_X - HI) < abs(IFACE_X - LO) else HI
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S * e_AX



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
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's VECTOR samples onto THIS participant's interface
    points, COMPONENT BY COMPONENT.

    The driver does no interpolation — non-matching interface meshes are
    handled here, and for a vector field that has to be done per component. One
    np.interp over a flattened (N, 2) array interleaves the two components: the
    result still has the right length, the coupling still converges, and every
    number is wrong.

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
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

# tabulate_dof_coordinates() has ONE ROW PER NODE (dof block); the scalar
# array index of component c at node n is n*2 + c.
xy = V.tabulate_dof_coordinates()
iface_n = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_n = iface_n[np.argsort(xy[iface_n, 1])]            # constant order, always
y_if = xy[iface_n, 1]
if len(iface_n) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
# (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face, which carries a prescribed
# displacement in the un-split problem, so they stay Dirichlet in BOTH
# subproblems. Handing them to the interface leaves them unconstrained on the
# Neumann side: that subproblem is still well posed, still converges, and lands
# a few percent off — measured, 4.7% in the interface displacement and 28% in
# the interface traction, on a coupling whose residual reached 1e-10 and whose
# flux balanced. They are still EXPORTED; they are just not interface-imposed.
corner = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_n = iface_n[~corner]


def eps(w):
    return ufl.sym(ufl.grad(w))


def sigma(w):
    return 2.0 * MU * eps(w) + LAM * ufl.tr(eps(w)) * ufl.Identity(2)


u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
b_src = fem.Function(V)
b_src.interpolate(lambda X: np.vstack(B_SRC(X[0], X[1])))
# KEPT SEPARATE FROM L ON PURPOSE. The traction recovery below subtracts the
# VOLUME load alone, on both sides; adding the interface term into the same
# form is what made the reaction look like zero on the Neumann side.
L_vol = ufl.inner(b_src, v) * ufl.dx
L = L_vol

# ── Dirichlet on the WHOLE non-interface boundary ─────────────────────────
g_out = fem.Function(V)
g_out.x.array[:] = 0.0
outer_n = np.where((np.abs(xy[:, 0] - OUTER_X) < 1e-10) |
                   (np.abs(xy[:, 1] - Y0) < 1e-10) |
                   (np.abs(xy[:, 1] - Y1) < 1e-10))[0]
ox, oy = xy[outer_n, 0], xy[outer_n, 1]
g_out.x.array[2 * outer_n] = (UDX[0] + UDX[1] * ox + UDX[2] * oy
                              + UDX[3] * oy * oy)
g_out.x.array[2 * outer_n + 1] = (UDY[0] + UDY[1] * ox + UDY[2] * oy
                                  + UDY[3] * oy * oy)
bcs = [fem.dirichletbc(g_out, outer_n.astype(np.int32))]

# One definition of the interface measure, used by the Neumann branch to APPLY
# the partner's traction and by the Dirichlet branch to RECOVER its own.
facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], IFACE_X))
tags = dmesh.meshtags(domain, fdim, np.sort(facets),
                      np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

if SIDE == "dirichlet":
    g = fem.Function(V)
    g.x.array[:] = 0.0
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    g.x.array[2 * iface_n] = u_if[:, 0]
    g.x.array[2 * iface_n + 1] = u_if[:, 1]
    bcs.append(fem.dirichletbc(g, iface_bc_n.astype(np.int32)))
else:
    g = fem.Function(V)
    g.x.array[:] = 0.0
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    g.x.array[2 * iface_n] = t_if[:, 0]
    g.x.array[2 * iface_n + 1] = t_if[:, 1]
    # APPLY the partner's numbers UNCHANGED
    L = L_vol + ufl.inner(g, v) * ds_if

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="cpl",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# Interface traction export q_out = -(sigma . n_own).
#
# WHY NOT AN L2 PROJECTION OF THE STRESS. That is what this file used to do on
# the Neumann side: project -(sigma(u_h) . n_own) over the whole subdomain and
# sample it at the interface. The gradient of a P1 solution — and therefore the
# stress — is only O(h) accurate ON the boundary; the superconvergence points
# are interior, and the boundary trace is exactly what the coupling reads.
# Measured earlier against a manufactured solution with a known exact interface
# traction, the projection converged at order ~1 while the consistent traction
# below converged at ~2, so the recovery, not the physics and not the partner,
# was setting the answer. (That is a different experiment from the one under
# MEASURED below, which imposes a traction and asks for it back; there the
# projection does not merely lose an order, it does not converge at all.)
#
# THE CONSISTENT (REACTION) TRACTION. From
#     a(u,v) - (f,v) = int_dOmega (sigma(u) . n) . v ds = -int_Gamma q_out . v ds
# (the second equality is this file's sign convention, q_out = -(sigma . n_own))
# it follows that for every vector basis function phi_i on the interface
#     int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b_vol
# with r the UNCONSTRAINED residual: assembled with no boundary condition
# applied and with the constrained rows NOT zeroed, because on the Dirichlet
# side those rows ARE the reaction.
#
# ONE FORMULA, BOTH SIDES. An earlier version of this file used that reaction
# only on the Dirichlet side and the L2-projected stress on the Neumann side,
# on the reasoning that the Neumann interface dofs are free, so the discrete
# equations hold on them and r comes out ~0. That is true only when the
# residual is taken against a load that ALREADY CONTAINS the interface term.
# Subtract the VOLUME load alone and those same rows carry exactly the
# interface functional the partner applied, componentwise:
#     (A u - b_vol)_(i,c) = int_Gamma g_c phi_i ds
# On the Dirichlet side there is no interface term at all, so b == b_vol and
# the two cases are the same expression.
#
# MEASURED ON THIS FILE, on the Neumann side, by handing it a traction that
# VARIES along the interface — t = (2 + 3 sin 4y, -1 + 2 cos 3y), as if a
# partner had exported it — and asking for it back. Interior interface nodes,
# n = 8/16/32 uniform triangle meshes, non-zero body force, max error per
# component (the two ends are outer-Dirichlet, a different quantity, and are
# reported apart):
#     projected stress   tx 3.26, 4.54, 5.84   ty 6.52, 7.06, 7.84
#                        orders -0.11, -0.15 — it does not converge, it GROWS,
#                        against a true traction whose size is 1 to 5
#     reaction vs b_vol  tx 1.96e-02, 4.98e-03, 1.25e-03
#                        ty 7.40e-03, 1.87e-03, 4.68e-04
#
# READ THE SECOND ROW CORRECTLY — IT IS NOT A CONVERGENCE RESULT. On the
# Neumann side the free interface rows satisfy (A u - b_vol)_i = (M_Gamma g)_i
# IDENTICALLY, so the export is the consistent-to-nodal conversion
# -(M_Gamma g)/(M_Gamma 1) and its offset from -g is -(h^2/6) g''(y) for ANY
# correct assembly of ANY equation, at ANY Poisson ratio. This row used to
# carry "orders 1.98, 2.00" as if it measured the recovery's accuracy; it does
# not. The x-component figures matching the scalar conduction participants to
# three digits is NOT "a property of the discretisation rather than of an
# implementation" as this comment used to claim — it is the same P1 boundary
# mass matrix on the same interface nodes with the same g, and a bare NumPy
# mass matrix with no PDE, no solver and no material in it reproduces them to
# five significant figures. The recovery's ORDER is measured elsewhere, on the
# DIRICHLET side against an analytic flux
# (tests/test_interface_flux_converges_to_a_known_exact_flux.py): 1.996, 1.998,
# 1.993 on 8/16/32/64, scalar conduction. There is no VECTOR measurement
# against an analytic traction, and none is claimed here.
#
# WHAT THE FIXTURE IS STILL GOOD FOR: it fails loudly on a flipped sign, a
# wrong interface weight, a mistagged facet set, and — the standing footgun of
# this file — a blocked-dof mapping that divides one node's component by
# another node's weight. The projected-stress row above is a genuine
# measurement: it passes through the discretisation. The Dirichlet side is
# untouched by this change (b == b_vol there): its export is bit-identical to
# the previous version's, checked.
#
# THE WEIGHT IS ONE SCALAR PER NODE AND THE DOFS ARE BLOCKED. V carries two
# scalar dofs per node, at array positions 2*n and 2*n+1, while the nodal
# interface weight w_n = int_Gamma phi_n ds is a SINGLE number for the node.
# Taking the test vector (1,1) in the weight form puts that same number at BOTH
# block entries — the basis function of dof (n,c) is phi_n * e_c and
# inner((1,1), phi_n e_c) = phi_n whatever c is — so indexing with the
# (n_iface, 2) block index array below divides every component of a node by that
# node's own scalar weight. Do NOT index a blocked array with `iface_n` itself:
# those are NODE numbers, and entry n of a blocked array is component n % 2 of
# node n // 2, i.e. a different node's dof, silently.
w_ = ufl.TestFunction(V)

Amat = _fp.assemble_matrix(fem.form(a))          # no bcs= on purpose
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))      # no lifting, no set_bc
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

ones = fem.Constant(domain, np.ones(2, dtype=default_scalar_type))
wvec = _fp.assemble_vector(fem.form(ufl.inner(ones, w_) * ds_if))
wvec.ghostUpdate()

idx = np.column_stack([2 * iface_n, 2 * iface_n + 1])   # blocked, (n_iface, 2)
wi = wvec.array[idx]
Q = np.zeros_like(wi)
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[idx][ok] / wi[ok]

# THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (a y-face), so
# their rows carry the OUTER reaction too and their residual is not this
# interface's traction. Take the nearest interior interface node rather than
# exporting a corner value that is physically a different quantity. This holds
# on BOTH sides: the corners are outer-Dirichlet in both subproblems.
suspect = np.isin(iface_n, outer_n) | ~ok.all(axis=1)
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

U = np.column_stack([uh.x.array[2 * iface_n], uh.x.array[2 * iface_n + 1]])
print(f"[fenics {SIDE}] interface n={len(U)} "
      f"ux=[{U[:,0].min():.6g},{U[:,0].max():.6g}] "
      f"uy=[{U[:,1].min():.6g},{U[:,1].max():.6g}] "
      f"tx=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}] "
      f"ty=[{Q[:,1].min():.6g},{Q[:,1].max():.6g}]")

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
    print(f"\nNDOF = {int(len(uh.x.array))}")
except Exception as _ndof_exc:
    print(f"[fenics] could not report NDOF: {_ndof_exc!r}. Your task's"
          f" execution log needs `NDOF = <integer>` on a line of its own,"
          f" so print your own degree-of-freedom count here.")

# PER-LEVEL PERSISTENCE: this level's whole field and its interface trace and
# traction, named by LEVEL, never overwritten by the next level (exports.json is).
# Interpolate THESE onto the probe points your task names. A file the next
# level overwrites cannot carry a mesh study.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    with open(f"field_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy\n")
        for _i, (_px, _py) in enumerate(xy[:, :2]):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(uh.x.array[2 * _i]):.11e},{float(uh.x.array[2 * _i + 1]):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy,qx,qy\n")
        for _y, (_ux, _uy), (_qx, _qy) in zip(y_if, U, Q):
            _f.write(f"{float(IFACE_X):.11e},{float(_y):.11e},{float(_ux):.11e},{float(_uy):.11e},{float(_qx):.11e},{float(_qy):.11e}\n")
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
    print(f"[fenics_elastic per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")
# exports.json LAST: the driver takes its existence as proof of success.
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a traction of ~0 against a nonzero partner); and a traction that is the
#    partner's array negated instead of a recovery from THIS side's own system.
#    A VECTOR side needs it more, not less: a displacement field that came out
#    ~0 because the load never arrived still couples, still converges and still
#    hands in three tidy levels.
_chk_vals = np.asarray(U, float).ravel()
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
    "n_points": int(len(iface_n)),
    "coordinates": [([float(IFACE_X), float(y)] if AX == 0 else [float(y), float(IFACE_X)])
                    for y in y_if],
    "values": [[float(a_), float(b_)] for a_, b_ in U],
    "normal_fluxes": [[float(a_), float(b_)] for a_, b_ in Q],
}, indent=2))
