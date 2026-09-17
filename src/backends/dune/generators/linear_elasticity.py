"""DUNE-fem linear elasticity generator and knowledge.

HISTORY. Until 2026-08-03 the generator registered under
``linear_elasticity_2d`` was one line — ``return _poisson_2d(params)``
— so asking openPASO for DUNE linear elasticity produced a byte-identical
Poisson script that printed "DUNE-fem Poisson solve complete." under an
elasticity name. Nothing in the output said so. It is replaced below by
a template that was executed against dune-fem 2.12.0.2 and checked
against the plane-strain uniaxial-tension solution.
"""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity, plane strain, uniaxial tension — DUNE-fem.
    """
    nx = params.get("nx", 32)
    order = params.get("order", 1)
    E = params.get("E", 210e9)
    nu = params.get("nu", 0.3)
    traction = params.get("traction", 1.0e6)
    return f'''\
"""Linear elasticity (PLANE STRAIN) on [0,1]^2 — DUNE-fem.

Uniaxial tension: traction t in +x on the right edge x=1,
roller symmetry on x=0 (u_x=0) and y=0 (u_y=0), rest traction free.

Closed-form answer this script checks itself against:
    eps_xx = (1 - nu^2) / E * t        (extension along the pull)
    eps_yy = -nu * (1 + nu) / E * t    (lateral contraction, NEGATIVE)
so u_x(x=1) = eps_xx and u_y(y=1) = eps_yy on the unit square.
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, Identity,
                 grad, inner, sym, tr, dx, ds, conditional, lt)
import numpy as np
import json

# --- material (change these) ------------------------------------------
E_mod = {E}          # Young's modulus [Pa]
nu_val = {nu}        # Poisson ratio [-]
t_val = {traction}   # traction on x=1, +x direction [Pa]

# Lame parameters. mu is the SHEAR modulus, lam is the first Lame
# parameter; swapping them still runs and still converges.
mu_val = E_mod / (2.0 * (1.0 + nu_val))
lam_val = E_mod * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])

# REQUIRED: dimRange=2 makes this a VECTOR space. Without dimRange the
# space is scalar and the elasticity form fails in UFL.
space = lagrange(gridView, dimRange=2, order={order})

u = TrialFunction(space)
v = TestFunction(space)
x = SpatialCoordinate(space)
I = Identity(2)


def eps(w):
    """Small-strain tensor. sym() is REQUIRED: grad(w) alone couples
    rigid-body rotation into the stress."""
    return sym(grad(w))


def sigma(w):
    """Plane-strain Cauchy stress."""
    return lam_val * tr(eps(w)) * I + 2.0 * mu_val * eps(w)


a = inner(sigma(u), eps(v)) * dx

# Neumann traction: a boundary integral over ds, masked by a conditional
# on the coordinate. There is no facet-tag mechanism in this API.
tol = 1e-8
L = conditional(lt(1.0 - x[0], tol), t_val * v[0], 0.0) * ds

# Roller (symmetry) BCs. A None entry leaves that component FREE.
bc_left = DirichletBC(space, [0, None], conditional(lt(x[0], tol), 1, 0))
bc_bot = DirichletBC(space, [None, 0], conditional(lt(x[1], tol), 1, 0))

# REQUIRED: the BCs must be ELEMENTS OF THE LIST passed to galerkin.
# galerkin(a == L, space=space) with the BC objects built but not listed
# runs, reports converged, and returns garbage.
scheme = galerkin([a == L, bc_left, bc_bot], solver="cg",
                  parameters={{"linear.tolerance": 1e-12,
                               "linear.maxiterations": 50000}})

uh = space.interpolate([0, 0], name="displacement")
info = scheme.solve(target=uh)

# as_numpy on a dimRange=2 space is INTERLEAVED [u0x,u0y,u1x,u1y,...]
vals = np.array(uh.as_numpy).reshape(-1, 2)
ux, uy = vals[:, 0], vals[:, 1]

eps_xx = (1.0 - nu_val ** 2) / E_mod * t_val
eps_yy = -nu_val * (1.0 + nu_val) / E_mod * t_val

print(f"converged={{info['converged']}} "
      f"linear_iterations={{info['linear_iterations']}}")
print(f"DOFs: {{vals.size}}")
print(f"u_x(x=1) = {{ux.max():.6e}}  analytic {{eps_xx:.6e}}  "
      f"ratio {{ux.max() / eps_xx:.4f}}")
print(f"u_y(y=1) = {{uy.min():.6e}}  analytic {{eps_yy:.6e}}  "
      f"ratio {{uy.min() / eps_yy:.4f}}")
print(f"lateral contraction sign: min(u_y) = {{uy.min():.3e}} "
      f"(MUST be < 0 for tension)")

gridView.writeVTK("result", pointdata={{"displacement": uh}})
summary = {{
    "max_ux": float(ux.max()),
    "min_uy": float(uy.min()),
    "analytic_eps_xx": eps_xx,
    "analytic_eps_yy": eps_yy,
    "ratio_ux": float(ux.max() / eps_xx),
    "n_dofs": int(vals.size),
    "E": E_mod, "nu": nu_val, "traction": t_val,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DUNE-fem linear elasticity solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": (
            "Linear elasticity (plane strain / plane stress / 3D) on a "
            "VECTOR Lagrange space, assembled with the same galerkin "
            "scheme as Poisson."),

        # ── what to type, before any explanation ─────────────────────
        "required_calls_in_order": [
            "gridView = dune.grid.structuredGrid([0,0],[1,1],[n,n])",
            "space = dune.fem.space.lagrange(gridView, dimRange=2, order=k)"
            "   <- dimRange IS the whole difference from Poisson",
            "u, v = ufl.TrialFunction(space), ufl.TestFunction(space)",
            "a = ufl.inner(sigma(u), eps(v))*ufl.dx  with "
            "eps(w)=ufl.sym(ufl.grad(w)) and "
            "sigma(w)=lam*ufl.tr(eps(w))*ufl.Identity(2)+2*mu*eps(w)",
            "bc = dune.ufl.DirichletBC(space, [0, 0], <indicator>)",
            "scheme = dune.fem.scheme.galerkin([a == L, bc], solver='cg')",
            "uh = space.interpolate([0, 0], name='displacement')",
            "scheme.solve(target=uh)",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "dimRange=2 (2D) or dimRange=3 (3D) on the lagrange call",
                "an initial guess with the right shape: "
                "space.interpolate([0, 0], ...) — a scalar 0 raises",
                "every DirichletBC listed INSIDE the galerkin list",
                "sym() around grad(u) in the strain",
            ],
            "OPTIONAL": [
                "parameters={'linear.tolerance': ..., "
                "'linear.maxiterations': ...} — defaults work for "
                "small problems",
                "solver='cg' — elasticity is SPD so cg is the right "
                "default; gmres and bicgstab also solve it",
                "gridView.writeVTK(...) — only needed for pictures",
            ],
        },
        "verification_you_can_run": (
            "Uniaxial tension of a unit square, plane strain: pull the "
            "right edge with traction t, roller on x=0 and y=0. The "
            "answer is u_x(x=1) = (1-nu^2)/E * t and "
            "u_y(y=1) = -nu(1+nu)/E * t, and min(u_y) MUST be "
            "negative. Executed 2026-08-03 with E=210e9, nu=0.3, "
            "t=1e6 on structuredGrid: measured/analytic came out 1.025 "
            "at 8x8, 1.006 at 16x16 and 1.002 at 32x32 (Q1), and 1.001 "
            "at 16x16 with Q2.\n"
            "READ THE NEXT SENTENCE BEFORE INTERPRETING THOSE "
            "NUMBERS. The exact solution here is LINEAR, so it lies "
            "inside the P1 space and a correct solve would return it "
            "to solver tolerance, not to 0.2%. Every digit of the "
            "residual disagreement is the shared-corner constraint "
            "defect described in the pitfalls below, NOT "
            "discretisation error — the same problem with the exact "
            "field imposed on the WHOLE boundary instead of "
            "component-wise rollers returned relative L2 8.7e-16 at "
            "4x4 and 7.8e-14 at 32x32, i.e. exact at every "
            "resolution, and on aluConformGrid the roller version is "
            "itself exact at some resolutions (4.2e-15 at 8x8 and "
            "7.1e-14 at 32x32) and wrong at others (1.2e-01 at 4x4, "
            "8.1e-03 at 16x16). So: use the sign of u_y and the "
            "order of magnitude of the ratio to catch a wrong "
            "constitutive law, and use the whole-boundary patch test "
            "below when you need an exact check. (Verified by "
            "adversarial re-execution 2026-08-03; an earlier version "
            "of this entry read the shrinking ratio as convergence, "
            "which was wrong.)"),
        "exactness_check_that_costs_nothing": (
            "PATCH TEST. Interpolate the linear field "
            "uex = as_vector([exx*x[0], eyy*x[1]]) and impose it as "
            "DirichletBC(space, uex) on the WHOLE boundary with a zero "
            "body force. A conforming Lagrange space of any order "
            "contains that field exactly, so the discrete solution must "
            "reproduce it to solver tolerance. Executed 2026-08-03 on "
            "Q1, 8x8 and 16x16 structuredGrid: max nodal difference "
            "9.8e-16 and 3.6e-15 RELATIVE to the field amplitude. If "
            "your patch test does not come back at ~1e-15 the operator "
            "or the BC wiring is broken, and no amount of mesh "
            "refinement will fix it."),

        "pitfalls": [
            (
                "[API] lagrange(gridView, order=k, dimRange=2) is what "
                "makes the space vector-valued; dimRange is NOT "
                "optional for elasticity. Signal: with a scalar space "
                "the elasticity form dies in UFL at the first "
                "sym(grad(u)) product, and space.interpolate([0, 0]) "
                "on a scalar space raises before that. The scalar "
                "space is the default: lagrange(gridView, order=1) "
                "gives dimRange=1. (Executed 2026-08-03 on dune-fem "
                "2.12.0.2.)"
            ),
            (
                "[Numerical] Two component-wise DirichletBCs whose "
                "edges MEET lose one constraint at the shared corner "
                "dof, silently, and the rule is GEOMETRIC: the "
                "constraint on the VERTICAL edge is the one dropped, "
                "whichever component it names and whatever order the "
                "BCs are listed in. Signal: with "
                "DirichletBC(space,[0,None],x[0]<tol) and "
                "DirichletBC(space,[None,0],x[1]<tol) on a 4x4 "
                "structuredGrid, the dof at (0,0) came back "
                "u_x = +1.909986e-06 instead of 0 while u_y was "
                "exactly 0 — 40% of the field magnitude, at exactly "
                "one dof, with converged=True. Swapping the two BCs "
                "in the list changed nothing. Swapping which "
                "COMPONENT each edge names flipped the victim, which "
                "is how the vertical-edge rule was established. The "
                "mirrored problem (u_x=0 on x=1, u_y=0 on y=1) fails "
                "identically at (1,1). On aluConformGrid it is "
                "MESH-DEPENDENT: broken at 4x4 and 16x16, exactly "
                "right at 8x8 and 32x32 — which is worse, because a "
                "single passing resolution proves nothing. No "
                "workaround was found: a third BC naming both "
                "components at the corner does not fix it, nor does "
                "re-listing the losing BC. Detect it by reading the "
                "constrained dof back and comparing it with the value "
                "you asked for; validate the operator itself with the "
                "whole-boundary patch test instead. (Executed "
                "2026-08-03 on dune-fem 2.12.0.2; the "
                "vertical-edge rule and the ALUGrid intermittency "
                "come from adversarial re-execution the same day.)"
            ),
            (
                "[API] A DirichletBC indicator is evaluated PER "
                "BOUNDARY FACET, not per dof, so an indicator that is "
                "true only at a point selects nothing. Signal: adding "
                "DirichletBC(space,[0,0], And(x[0]<tol, x[1]<tol)) to "
                "pin the single corner node changed the solution by "
                "exactly zero — the corner dof kept the value it had "
                "without that BC (1.909986e-06 at 4x4, unchanged to "
                "all printed digits), because no FACET centre "
                "satisfies both coordinate tests. Use edge-sized "
                "indicators, never a point. This is also why the "
                "shared-corner defect above has no BC-level "
                "workaround. (Executed 2026-08-03 on dune-fem "
                "2.12.0.2, confirmed by adversarial re-execution.)"
            ),
            (
                "[Numerical] Lame parameters from E and nu: "
                "mu = E/(2*(1+nu)), lam = E*nu/((1+nu)*(1-2*nu)). "
                "Signal: swapping the two formulae still assembles, "
                "still converges and still prints a plausible "
                "displacement — the detector is the SIGN and SIZE of "
                "the lateral contraction, which the uniaxial check "
                "above pins to -nu(1+nu)/E*t. A positive min(u_y) "
                "under tension means the constitutive law is wrong, "
                "not the solver. (Audit 2026-06-02; the sign detector "
                "added after executing the corrected template "
                "2026-08-03.)"
            ),
            (
                "[Numerical] Strain must be sym(grad(u)), stress "
                "lam*tr(eps)*I + 2*mu*eps. Signal: writing eps as a "
                "bare grad(u) couples rotation into stress — a rigid "
                "body rotation then produces non-zero stress and a "
                "cantilever stiffens without bound as it rotates. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[API] as_numpy on a dimRange=2 space is a single flat "
                "INTERLEAVED array [u0_x, u0_y, u1_x, u1_y, ...], not "
                "two blocks. Signal: reading it as "
                "vals[:len(vals)//2] gives you the x-displacement of "
                "the first half of the nodes and calls it 'u_x'. Use "
                "np.array(uh.as_numpy).reshape(-1, 2) and take "
                "columns. (Executed 2026-08-03: 578 entries for a "
                "16x16 Q1 vector space = 289 nodes x 2.)"
            ),
            (
                "[API] Plane STRAIN and plane STRESS differ only in "
                "lam, and DUNE-fem has no switch for it. Signal: the "
                "form above is plane strain; if you meant plane stress "
                "you must replace lam by lam_ps = 2*mu*lam/(lam+2*mu) "
                "= E*nu/(1-nu^2) yourself, and the check value becomes "
                "eps_xx = t/E, eps_yy = -nu*t/E. Mind the DIRECTION of "
                "the mismatch, and note that this catalog's 'ratio' is "
                "always measured/analytic, the same convention the "
                "template prints: judging a plane-STRAIN run against "
                "the plane-STRESS formula gives (1-nu^2) ~ 0.91 at "
                "nu=0.3, i.e. the run looks 9% too SOFT; it is the "
                "reverse pairing, a plane-stress run judged against the "
                "plane-strain formula, that gives 1/(1-nu^2) ~ 1.10. "
                "Either way it looks like a 10% mesh error. Measured by "
                "adversarial audit 2026-08-03: the shipped plane-strain "
                "template's u_x(1) = 4.340198e-06 against the "
                "plane-stress value t/E = 4.761905e-06 is a ratio of "
                "0.9114, not 1.0989 — an earlier revision of this entry "
                "stated the inverse for this direction. (Audit "
                "2026-08-03.)"
            ),
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
}
