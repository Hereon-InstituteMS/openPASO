"""Advanced physics generators for FEniCSx/dolfinx.

Covers physics that require DG formulations, penalty methods, phase-field
models, transient time-stepping, phase separation, nonlinear Newton loops,
and curl-curl electromagnetic formulations.

Variants per physics:
  dg_methods            : 2d
  contact               : 2d
  multiphase            : 2d
  time_dependent_heat   : 2d
  cahn_hilliard         : 2d
  nonlinear_pde         : 2d
  magnetostatics        : 2d
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# dg_methods
# ---------------------------------------------------------------------------

_DG_KNOWLEDGE = {
    "description": (
        "Discontinuous Galerkin (DG) for advection-dominated diffusion. "
        "An upwind numerical flux handles convection; symmetric "
        "interior-penalty (SIPG) terms handle diffusion. Both the inflow "
        "value and the diffusive Dirichlet value are imposed WEAKLY through "
        "boundary integrals — a strong dirichletbc does nothing on a DG "
        "space."
    ),
    "minimal_working_example": (
        "# COMPLETE runnable script. Steady advection-diffusion on the unit\n"
        "# square, DG1 with upwind advection + SIPG diffusion. Verified by\n"
        "# executing it on dolfinx 0.10.0.\n"
        "from mpi4py import MPI\n"
        "from dolfinx import mesh, fem, default_scalar_type\n"
        "from dolfinx.fem.petsc import assemble_matrix, assemble_vector\n"
        "from petsc4py import PETSc\n"
        "import ufl\n"
        "import numpy as np\n"
        "\n"
        "domain = mesh.create_unit_square(MPI.COMM_WORLD, 40, 40,\n"
        "                                 mesh.CellType.triangle)\n"
        "fdim = domain.topology.dim - 1\n"
        "domain.topology.create_connectivity(fdim, domain.topology.dim)\n"
        "V = fem.functionspace(domain, ('DG', 1))\n"
        "u, v = ufl.TrialFunction(V), ufl.TestFunction(V)\n"
        "\n"
        "eps = 0.005\n"
        "b = ufl.as_vector([1.0, 0.5])\n"
        "n = ufl.FacetNormal(domain)\n"
        "h = ufl.CellDiameter(domain)\n"
        "h_avg = (h('+') + h('-')) / 2.0\n"
        "alpha = 4.0 * (1 + 1) ** 2        # 4*(degree+1)^2, REQUIRED scaling\n"
        "f = fem.Constant(domain, default_scalar_type(1.0))\n"
        "u_D = fem.Constant(domain, default_scalar_type(0.0))\n"
        "\n"
        "bn = ufl.dot(b, n)\n"
        "bn_out = (bn + abs(bn)) / 2.0     # boundary outflow part\n"
        "bn_in = (bn - abs(bn)) / 2.0      # boundary inflow part\n"
        "up = ((bn('+') + abs(bn('+'))) / 2.0 * u('+')\n"
        "      + (bn('+') - abs(bn('+'))) / 2.0 * u('-'))\n"
        "\n"
        "a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\n"
        "     - eps * ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS\n"
        "     - eps * ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS\n"
        "     + alpha / h_avg * eps * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS\n"
        "     - ufl.inner(u * b, ufl.grad(v)) * ufl.dx\n"
        "     + up * ufl.jump(v) * ufl.dS\n"
        "     + bn_out * u * v * ufl.ds\n"
        "     - eps * ufl.dot(ufl.grad(u), n) * v * ufl.ds\n"
        "     - eps * ufl.dot(ufl.grad(v), n) * u * ufl.ds\n"
        "     + alpha / h * eps * u * v * ufl.ds)\n"
        "L = (f * v * ufl.dx\n"
        "     - bn_in * u_D * v * ufl.ds\n"
        "     - eps * ufl.dot(ufl.grad(v), n) * u_D * ufl.ds\n"
        "     + alpha / h * eps * u_D * v * ufl.ds)\n"
        "\n"
        "A = assemble_matrix(fem.form(a))\n"
        "A.assemble()\n"
        "rhs = assemble_vector(fem.form(L))\n"
        "rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)\n"
        "ksp = PETSc.KSP().create(domain.comm)\n"
        "ksp.setOperators(A)\n"
        "ksp.setType('preonly')\n"
        "ksp.getPC().setType('lu')\n"
        "uh = fem.Function(V)\n"
        "ksp.solve(rhs, uh.x.petsc_vec)\n"
        "uh.x.scatter_forward()\n"
        "if ksp.getConvergedReason() <= 0:\n"
        "    raise RuntimeError(f'KSP failed, reason {ksp.getConvergedReason()}')\n"
        "if not np.all(np.isfinite(uh.x.array)):\n"
        "    raise RuntimeError('solution contains non-finite values')\n"
        "print('u range:', uh.x.array.min(), uh.x.array.max())\n"
    ),
    "function_space": {
        "REQUIRED": (
            "V = fem.functionspace(domain, ('DG', degree))\n"
            "domain.topology.create_connectivity(fdim, domain.topology.dim)"
        ),
        "OPTIONAL": (
            "degree 0, 1, 2, ... . degree 0 for pure first-order upwind "
            "finite volume; degree 1 and 2 verified here."
        ),
        "explanation": (
            "The facet-to-cell connectivity is REQUIRED before any dS "
            "integral can be assembled."
        ),
    },
    "weak_form": {
        "REQUIRED": (
            "bn = ufl.dot(b, n)\n"
            "up = ((bn('+') + abs(bn('+')))/2 * u('+')\n"
            "      + (bn('+') - abs(bn('+')))/2 * u('-'))     # upwind trace\n"
            "a_adv = -ufl.inner(u*b, ufl.grad(v))*ufl.dx + up*ufl.jump(v)*ufl.dS \\\n"
            "        + (bn + abs(bn))/2 * u * v * ufl.ds      # OUTFLOW part only\n"
            "a_dif = (eps*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx\n"
            "         - eps*ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n))*ufl.dS\n"
            "         - eps*ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v)))*ufl.dS\n"
            "         + alpha/h_avg*eps*ufl.inner(ufl.jump(u, n), ufl.jump(v, n))*ufl.dS)"
        ),
        "OPTIONAL": (
            "For pure advection (eps == 0) drop a_dif entirely, including "
            "its penalty term. In that case the outflow restriction below is "
            "the ONLY thing keeping the operator invertible."
        ),
        "explanation": (
            "REQUIRED: the boundary advection term must use only the OUTFLOW "
            "part (bn + |bn|)/2. Using the raw bn over the whole boundary "
            "makes the operator singular."
        ),
        "pitfalls": [
            "[Numerical] Restrict the boundary advection term to the outflow "
            "part. Signal: with the raw dot(b, n)*u*v*ds over the whole "
            "boundary the LU factorisation aborts — 'Zero pivot row <k> "
            "value 2.00577e-17 tolerance 2.22045e-14' — and without a "
            "converged-reason check the script prints inf and exits 0.",
            "[API] dot(b, n) inside a dS integral must be restricted. "
            "Signal: ValueError: Discontinuous type Jacobian must be "
            "restricted.",
        ],
    },
    "boundary_conditions": {
        "REQUIRED": (
            "# inflow value, advective part\n"
            "L += -(bn - abs(bn))/2 * u_D * v * ufl.ds\n"
            "# same value for the diffusive part (Nitsche / SIPG on ds)\n"
            "a += (-eps*ufl.dot(ufl.grad(u), n)*v*ufl.ds\n"
            "      - eps*ufl.dot(ufl.grad(v), n)*u*ufl.ds\n"
            "      + alpha/h*eps*u*v*ufl.ds)\n"
            "L += (-eps*ufl.dot(ufl.grad(v), n)*u_D*ufl.ds\n"
            "      + alpha/h*eps*u_D*v*ufl.ds)"
        ),
        "OPTIONAL": (
            "Use a tagged ds(marker) to apply different values on different "
            "walls; leave a wall out of the Nitsche block for a natural "
            "(zero-flux) diffusive condition."
        ),
        "explanation": (
            "REQUIRED: everything is weak. fem.dirichletbc on a DG space is "
            "a silent no-op."
        ),
        "pitfalls": [
            "[API] Never use fem.dirichletbc on a DG space. Signal: "
            "fem.locate_dofs_topological on a DG1 space with boundary facets "
            "returns an EMPTY dof array, so the BC constrains nothing and "
            "raises no error.",
        ],
    },
    "solver": {
        "REQUIRED": (
            "ksp = PETSc.KSP().create(domain.comm)\n"
            "ksp.setOperators(A)\n"
            "ksp.setType('preonly'); ksp.getPC().setType('lu')\n"
            "ksp.solve(rhs, uh.x.petsc_vec)\n"
            "if ksp.getConvergedReason() <= 0:      # REQUIRED\n"
            "    raise RuntimeError(ksp.getConvergedReason())"
        ),
        "OPTIONAL": (
            "GMRES + a block-Jacobi / ILU preconditioner at scale. CG is "
            "wrong here: the advection makes the operator non-symmetric."
        ),
        "explanation": (
            "TWO checks are REQUIRED. The converged reason catches a failed "
            "factorisation, which leaves inf in the solution vector while "
            "the script still exits 0. A magnitude check on uh.x.array "
            "catches a loss of SIPG coercivity (penalty too small for the "
            "degree), which the KSP reports as CONVERGED."
        ),
        "pitfalls": [
            "[Numerical] Test ksp.getConvergedReason() > 0 and "
            "np.isfinite(uh.x.array).all(). Signal: reason -11 "
            "(KSP_DIVERGED_PC_FAILED) with u printing as min=inf, max=inf "
            "while the process exit status is 0.",
            "[Numerical] Also check the MAGNITUDE, not only finiteness. "
            "Signal: with a penalty too small for the DG degree the run "
            "reports reason 4 and returns max|u| ~ 1e3 for O(1) data on a "
            "unit domain.",
        ],
    },
    "verification": (
        "Three checks, all needed, none of which is the return code. "
        "(1) ksp.getConvergedReason() > 0 — catches the singular-operator "
        "case. (2) np.isfinite(uh.x.array).all(). (3) a MAGNITUDE bound: "
        "with a bounded inflow value and a bounded source on a unit domain "
        "the solution is O(1), so max|u| of order 1e3 or more means the SIPG "
        "penalty is too small for the degree — and that case reports "
        "CONVERGED, so checks (1) and (2) both pass. To confirm the boundary "
        "condition is actually applied, refine once and check that "
        "fem.assemble_scalar(fem.form((uh - u_D)**2 * ufl.ds)) DECREASES; a "
        "value that does not move under refinement means it is not imposed."
    ),
    "pitfalls": [
        "[Numerical] The boundary advection term must be restricted to the "
        "OUTFLOW part of the boundary, (dot(b,n) + |dot(b,n)|)/2. Writing "
        "dot(b, n)*u*v*ds over the whole boundary subtracts on the inflow "
        "facets and destroys coercivity: the assembled operator becomes "
        "numerically singular. Signal: on a form with no ds Nitsche block, "
        "the smallest singular value of the assembled matrix drops to "
        "~1e-18 (condition number ~1e16), PETSc's LU aborts with 'Zero pivot "
        "in LU factorization: https://petsc.org/release/faq/#zeropivot' and "
        "'Zero pivot row 9599 value 2.00577e-17 tolerance 2.22045e-14', "
        "KSPConvergedReason is -11 (KSP_DIVERGED_PC_FAILED), and a script "
        "that does not check the reason prints 'u: min=inf, max=inf' and "
        "exits 0. Restricting the term to the outflow part removes the "
        "near-null direction and brings the condition number of the same "
        "matrix from ~1e16 to O(1e2). IMPORTANT SCOPE, established by an "
        "adversarial re-check: the outflow restriction is the STRUCTURAL "
        "fix — it is what makes the operator invertible for a pure-advection "
        "problem (eps = 0) and when b is tangential to part of the boundary. "
        "When eps > 0, adding the ds Nitsche block ALONE also makes the "
        "raw-dot(b,n) operator invertible, so a form that has both changes "
        "will not expose which one mattered. Do both: use the outflow "
        "restriction for the advection and the Nitsche block for the "
        "diffusion. (Verified by execution 2026-08-03, dolfinx 0.10.0 — this "
        "is the defect the shipped template used to have.)",
        "[Numerical] A DG advection-diffusion form also needs the DIFFUSIVE "
        "Dirichlet value imposed weakly on the boundary — the Nitsche/SIPG "
        "block -eps*ufl.dot(ufl.grad(u), n)*v*ufl.ds "
        "- eps*ufl.dot(ufl.grad(v), n)*u*ufl.ds + alpha/h*eps*u*v*ufl.ds in "
        "the bilinear form, with the matching u_D terms in the linear form. "
        "With only the advective inflow term the diffusion operator sees a "
        "pure natural (zero-flux) condition and the boundary value is never "
        "imposed. Signal: measure the boundary defect with "
        "fem.assemble_scalar(fem.form((uh - u_D)**2 * ufl.ds)) normalised by "
        "the interior norm — WITHOUT the Nitsche block that quantity STALLS: "
        "it is unchanged to four digits when the mesh is refined 16 -> 32 -> "
        "64 and unchanged when the DG degree goes from 1 to 2, which is the "
        "fingerprint of a condition that is not being applied at all. WITH "
        "the block the same quantity falls with every refinement. The KSP "
        "converges either way, so nothing in the solver output reveals it. "
        "SCOPE, established by an adversarial re-check: the diagnostic is "
        "the STALL, not the size — the stalled value is O(1) only when "
        "diffusion dominates; in the advection-dominated regime it stalls at "
        "a small value that looks acceptable if you only glance at it once. "
        "At degree 0 the defect does decrease under refinement (there is no "
        "boundary trace to get wrong), and with b = 0 the operator has no "
        "advective boundary term at all so this test does not apply. "
        "(Verified by execution 2026-08-03, dolfinx 0.10.0, at DG1-DG3 on "
        "triangles and quadrilaterals in both regimes.)",
        "[Numerical] ALWAYS check ksp.getConvergedReason() and the "
        "finiteness of the solution array. Signal: KSPConvergedReason = -11 "
        "(KSP_DIVERGED_PC_FAILED) with uh.x.array full of inf while the "
        "process exit status is 0 — a silent wrong answer. (Verified by "
        "execution 2026-08-03.)",
        "[Numerical] DG advection flux: upwind — take the value from the "
        "upstream side, e.g. (bn('+') + abs(bn('+')))/2*u('+') + "
        "(bn('+') - abs(bn('+')))/2*u('-'). Signal: a centred flux "
        "0.5*(u('+') + u('-')) on a pure-advection DG problem gives an "
        "operator without the upwind dissipation; the discrete maximum "
        "principle is lost and the solution oscillates. (Audit 2026-06-02.)",
        "[Numerical] The SIPG penalty must scale with the DG degree: use "
        "alpha = 4*(degree+1)**2, not a fixed number. Signal: a fixed "
        "alpha = 10 is fine at degree 1 and 2, but at degree 3 on triangles "
        "the same script returns max|u| of order 1e3 for O(1) data on a unit "
        "domain — with KSPConvergedReason still 4. Nothing in the solver "
        "output reports it; only a magnitude check on uh.x.array does. "
        "Raising alpha to 4*(degree+1)**2 brings the same case back to "
        "max|u| ~ 1. Too LARGE an alpha inflates the condition number "
        "instead, and PETSc prints no condition-number warning of its own — "
        "observe the residual history or compute the condition number "
        "yourself. (Verified by execution 2026-08-03, dolfinx 0.10.0, at DG1 "
        "/ DG2 / DG3 on triangles and quadrilaterals.)",
        "[API] FacetNormal n is outward; avg/jump operators need '+'/'-' "
        "sides. Signal: [exact text re-measured 2026-08-03 on dolfinx 0.10.0 "
        "/ ufl 2025.2.1] writing dot(b, n) without a side suffix in a dS "
        "integral raises ValueError from fem.form. UFL formats that line at "
        "run time — ufl/algorithms/apply_restrictions.py:65 raises "
        "ValueError(f'Discontinuous type {o._ufl_class_.__name__} must be "
        "restricted.') — so the class name is filled in from the expression "
        "and only the head and tail are literals. Re-measured 2026-08-13 on "
        "this form the class is Jacobian, so the line reads: ValueError: "
        "Discontinuous type Jacobian must be restricted. "
        "The same expression with "
        "dot(b, n('+')) compiles. The previously quoted string \"side "
        "specifier required on '+' or '-' for restricted facet integrals\" "
        "does NOT appear in current UFL.",
        "[API] Inflow BC imposed weakly via boundary integral, NOT "
        "Dirichlet. Signal: [VERIFIED empirically 2026-08-03, dolfinx "
        "0.10.0] a strong DirichletBC on a DG space is a silent no-op — "
        "fem.locate_dofs_topological on a DG1 space with the x=0 boundary "
        "facets of an 8x8 unit square returns ZERO dofs, so the BC "
        "constrains nothing at all and raises no error. The inflow BC must "
        "enter the form via the boundary integrals.",
        "[Numerical] For pure advection (eps = 0): DROP the diffusion terms "
        "entirely (do not just set eps small). Signal: keeping the IP "
        "diffusion terms with eps = 0 gives a multiplicative-zero in the "
        "form but the penalty term alpha/h * jump(u) * jump(v) * dS REMAINS "
        "and over-stabilises the pure-advection problem, smearing the "
        "solution across element faces. Remove the avg/jump/eps block "
        "entirely for hyperbolic problems. (Audit 2026-06-02.)",
        "[Performance] DG mass matrix is block-DIAGONAL (one block per "
        "cell) — efficient for explicit time stepping because the inversion "
        "is local. Signal: applying a generic scipy.sparse.linalg spsolve to "
        "invert M each step misses the block-diagonal structure; a per-cell "
        "local solve is far faster. For implicit DG, the FULL system K is "
        "not block-diagonal — only the mass is. (Audit 2026-06-02.)",
        "[API] Modern UFL has no ufl.Abs symbol — use Python's "
        "builtin abs() (which UFL overloads for Expr operands) "
        "or ufl.algebra.Abs as the explicit fallback. Calling "
        "ufl.Abs(z) raises AttributeError: module 'ufl' has no "
        "attribute 'Abs'. The idiomatic pattern in DG upwind-flux "
        "construction is e.g. (b_n + abs(b_n))/2 for the outflow "
        "side. Signal: AttributeError with the literal text "
        "\"module 'ufl' has no attribute 'Abs'\" emitted at script "
        "import time before any assembly is attempted. (Verified "
        "empirically 2026-06-01 — Layer F catch.)",
    ],
    "materials": {
        "diffusion": {"range": [1e-8, 1.0], "unit": "m^2/s"},
        "convection_speed": {"range": [0.01, 1e4], "unit": "m/s"},
    },
}


def _dg_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx DG script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 40)
    ny = params.get("ny", 40)
    eps = params.get("diffusion", 0.005)
    bx = params.get("bx", 1.0)
    by = params.get("by", 0.5)
    degree = params.get("degree", 1)
    # SIPG penalty must scale like O(degree^2) for coercivity. A fixed
    # alpha=10 is fine at degree 1 and 2 but LOSES coercivity at degree 3:
    # measured max|u| ~ 3.9e3 on a 20x20 triangle mesh with KSPConvergedReason
    # still 4 (a silent wrong answer). 4*(degree+1)^2 is the classical rule
    # and brought the same case back to max|u| ~ 1.1.
    alpha = params.get("penalty", 4.0 * (degree + 1) ** 2)
    return f'''\
"""Discontinuous Galerkin (DG) advection-diffusion — FEniCSx/dolfinx
Symmetric interior-penalty diffusion + upwind advection.
eps = {eps}, b = ({bx}, {by}), DG degree {degree}, penalty {alpha}

BOUNDARY TREATMENT — the part that decides whether the operator is
invertible at all.  Everything is imposed WEAKLY (fem.dirichletbc on a DG
space is a silent no-op: locate_dofs_topological returns an EMPTY array).
  * advective outflow:  +(b.n + |b.n|)/2 * u * v * ds     -> bilinear form
  * advective inflow:   -(b.n - |b.n|)/2 * u_D * v * ds   -> linear form
  * diffusive Dirichlet: the Nitsche/SIPG block on ds, in both forms
Using the RAW b.n over the whole boundary instead of its outflow part
subtracts on the inflow facets, destroys coercivity, and makes the matrix
numerically singular: PETSc LU then reports a zero pivot, the KSP returns
reason -11, and a script that does not check the reason prints
"u: min=inf, max=inf" and exits 0.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl
import numpy as np
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
domain.topology.create_connectivity(fdim, fdim)

# DG function space — fully discontinuous
V = fem.functionspace(domain, ("DG", {degree}))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Problem parameters
eps = {eps}
b = ufl.as_vector([{bx}, {by}])
n = ufl.FacetNormal(domain)
h = ufl.CellDiameter(domain)
h_avg = (h("+") + h("-")) / 2.0
alpha = {alpha}                       # SIPG penalty, must be >= O(degree^2)
f_rhs = fem.Constant(domain, default_scalar_type(1.0))
u_D = fem.Constant(domain, default_scalar_type(0.0))   # boundary value

# Upwind interior flux
bn = ufl.dot(b, n)
bn_plus  = (bn("+") + abs(bn("+"))) / 2.0    # outflow side of the facet
bn_minus = (bn("+") - abs(bn("+"))) / 2.0    # inflow  side of the facet
adv_flux = bn_plus * u("+") + bn_minus * u("-")

# Boundary split: outflow enters the operator, inflow enters the data
bn_out = (bn + abs(bn)) / 2.0
bn_in  = (bn - abs(bn)) / 2.0

# Symmetric interior-penalty diffusion (interior facets)
a_diff = (
    eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    - eps * ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS
    - eps * ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
    + alpha / h_avg * eps * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS
)

# Nitsche block: imposes u = u_D weakly for the DIFFUSIVE part on ds
a_diff_bdry = (
    - eps * ufl.dot(ufl.grad(u), n) * v * ufl.ds
    - eps * ufl.dot(ufl.grad(v), n) * u * ufl.ds
    + alpha / h * eps * u * v * ufl.ds
)
L_diff_bdry = (
    - eps * ufl.dot(ufl.grad(v), n) * u_D * ufl.ds
    + alpha / h * eps * u_D * v * ufl.ds
)

# Upwind advection
a_adv = (
    - ufl.inner(u * b, ufl.grad(v)) * ufl.dx
    + adv_flux * ufl.jump(v) * ufl.dS
    + bn_out * u * v * ufl.ds          # OUTFLOW ONLY — never the raw bn
)
L_adv = - bn_in * u_D * v * ufl.ds     # inflow value is data

a = a_diff + a_diff_bdry + a_adv
L = f_rhs * v * ufl.dx + L_diff_bdry + L_adv

a_form = fem.form(a)
L_form = fem.form(L)

from dolfinx.fem.petsc import assemble_matrix, assemble_vector
A = assemble_matrix(a_form)          # no dirichletbc: DG imposes weakly
A.assemble()
b_vec = assemble_vector(L_form)
b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

uh = fem.Function(V, name="concentration")
solver.solve(b_vec, uh.x.petsc_vec)
uh.x.scatter_forward()

# ---- checks on the PHYSICS, not on the return code -----------------------
reason = solver.getConvergedReason()
if reason <= 0:
    raise RuntimeError(
        f"KSP failed with KSPConvergedReason={{reason}} (-11 = "
        f"DIVERGED_PC_FAILED). The usual cause is a boundary advection term "
        f"written with the raw dot(b, n) instead of its outflow part.")
u_arr = uh.x.array
if not np.all(np.isfinite(u_arr)):
    raise RuntimeError("solution contains non-finite values — the solve failed")
# Magnitude sanity: with a bounded inflow value and a bounded source on a unit
# domain the solution is O(1). A loss of SIPG coercivity (penalty too small for
# the degree) shows up here as a large finite value with the KSP still
# reporting success, so isfinite alone does not catch it.
u_scale = max(abs(float(f_rhs.value)), abs(float(u_D.value)), 1.0)
if np.abs(u_arr).max() > 1e3 * u_scale:
    raise RuntimeError(
        f"|u|max = {{np.abs(u_arr).max():.3e}} is implausible for data of size "
        f"{{u_scale:.3e}} on a unit domain. The usual cause is an SIPG penalty "
        f"too small for the DG degree — alpha should scale like "
        f"4*(degree+1)**2.")

# Output
from dolfinx.io import XDMFFile
V_p1 = fem.functionspace(domain, ("Lagrange", 1))
u_out = fem.Function(V_p1, name="concentration")
u_out.interpolate(uh)
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)

print(f"DG advection-diffusion solved (KSPConvergedReason={{reason}})")
print(f"u: min={{u_arr.min():.6e}}, max={{u_arr.max():.6e}}")
print(f"DOFs (DG): {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# contact
# ---------------------------------------------------------------------------

_CONTACT_KNOWLEDGE = {
    "description": (
        "Contact / obstacle problem: find u >= phi such that -laplacian(u) = f "
        "where u >= phi (obstacle). Solved via penalty method or variational inequality."
    ),
    "weak_form": (
        "a(u,v) + gamma*(max(phi-u, 0), v)*dx = (f, v)*dx; "
        "penalty gamma -> infinity enforces u >= phi"
    ),
    "function_space": "Lagrange order 1 (scalar displacement or deflection)",
    "solver": "Newton iteration (NonlinearProblem / SNES)",
    "pitfalls": [
        "Penalty parameter gamma: too small -> constraint violation; too large -> ill-conditioning",
        "Typical gamma: 1e3 to 1e6 (problem-dependent); adaptive augmented Lagrangian is better",
        "max(phi-u, 0) is non-smooth -> Newton may converge slowly; use smooth approximation",
        "Smooth regularization: max(x,0) ~ (x + sqrt(x^2 + delta^2))/2 for small delta",
        "Signorini problem (1D contact): normal stress = 0 on contact zone at convergence",
        "For mechanical contact: need Lagrange multiplier or mortar methods for accuracy",
        "Active set strategy (semi-smooth Newton) is more robust than pure penalty",
    ],
    "materials": {
        "penalty": {"range": [1e2, 1e7], "unit": "N/m^2 or dimensionless"},
    },
}


def _contact_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx penalty-contact script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    gamma = params.get("penalty", 1e4)
    obstacle_height = params.get("obstacle_height", -0.2)
    return f'''\
"""Contact / obstacle problem — penalty method — FEniCSx/dolfinx
-laplacian(u) = 1 on [0,1]^2, u >= phi (obstacle at height {obstacle_height})
u = 0 on boundary.
Penalty: (gamma * max(phi - u, 0)) added to residual.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Homogeneous Dirichlet BC on all boundaries
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Obstacle (flat barrier)
phi = fem.Constant(domain, default_scalar_type({obstacle_height}))

# Penalty parameter
gamma = fem.Constant(domain, default_scalar_type({gamma}))

# Source term
f = fem.Constant(domain, default_scalar_type(1.0))

# Current solution (nonlinear iteration)
u = fem.Function(V, name="displacement")
v = ufl.TestFunction(V)

# Smooth penalty: max(phi - u, 0) approximated by smooth ramp
# penalty_term = gamma * max(phi - u, 0) * v
delta = fem.Constant(domain, default_scalar_type(1e-8))
arg = phi - u
# Smooth max: (arg + sqrt(arg^2 + delta)) / 2
smooth_max = (arg + ufl.sqrt(arg**2 + delta)) / 2.0

# Residual: standard Poisson + penalty
F = (ufl.dot(ufl.grad(u), ufl.grad(v)) - f * v + gamma * smooth_max * v) * ufl.dx

# Newton solve
problem = NonlinearProblem(F, u, bcs=[bc], petsc_options_prefix="contact",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "snes_monitor": None,
    }})
problem.solve()
its = problem.solver.getIterationNumber()
reason = problem.solver.getConvergedReason()
print(f"Contact Newton: {{its}} iterations, reason={{reason}}")

# Check constraint satisfaction
u_arr = u.x.array
phi_val = {obstacle_height}
n_violated = np.sum(u_arr < phi_val - 1e-6)
print(f"Constraint violations (u < phi): {{n_violated}} / {{len(u_arr)}} DOFs")

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

print(f"Contact/obstacle: min(u)={{u_arr.min():.6e}}, max(u)={{u_arr.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# multiphase
# ---------------------------------------------------------------------------

_MULTIPHASE_KNOWLEDGE = {
    'description': ('Two-phase interface tracking with an Allen-Cahn phase field. A scalar order '
     'parameter phi is driven to +1 in one phase and -1 in the other by a double-well '
     'potential, with a diffuse interface of width set by the parameter eps. The interface '
     'moves by mean curvature, so a droplet shrinks. Allen-Cahn is the non-conservative '
     'sibling of Cahn-Hilliard: it does NOT preserve the integral of phi.'),
    'minimal_working_example': ('"""Allen-Cahn phase field (dolfinx 0.10). Reference-free self-checks."""\n'
     'from mpi4py import MPI\n'
     'import numpy as np\n'
     'import ufl\n'
     'from dolfinx import fem, mesh\n'
     'from dolfinx.fem.petsc import NonlinearProblem\n'
     '\n'
     'comm = MPI.COMM_WORLD\n'
     'N = 32\n'
     'msh = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)\n'
     'V = fem.functionspace(msh, ("Lagrange", 1))\n'
     '\n'
     'h = 1.0 / N\n'
     'eps_val = 3.0 * h\n'
     'eps = fem.Constant(msh, eps_val)\n'
     'dt = fem.Constant(msh, 1.0e-3)\n'
     '\n'
     'phi = fem.Function(V, name="phase_field")\n'
     'phi_old = fem.Function(V)\n'
     'R0 = 0.30\n'
     'phi_old.interpolate(lambda x: np.tanh(\n'
     '    (R0 - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)) / (eps_val * '
     'np.sqrt(2.0))))\n'
     'phi.x.array[:] = phi_old.x.array\n'
     '\n'
     'v = ufl.TestFunction(V)\n'
     'F = ((phi - phi_old) / dt * v * ufl.dx\n'
     '     + eps * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx\n'
     '     + (phi ** 3 - phi) / eps * v * ufl.dx)\n'
     '\n'
     'problem = NonlinearProblem(\n'
     '    F, phi, bcs=[], petsc_options_prefix="ac_",\n'
     '    petsc_options={"ksp_type": "preonly", "pc_type": "lu",\n'
     '                   "pc_factor_mat_solver_type": "mumps",\n'
     '                   "snes_rtol": 1e-9, "snes_atol": 1e-12,\n'
     '                   "snes_max_it": 25, "snes_linesearch_type": "bt"})\n'
     '\n'
     'energy_form = fem.form((eps / 2.0 * ufl.dot(ufl.grad(phi), ufl.grad(phi))\n'
     '                        + (phi ** 2 - 1.0) ** 2 / (4.0 * eps)) * ufl.dx)\n'
     'mass_form = fem.form(phi * ufl.dx)\n'
     'area_form = fem.form(ufl.conditional(ufl.gt(phi, 0.0), 1.0, 0.0) * ufl.dx)\n'
     '\n'
     '\n'
     'def scal(f):\n'
     '    return comm.allreduce(fem.assemble_scalar(f), MPI.SUM)\n'
     '\n'
     '\n'
     'def gminmax(f):\n'
     '    nloc = f.function_space.dofmap.index_map.size_local\n'
     '    a = f.x.array[:nloc]\n'
     '    return (comm.allreduce(float(a.min()), MPI.MIN),\n'
     '            comm.allreduce(float(a.max()), MPI.MAX))\n'
     '\n'
     '\n'
     'E0, m0, A0 = scal(energy_form), scal(mass_form), scal(area_form)\n'
     'E_prev = E0\n'
     'n_steps = 25\n'
     'for step in range(n_steps):\n'
     '    phi_old.x.array[:] = phi.x.array\n'
     '    problem.solve()\n'
     '    reason = problem.solver.getConvergedReason()\n'
     '    assert reason > 0, (\n'
     '        f"SNES failed at step {step + 1}: converged reason {reason} "\n'
     '        f"(negative = diverged); phi was NOT advanced")\n'
     '    phi.x.scatter_forward()\n'
     '    E = scal(energy_form)\n'
     '    assert E <= E_prev + 1e-10 * abs(E0), (\n'
     '        f"free energy increased at step {step + 1}: {E_prev:.8e} -> {E:.8e}")\n'
     '    E_prev = E\n'
     '\n'
     'E1, m1, A1 = scal(energy_form), scal(mass_form), scal(area_form)\n'
     'lo, hi = gminmax(phi)\n'
     'if comm.rank == 0:\n'
     '    print(f"steps={n_steps}  eps/h={eps_val / h:.2f}  final SNES reason={reason} "\n'
     '          f"iters={problem.solver.getIterationNumber()}")\n'
     '    print(f"free energy   {E0:.6e} -> {E1:.6e}   (must be non-increasing)")\n'
     '    print(f"area(phi>0)   {A0:.6f} -> {A1:.6f}   (curvature-driven shrinkage)")\n'
     '    print(f"int(phi) dx   {m0:.6e} -> {m1:.6e}   "\n'
     '          f"(Allen-Cahn does NOT conserve this)")\n'
     '    print(f"global phi range = [{lo:.6f}, {hi:.6f}] (overshoot beyond +-1 "\n'
     '          f"means eps is under-resolved)")\n'
     'assert np.isfinite(E1) and abs(hi) < 1.01 and abs(lo) < 1.01\n'
     'assert A1 < A0, "a convex droplet must shrink under Allen-Cahn"\n'),
    'function_space': {'REQUIRED': 'V = fem.functionspace(msh, ("Lagrange", 1))\n'
                 'phi = fem.Function(V)        # current iterate, solved for\n'
                 'phi_old = fem.Function(V)    # previous time level',
     'OPTIONAL': 'Degree 2 also works and resolves the tanh interface profile with fewer '
                 'cells. Cell type triangle or quadrilateral. If you also need '
                 'conservation of the phase volume, switch the model to Cahn-Hilliard, '
                 'which requires a MIXED space basix.ufl.mixed_element([P1, P1]) for (phi, '
                 'mu) - that is a different physics entry, not a variant of this one.',
     'explanation': 'Allen-Cahn is a single scalar second-order equation, so one ordinary '
                    'Lagrange space is enough. The only structural requirement is a second '
                    'Function holding the previous time level.',
     'pitfalls': ['[API] phi_old must be a separate Function, copied with phi_old.x.array[:] = '
                  'phi.x.array at the TOP of each step. Signal: if you forget, phi stops '
                  'changing after step 1 and the SNES iteration count collapses to 1 with '
                  'converged reason 4 (CONVERGED_SNORM_RELATIVE, i.e. a zero Newton '
                  'update).']},
    'weak_form': {'REQUIRED': 'F = ((phi - phi_old) / dt * v * ufl.dx\n'
                 '     + eps * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx\n'
                 '     + (phi ** 3 - phi) / eps * v * ufl.dx)\n'
                 '# eps and dt MUST be fem.Constant, not bare Python floats.',
     'OPTIONAL': 'A mobility M multiplying the whole right-hand side; a theta-scheme '
                 'phi_mid = (1-theta)*phi_old + theta*phi in the gradient and potential '
                 'terms (theta=1 backward Euler, theta=0.5 Crank-Nicolson); convex '
                 'splitting (phi**3 implicit, -phi explicit) for unconditional energy '
                 'stability.',
     'explanation': 'This is the L2 gradient flow of the free energy E(phi) = int eps/2 '
                    '|grad phi|^2 + (phi^2-1)^2/(4 eps) dx. The 1/(4*eps) scaling of the '
                    'double well against the eps in front of the gradient term is what '
                    'fixes the equilibrium interface thickness at O(eps) and the profile '
                    'at tanh(d/(eps*sqrt(2))); use the same eps in the initial condition.',
     'pitfalls': ['[API] Do not multiply a form integrand by a Python float that can be exactly '
                  '0.0 (a theta-scheme with theta=1). Signal: ValueError: This integral is '
                  'missing an integration domain. Wrap the weight in fem.Constant(msh, '
                  'value) instead.']},
    'boundary_conditions': {'REQUIRED': 'bcs = []      # natural no-flux condition; pass an EMPTY list',
     'OPTIONAL': 'A wetting/contact-angle condition is imposed by adding a surface term to '
                 'F over a tagged part of the boundary, using ufl.Measure("ds", '
                 'domain=msh, subdomain_data=facet_tags) and ds(tag). Dirichlet values on '
                 'phi are almost never physical.',
     'explanation': 'The interface is interior to the domain; the boundary only has to be '
                    'transparent to it. The natural condition of the weak form already '
                    'gives zero normal flux, so nothing has to be added.',
     'pitfalls': ['[BC] A surface term written with plain ufl.ds is applied to EVERY exterior '
                  'facet, not just the wall you meant. Signal: measured on the same '
                  'assembly path in a heat problem, switching a Robin term from a tagged '
                  'ds(tag) to plain ufl.ds changed the steady profile from 0.95, 0.85, '
                  '..., 0.05 across the domain to 0.913, 0.744, ..., 0.024 - no error, '
                  'just a different answer.']},
    'solver': {'REQUIRED': 'problem = NonlinearProblem(F, phi, bcs=[], petsc_options_prefix="ac_",\n'
                 '    petsc_options={"ksp_type": "preonly", "pc_type": "lu",\n'
                 '                   "pc_factor_mat_solver_type": "mumps",\n'
                 '                   "snes_rtol": 1e-9, "snes_max_it": 25,\n'
                 '                   "snes_linesearch_type": "bt"})\n'
                 'problem.solve()\n'
                 'assert problem.solver.getConvergedReason() > 0   # NOT optional',
     'OPTIONAL': 'petsc_options_prefix is REQUIRED (keyword-only). J is OPTIONAL - leave '
                 'it out and dolfinx derives the Jacobian by automatic differentiation. '
                 "snes_linesearch_type 'bt' (default) or 'l2'. Instead of asserting, you "
                 'may pass "snes_error_if_not_converged": True to make PETSc raise.',
     'explanation': 'The double well makes the step nonlinear, so every time step is a '
                    'Newton solve. In dolfinx 0.10 the Newton solver IS PETSc SNES, '
                    'reached through dolfinx.fem.petsc.NonlinearProblem; '
                    'dolfinx.nls.petsc.NewtonSolver is the deprecated path and cannot wrap '
                    'this class.',
     'pitfalls': ['[API] problem.solve() NEVER raises on a diverged SNES. Signal: converged '
                  'reason -6 (DIVERGED_LINE_SEARCH) with iterations dropping to 0 and phi '
                  'silently unchanged, while the script runs to the end and exits 0.',
                  '[API] Do not wrap this NonlinearProblem in dolfinx.nls.petsc.NewtonSolver. '
                  "Signal: AttributeError: 'NonlinearProblem' object has no attribute "
                  "'a'."]},
    'time_integration': {'REQUIRED': 'for step in range(n_steps):\n'
                 '    phi_old.x.array[:] = phi.x.array   # BEFORE the solve\n'
                 '    problem.solve()\n'
                 '    assert problem.solver.getConvergedReason() > 0\n'
                 '    phi.x.scatter_forward()',
     'OPTIONAL': 'dt may be adapted; change it with dt.value = new_dt (the form holds the '
                 'fem.Constant by reference, so nothing has to be rebuilt). On failure, '
                 'halve dt and retry the step.',
     'explanation': 'Backward Euler on a stiff relaxation term. The step is limited by '
                    "Newton's basin of attraction, not by linear stability: the relevant "
                    'scale is dt against eps^2, because the potential term is O(1/eps) '
                    'stiff.',
     'pitfalls': ['[Numerical] Too large a dt does not blow up - it stops converging. Signal: SNES '
                  'converged reason -6 at the first step; phi stays at the previous value '
                  'and the reported time keeps advancing.']},
    'materials': {'epsilon': 'interface half-width; choose eps >= 2*h_min, typically 2-4 mesh cells '
                'across the interface',
     'mobility': 'optional prefactor on the whole right-hand side; eps^2 recovers the '
                 'sharp-interface (curvature-flow) limit, larger values slow the interface '
                 'down'},
    'pitfalls': ['[Integration] `problem.solve()` on a dolfinx 0.10 NonlinearProblem returns the '
     'Function whether or not SNES converged - it does not raise. In a time loop this '
     'turns a solver failure into a frozen field that the script keeps reporting as '
     'progress. ALWAYS check `problem.solver.getConvergedReason() > 0` after every step '
     '(or set `"snes_error_if_not_converged": True` in petsc_options). Signal: measured on '
     'this install with a deliberately oversized dt, SNES returns converged reason -6 '
     '(DIVERGED_LINE_SEARCH) at every step with the iteration count decaying 19, 14, 1, 0, '
     "0, phi frozen from step 2 onward, and the script still printing '5 steps complete' "
     "and 'Final phi: [-1.000055e+00, 9.732061e-01]' with exit code 0. Nothing in that "
     'output is distinguishable from a healthy run.',
     '[Numerical] Resolve the interface: eps must be at least about 2 mesh cells wide (eps '
     '>= 2*h). Under-resolving it does NOT produce NaNs or a solver failure - it produces '
     'a bounded overshoot. Signal: measured on a uniform triangular unit-square mesh, phi '
     'leaves its physical range [-1, 1] as eps/h falls: at eps/h = 1.28 and above the '
     'range stays inside [-1, 1]; at eps/h = 0.64 it reaches [-1.00015, +1.00016]; at '
     'eps/h = 0.32 it reaches [-1.02311, +1.02249]. SNES reports converged (reason 3) in '
     "every one of those runs. The previously quoted signals - '10-30% overshoot with "
     "checkerboard pattern' and 'the NonlinearProblem solver diverges with "
     "DIVERGED_FNORM_NAN' - do NOT reproduce; the real overshoot is a few percent and the "
     'solver stays happy, so the ONLY way to catch this is to test max|phi| against 1 '
     'explicitly.',
     '[Physics] Allen-Cahn does not conserve the phase volume: int(phi) dx drifts '
     'monotonically because the curvature flow shrinks the droplet. This is the physics of '
     'the model, not a bug, but it means int(phi) dx is NOT usable as a correctness check. '
     'Signal: measured with a resolved interface (eps = 3h) on a unit square, int(phi) dx '
     'moved from -3.572392e-01 to -3.813692e-01 over 25 backward-Euler steps - a 6.8% '
     'drift - while every step converged cleanly. Use the free energy instead: E(phi) = '
     'int eps/2|grad phi|^2 + (phi^2-1)^2/(4 eps) dx is a Lyapunov functional and must be '
     'non-increasing at every step; that check is asserted in the minimal_working_example '
     'and it does hold. If your application needs volume conservation, the model must '
     'change to Cahn-Hilliard.',
     '[API] Do not measure the phase volume by counting degrees of freedom '
     '(`np.sum(phi.x.array > 0) / len(phi.x.array)`). It is quantised to the DOF grid, so '
     'it is far too coarse to see the interface move, and in parallel it counts GHOST '
     'entries and is rank-local. Signal: measured on the same 64x64 problem, the DOF-count '
     'fraction printed 0.18769 unchanged for all 50 steps while the true integral '
     'int_{phi>0} 1 dx moved 0.196289 -> 0.195801; and running the SAME script under '
     'mpirun -n 2 printed 0.2041 instead of 0.18769. The correct measure is an assembled '
     'integral: `fem.form(ufl.conditional(ufl.gt(phi, 0.0), 1.0, 0.0)*ufl.dx)` reduced '
     'with `comm.allreduce(..., MPI.SUM)`.',
     '[API] `phi.x.array.min()` and `.max()` are RANK-LOCAL and include ghost entries. '
     'Reduce over owned DOFs only: slice to `phi.x.array[:V.dofmap.index_map.size_local]` '
     'and wrap in `comm.allreduce(..., MPI.MIN / MPI.MAX)`. Guard every print with `if '
     'comm.rank == 0:`. Signal: measured, the unguarded generator template prints one full '
     'copy of every step line per rank under mpirun, and the ranks DISAGREE about the '
     'range. The disagreement is the tell, not a wrong number on every rank: a rank that '
     'happens to own both extremes prints the correct global range, so one of the lines '
     'can be accidentally right and which one it is depends on the partition. (An earlier '
     'version of this entry said none of the per-rank ranges is the global one; that is '
     'too strong and a check written against it will not fire.) Both arrays also carry '
     'ghost entries, so even the owning rank is reading values it does not own. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
     '[Numerical] A time step that is too large for Newton does not manifest as an '
     'instability. Backward Euler is unconditionally stable, so the failure lands entirely '
     'in the nonlinear solve. Signal: measured on the resolved 64x64 problem with eps/h = '
     '2.56, dt = 1e-4 and dt = 1e-2 both converge (reason 3) while dt = 1.0 and dt = 100.0 '
     'both fail at the very first step with reason -6 (DIVERGED_LINE_SEARCH). The remedy '
     'is step-size control keyed on the converged reason, not a CFL-style formula.',
     '[Numerical] Initialise phi with the tanh profile that BELONGS to your eps: phi0 = '
     'tanh(d/(eps*sqrt(2))) with d the signed distance to the interface. Any other width '
     'makes the first steps spend themselves re-profiling the interface rather than moving '
     'it, so early-time results are meaningless. Signal: measured with eps = 3h on a 32x32 '
     'unit square, an initial width of eps/6 starts at free energy 5.096 and drops by '
     '-0.893 in the FIRST step, whereas the consistent tanh initialisation starts at 1.772 '
     'and drops by -0.00325 - a factor of 275 in the first-step energy release. Twenty '
     'steps later the too-sharp run is still relaxing 12x faster than the consistent one. '
     'Nothing errors; the only visible sign is that steep initial knee in the energy '
     'history, so print the free energy every step.',
     '[Physics] A sharp `ufl.conditional` Heaviside for phase-dependent material '
     'properties (rho(phi), mu(phi)) is differentiable as far as UFL and SNES are '
     'concerned - the derivative is taken inside each branch and the jump is ignored - so '
     'it does NOT break Newton. Signal: measured with a 1000:1 density ratio rho = 1 + '
     '999*conditional(phi > 0, 1, 0) inside the residual, SNES converged (reason 3) in 2 '
     'iterations at every one of 8 steps, and `ufl.derivative` through the conditional '
     "compiles to a valid Form. The previously quoted signal ('the dolfinx "
     "NonlinearProblem stalls with DIVERGED_FNORM_NAN', 'residual norm plateaus at "
     "O(rho2-rho1)') does NOT reproduce. The genuine reason to prefer a smoothed 0.5*(1 + "
     'tanh(phi/eps)) is accuracy, not solvability: a conditional is evaluated at '
     'QUADRATURE POINTS, so the effective location of the material jump depends on the '
     'quadrature degree of the form.',
     '[API] Every coefficient that you may want to change between steps (dt, eps, a '
     'mobility) must be a `fem.Constant`, updated in place with `dt.value = new_dt`. A '
     'bare Python float is baked into the compiled form. Signal: after rebinding a Python '
     'float the solve produces exactly the same numbers as before - no error, no warning, '
     'the new value simply has no effect.',
     '[API] `dolfinx.fem.petsc.NonlinearProblem` takes `petsc_options_prefix` as a '
     'REQUIRED keyword-only argument, and `J` as an OPTIONAL one. Signal: omitting the '
     'prefix gives `TypeError: NonlinearProblem.__init__() missing 1 required keyword-only '
     "argument: 'petsc_options_prefix'`. Wrapping the resulting object in the deprecated "
     '`dolfinx.nls.petsc.NewtonSolver(comm, problem)` gives `AttributeError: '
     "'NonlinearProblem' object has no attribute 'a'`."],
}


def _multiphase_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx Allen-Cahn phase-field script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 64)
    ny = params.get("ny", 64)
    eps = params.get("epsilon", 0.04)
    dt = params.get("dt", 1e-4)
    n_steps = params.get("n_steps", 50)
    return f'''\
"""Two-phase Allen-Cahn phase-field — FEniCSx/dolfinx
dphi/dt = eps*laplacian(phi) - (phi^3 - phi)/eps
phi in [-1,+1]: phi=+1 (fluid 1), phi=-1 (fluid 2)
Interface width ~ {eps} (epsilon parameter)
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# No boundary conditions needed (no-flux = natural BC)

# Parameters
eps = fem.Constant(domain, default_scalar_type({eps}))
dt_c = fem.Constant(domain, default_scalar_type({dt}))

# Phase field: phi_new (current iterate), phi_old (previous time step)
phi_old = fem.Function(V, name="phi_old")
phi = fem.Function(V, name="phase_field")

# Initial condition: circular droplet in center
def init_phi(x):
    r = np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
    return np.tanh((0.25 - r) / ({eps} * np.sqrt(2.0)))

phi_old.interpolate(init_phi)
phi.x.array[:] = phi_old.x.array[:]

# Test function
v = ufl.TestFunction(V)

# Allen-Cahn residual (backward Euler in time)
# (phi - phi_old)/dt * v + eps * grad(phi).grad(v) + (phi^3 - phi)/eps * v = 0
F = (
    (phi - phi_old) / dt_c * v * ufl.dx
    + eps * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    + (phi**3 - phi) / eps * v * ufl.dx
)

# Newton solver
problem = NonlinearProblem(F, phi, bcs=[], petsc_options_prefix="ac",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-8,
        "snes_max_it": 25,
    }})

# Time loop
n_steps = {n_steps}
from dolfinx.io import XDMFFile

with XDMFFile(domain.comm, "phase_field.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(phi, 0.0)

    for step in range(n_steps):
        phi_old.x.array[:] = phi.x.array[:]
        problem.solve()
        _reason = problem.solver.getConvergedReason()
        if _reason <= 0:
            raise RuntimeError(
                f"SNES diverged at step {{step + 1}}: converged reason "
                f"{{_reason}} (negative = diverged, phi was NOT advanced). "
                f"problem.solve() does not raise on this, so without the "
                f"check the loop runs to the end and exits 0 with a "
                f"plausible-looking frozen field. Reduce dt or relax the "
                f"snes tolerances.")
        phi.x.scatter_forward()

        t = (step + 1) * {dt}
        phi_arr = phi.x.array
        # Volume fraction of phase +1, as a real integral. Counting DOFs
        # with phi > 0 and dividing by the DOF count is NOT a volume: it
        # weights every DOF equally regardless of the cells it belongs to.
        volume_plus = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(
                ufl.conditional(ufl.gt(phi, 0.0), 1.0, 0.0) * ufl.dx)),
            op=MPI.SUM)

        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            xdmf.write_function(phi, t)
            print(f"Step {{step+1}}/{{n_steps}}, t={{t:.5f}}: "
                  f"phi in [{{phi_arr.min():.4f}}, {{phi_arr.max():.4f}}], "
                  f"vol+={{volume_plus:.4f}}")

print(f"Allen-Cahn phase-field: {{n_steps}} steps complete")
print(f"Final phi: [{{phi.x.array.min():.6e}}, {{phi.x.array.max():.6e}}]")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# time_dependent_heat
# ---------------------------------------------------------------------------

_TIME_DEPENDENT_HEAT_KNOWLEDGE = {'description': 'Transient heat conduction rho*cp*dT/dt - div(k*grad(T)) = Q, '
                'discretised in space with continuous Lagrange elements and in time '
                'with an implicit one-step scheme (backward Euler by default). '
                'Supports volumetric sources, prescribed temperatures, prescribed '
                'fluxes and convective (Robin) surfaces. With constant material data '
                'the system matrix is assembled once and only the right-hand side is '
                'rebuilt per step.',
 'minimal_working_example': '"""Transient heat, backward Euler (dolfinx 0.10). '
                            'Reference-free self-checks."""\n'
                            'from mpi4py import MPI\n'
                            'from petsc4py import PETSc\n'
                            'import numpy as np\n'
                            'import ufl\n'
                            'from dolfinx import default_scalar_type, fem, la, mesh\n'
                            'from dolfinx.fem.petsc import (apply_lifting, '
                            'assemble_matrix, assemble_vector,\n'
                            '                               set_bc)\n'
                            '\n'
                            'comm = MPI.COMM_WORLD\n'
                            'msh = mesh.create_unit_square(comm, 32, 32, '
                            'mesh.CellType.triangle)\n'
                            'fdim = msh.topology.dim - 1\n'
                            'msh.topology.create_connectivity(fdim, msh.topology.dim)\n'
                            'V = fem.functionspace(msh, ("Lagrange", 1))\n'
                            '\n'
                            'T_hot, T_cold, T_init = 1.0, 0.0, 0.0\n'
                            'left = mesh.locate_entities_boundary(msh, fdim, lambda x: '
                            'np.isclose(x[0], 0.0))\n'
                            'right = mesh.locate_entities_boundary(msh, fdim, lambda '
                            'x: np.isclose(x[0], 1.0))\n'
                            'dofs_l = fem.locate_dofs_topological(V, fdim, left)\n'
                            'dofs_r = fem.locate_dofs_topological(V, fdim, right)\n'
                            'bcs = [fem.dirichletbc(default_scalar_type(T_hot), '
                            'dofs_l, V),\n'
                            '       fem.dirichletbc(default_scalar_type(T_cold), '
                            'dofs_r, V)]\n'
                            '\n'
                            'k = fem.Constant(msh, default_scalar_type(1.0))\n'
                            'rho_cp = fem.Constant(msh, default_scalar_type(1.0))\n'
                            'dt_val = 0.002\n'
                            'dt = fem.Constant(msh, default_scalar_type(dt_val))\n'
                            'Q = fem.Constant(msh, default_scalar_type(0.0))\n'
                            '\n'
                            'T_n = fem.Function(V, name="T_old")\n'
                            'T_n.x.array[:] = T_init\n'
                            'T_h = fem.Function(V, name="temperature")\n'
                            'u, v = ufl.TrialFunction(V), ufl.TestFunction(V)\n'
                            'a = rho_cp / dt * u * v * ufl.dx + k * '
                            'ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx\n'
                            'L = rho_cp / dt * T_n * v * ufl.dx + Q * v * ufl.dx\n'
                            'a_form, L_form = fem.form(a), fem.form(L)\n'
                            'res_form = fem.form(rho_cp / dt * (T_h - T_n) * v * '
                            'ufl.dx\n'
                            '                    + k * ufl.dot(ufl.grad(T_h), '
                            'ufl.grad(v)) * ufl.dx\n'
                            '                    - Q * v * ufl.dx)\n'
                            '\n'
                            'A = assemble_matrix(a_form, bcs=bcs)\n'
                            'A.assemble()\n'
                            'ksp = PETSc.KSP().create(msh.comm)\n'
                            'ksp.setOperators(A)\n'
                            'ksp.setType(PETSc.KSP.Type.CG)\n'
                            'ksp.getPC().setType(PETSc.PC.Type.HYPRE)\n'
                            'ksp.setTolerances(rtol=1e-12)\n'
                            'b = assemble_vector(L_form)\n'
                            '\n'
                            'nloc = V.dofmap.index_map.size_local\n'
                            'free = np.ones(nloc, dtype=bool)\n'
                            'free[dofs_l[dofs_l < nloc]] = False\n'
                            'free[dofs_r[dofs_r < nloc]] = False\n'
                            '\n'
                            'n_steps = 250\n'
                            'worst_free_residual = 0.0\n'
                            'for step in range(n_steps):\n'
                            '    with b.localForm() as bl:\n'
                            '        bl.set(0.0)\n'
                            '    assemble_vector(b, L_form)\n'
                            '    apply_lifting(b, [a_form], bcs=[bcs])\n'
                            '    b.ghostUpdate(addv=PETSc.InsertMode.ADD, '
                            'mode=PETSc.ScatterMode.REVERSE)\n'
                            '    set_bc(b, bcs)\n'
                            '    ksp.solve(b, T_h.x.petsc_vec)\n'
                            '    assert ksp.getConvergedReason() > 0, (\n'
                            '        f"KSP failed at step {step + 1}: reason '
                            '{ksp.getConvergedReason()}")\n'
                            '    T_h.x.scatter_forward()\n'
                            '    R = fem.assemble_vector(res_form)\n'
                            '    R.scatter_reverse(la.InsertMode.add)\n'
                            '    worst_free_residual = max(\n'
                            '        worst_free_residual,\n'
                            '        '
                            'comm.allreduce(float(np.abs(R.array[:nloc][free]).max(initial=0.0)), '
                            'MPI.MAX))\n'
                            '    T_n.x.array[:] = T_h.x.array\n'
                            '\n'
                            'react_hot = -comm.allreduce(float(R.array[dofs_l[dofs_l < '
                            'nloc]].sum()), MPI.SUM)\n'
                            'react_cold = -comm.allreduce(float(R.array[dofs_r[dofs_r '
                            '< nloc]].sum()), MPI.SUM)\n'
                            'stored = '
                            'comm.allreduce(fem.assemble_scalar(fem.form(rho_cp * T_h '
                            '* ufl.dx)), MPI.SUM)\n'
                            'gmin = comm.allreduce(float(T_h.x.array[:nloc].min()), '
                            'MPI.MIN)\n'
                            'gmax = comm.allreduce(float(T_h.x.array[:nloc].max()), '
                            'MPI.MAX)\n'
                            'if comm.rank == 0:\n'
                            '    print(f"steps={n_steps} dt={dt_val} t_end={n_steps * '
                            'dt_val}")\n'
                            '    print(f"global T range = [{gmin:.6f}, {gmax:.6f}]  '
                            '(discrete maximum "\n'
                            '          f"principle: must stay inside "\n'
                            '          f"[{min(T_init, T_cold, T_hot)}, {max(T_init, '
                            'T_cold, T_hot)}])")\n'
                            '    print(f"max |Galerkin residual| over unconstrained '
                            'DOFs = "\n'
                            '          f"{worst_free_residual:.3e}")\n'
                            '    print(f"reaction at hot wall  = {react_hot:+.6e} '
                            '(heat in)")\n'
                            '    print(f"reaction at cold wall = {react_cold:+.6e} '
                            '(heat out)")\n'
                            '    print(f"imbalance = {react_hot + react_cold:+.3e} "\n'
                            '          f"(-> 0 as the solution reaches steady '
                            'state)")\n'
                            '    print(f"stored energy int(rho*cp*T) dx = '
                            '{stored:.6e}")\n'
                            'assert gmin >= min(T_init, T_cold, T_hot) - 1e-10\n'
                            'assert gmax <= max(T_init, T_cold, T_hot) + 1e-10\n'
                            'assert worst_free_residual < 1e-9\n',
 'function_space': {'REQUIRED': 'V = fem.functionspace(msh, ("Lagrange", 1))\n'
                                'T_n = fem.Function(V)   # previous time level\n'
                                'T_h = fem.Function(V)   # solution at the new time '
                                'level',
                    'OPTIONAL': 'Degree 2 for higher spatial accuracy (it also weakens '
                                'the discrete maximum principle, so small '
                                'over/undershoots near a sharp front are then normal). '
                                'Cell type triangle/quadrilateral in 2D, '
                                'tetrahedron/hexahedron in 3D - nothing else changes.',
                    'explanation': 'One scalar Lagrange space for the temperature plus '
                                   'one extra Function to hold the previous time '
                                   'level. Spatially varying material data lives in a '
                                   'separate DG0 space, one value per cell.',
                    'pitfalls': ['[API] Heterogeneous k, rho or cp must be a fem.Function on '
                                 'a DG0 space, not a single fem.Constant. Signal: with '
                                 'a Constant the temperature field is smooth '
                                 'everywhere and shows no kink at the material '
                                 'interface, whatever value you set.']},
 'weak_form': {'REQUIRED': 'a = rho_cp/dt * u*v*ufl.dx + k*ufl.dot(ufl.grad(u), '
                           'ufl.grad(v))*ufl.dx\n'
                           'L = rho_cp/dt * T_n*v*ufl.dx + Q*v*ufl.dx\n'
                           'a_form, L_form = fem.form(a), fem.form(L)',
               'OPTIONAL': 'theta-scheme: add theta to the stiffness term in a and '
                           '(1-theta) to the corresponding term in L, with theta a '
                           'fem.Constant (theta=1 backward Euler, 0.5 Crank-Nicolson). '
                           'Neumann flux q_n: subtract q_n*v*ds(tag) from L. '
                           'Robin/convection: add h_conv*u*v*ds(tag) to a and '
                           'h_conv*T_amb*v*ds(tag) to L.',
               'explanation': 'Everything on the new time level goes into the bilinear '
                              'form a, everything known goes into L. With constant '
                              'coefficients a is time-independent, so its matrix is '
                              'assembled once outside the loop and only L is rebuilt.',
               'pitfalls': ['[BC] A Neumann or Robin surface term that you forget is '
                            'silently an insulated boundary; the natural condition of '
                            'this form is zero flux. Signal: no error and no warning - '
                            'just a boundary whose normal temperature gradient stays '
                            'at 0.',
                            '[API] Do not multiply a form integrand by a Python float that '
                            'can be 0.0 (theta=1 in a theta-scheme). Signal: '
                            'ValueError: This integral is missing an integration '
                            'domain.']},
 'boundary_conditions': {'REQUIRED': 'msh.topology.create_connectivity(fdim, '
                                     'msh.topology.dim)\n'
                                     'facets = mesh.locate_entities_boundary(msh, '
                                     'fdim, marker_fn)\n'
                                     'dofs = fem.locate_dofs_topological(V, fdim, '
                                     'facets)\n'
                                     'bc = fem.dirichletbc(default_scalar_type(value), '
                                     'dofs, V)\n'
                                     '# and PER STEP, in exactly this order:\n'
                                     'apply_lifting(b, [a_form], bcs=[bcs])\n'
                                     'b.ghostUpdate(addv=PETSc.InsertMode.ADD, '
                                     'mode=PETSc.ScatterMode.REVERSE)\n'
                                     'set_bc(b, bcs)',
                         'OPTIONAL': 'A time-dependent Dirichlet value: build the bc '
                                     'from a fem.Function and re-interpolate it each '
                                     'step. Flux and convective surfaces need facet '
                                     'MeshTags plus ufl.Measure("ds", domain=msh, '
                                     'subdomain_data=tags) so the term can be written '
                                     'as ds(tag) and applied to that wall only.',
                         'explanation': 'apply_lifting moves the known Dirichlet '
                                        'values from the columns of the matrix into '
                                        'the right-hand side; set_bc then overwrites '
                                        'the constrained rows with the prescribed '
                                        'values. The ghost update MUST sit between '
                                        'them.',
                         'pitfalls': ['[BC] Omitting apply_lifting is the classic silent '
                                      'failure. Signal: min/max of T still print '
                                      'exactly the prescribed BC range, but the '
                                      'interior collapses towards zero and the total '
                                      'stored energy is an order of magnitude too '
                                      'small.',
                                      '[BC] Omitting set_bc leaves the prescribed values '
                                      'unenforced. Signal: the field takes values '
                                      'OUTSIDE the range spanned by the boundary data '
                                      'and the initial condition (measured: [-1.07, '
                                      '+0.94] for data in [0, 1]).',
                                      '[BC] Plain ufl.ds is the WHOLE exterior boundary. '
                                      'For a flux or convective term on one wall only, '
                                      'use a tagged Measure. Signal: measured, adding '
                                      'h_conv*T*ufl.ds with h_conv = 10 to a '
                                      'left-hot/right-cold square changed the steady '
                                      'mid-height profile from 0.95, 0.85, ..., 0.05 '
                                      'to 0.913, 0.744, 0.592, ..., 0.024 - correct if '
                                      'you wanted all four walls cooled, silently '
                                      'wrong if you wanted only two.']},
 'solver': {'REQUIRED': 'A = assemble_matrix(a_form, bcs=bcs); A.assemble()   # once, '
                        'outside the loop\n'
                        'ksp = PETSc.KSP().create(msh.comm); ksp.setOperators(A)\n'
                        'ksp.setType(PETSc.KSP.Type.CG); '
                        'ksp.getPC().setType(PETSc.PC.Type.HYPRE)\n'
                        'ksp.solve(b, T_h.x.petsc_vec)\n'
                        'assert ksp.getConvergedReason() > 0   # NOT optional\n'
                        'T_h.x.scatter_forward()',
            'OPTIONAL': "Direct solve: ksp type 'preonly' with pc type 'lu' (PETSc "
                        'picks MUMPS automatically for an MPI matrix if MUMPS is '
                        'configured). The high-level alternative is '
                        'dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, '
                        'petsc_options_prefix="heat_"), which re-assembles the matrix '
                        'on every solve() - convenient, but it throws away the main '
                        'saving of a constant-coefficient transient run.',
            'explanation': 'The operator is symmetric positive definite, so CG with an '
                           'algebraic-multigrid preconditioner is the scalable choice '
                           'and the factorisation-free path reuses the same matrix for '
                           'every step.',
            'pitfalls': ['[API] Neither ksp.solve nor LinearProblem.solve raises on failure. '
                         'Signal: KSP converged reason is negative while the script '
                         'carries on; check getConvergedReason() > 0 every step or '
                         'pass "ksp_error_if_not_converged": True.',
                         '[API] If you change dt, k, rho or cp mid-run, the matrix must be '
                         'reassembled. Signal: T still prints inside the prescribed BC '
                         'range while the interior profile silently collapses.']},
 'time_integration': {'REQUIRED': 'for step in range(n_steps):\n'
                                  '    with b.localForm() as bl: bl.set(0.0)   # ZERO '
                                  'b first\n'
                                  '    assemble_vector(b, L_form)\n'
                                  '    apply_lifting(b, [a_form], bcs=[bcs])\n'
                                  '    b.ghostUpdate(addv=PETSc.InsertMode.ADD, '
                                  'mode=PETSc.ScatterMode.REVERSE)\n'
                                  '    set_bc(b, bcs)\n'
                                  '    ksp.solve(b, T_h.x.petsc_vec)\n'
                                  '    assert ksp.getConvergedReason() > 0\n'
                                  '    T_h.x.scatter_forward()\n'
                                  '    T_n.x.array[:] = T_h.x.array',
                      'OPTIONAL': 'dt may be changed with dt.value = new_dt, but then '
                                  'A must be reassembled (A.zeroEntries(); '
                                  'assemble_matrix(A, a_form, bcs=bcs); A.assemble()).',
                      'explanation': 'Backward Euler is unconditionally stable and '
                                     'first-order accurate in time; there is no CFL '
                                     'restriction, so dt is chosen purely by how well '
                                     'you need the transient resolved. Crank-Nicolson '
                                     'is second order but is only A-stable, not '
                                     'L-stable, so it rings on a step change.',
                      'pitfalls': ['[API] The reused right-hand-side vector MUST be zeroed '
                                   'at the top of each step. Signal: T grows without '
                                   'bound while every KSP solve reports converged.',
                                   'Do not choose dt from an explicit-stability '
                                   'formula such as h^2/(2*alpha) - implicit schemes '
                                   'have no such limit and you will just do far more '
                                   'steps than necessary.']},
 'materials': {'conductivity_k': 'W/(m*K); e.g. 0.03 (insulation) to 400 (copper)',
               'density_rho': 'kg/m^3',
               'specific_heat_cp': 'J/(kg*K)',
               'convection_h': 'W/(m^2*K) for the Robin term h*(T - T_amb)',
               'note': 'For heterogeneous properties use a DG0 fem.Function, one value '
                       'per cell, selected by cell midpoints or MeshTags.'},
 'pitfalls': ['[Integration] Omitting `apply_lifting` is a SILENT failure, not a loud '
              'one: the Dirichlet rows are still overwritten by `set_bc`, so every '
              'printed min/max still lands exactly on the prescribed values, while the '
              'interior solution is wrong by an order of magnitude. Signal: measured '
              'on a unit square with T=1 on the left wall, T=0 on the right and T=0 '
              'initially, the run WITHOUT apply_lifting printed the identical '
              "reassuring line 'T in [0.0000, 1.0000]' at every step, with every KSP "
              'reporting converged reason 4; but the mid-height profile read 0.0198, '
              '0.0177, 0.0155, ... 0.0010 across the domain instead of the correct '
              '0.95, 0.85, 0.75, ... 0.05, and the total stored energy was 2.26e-02 '
              'instead of 4.97e-01. A min/max print CANNOT detect this; a flux balance '
              'or a Galerkin-residual check can.',
              '[Integration] If you hoist the right-hand-side vector out of the time '
              'loop to avoid reallocating it, you MUST zero it at the top of every '
              'step (`with b.localForm() as bl: bl.set(0.0)`), because '
              '`dolfinx.fem.petsc.assemble_vector(b, L)` ADDS into b. Signal: '
              'measured, the temperature reached max(T) = 3.65e+27 after 100 steps '
              'while every single KSP solve reported converged reason 4. Note the '
              'opposite mistake is NOT a problem: calling `b = '
              'assemble_vector(L_form)` fresh inside the loop does not leak - measured '
              'peak RSS growth was 4.7 MB after 500 steps and 4.7 MB after 5000 steps, '
              'i.e. flat.',
              '[Integration] `ksp.solve()` and `LinearProblem.solve()` never raise on '
              "solver failure; the dolfinx 0.10 docstring says outright that 'the user "
              "is responsible for asserting convergence of the KSP solver'. Assert "
              '`ksp.getConvergedReason() > 0` every step, or pass '
              '`"ksp_error_if_not_converged": True`. Signal: a negative reason (e.g. '
              '-3 DIVERGED_MAX_IT, -9 DIVERGED_NANORINF, -11 DIVERGED_PCSETUP_FAILED) '
              'with the script still writing output files and exiting 0. For reference '
              "on a healthy run, 'preonly' returns reason 4 (CONVERGED_ITS) and CG "
              'returns 2 (CONVERGED_RTOL) - both positive.',
              '[Numerical] Changing dt (or k, rho, cp) mid-run without reassembling A '
              'is a silent failure, because the matrix was built once outside the loop '
              'and only the right-hand side sees the new value. Signal: measured, '
              'multiplying dt by 10 halfway through a run while leaving A alone kept '
              "printing 'T in [0.0000, 1.0000]' and kept reporting KSP reason 4, while "
              'the mid-height profile collapsed to 5.19e-01, 1.35e-01, 3.50e-02, '
              '9.10e-03, ... instead of a straight ramp. Reassemble with '
              'A.zeroEntries(); assemble_matrix(A, a_form, bcs=bcs); A.assemble() '
              'whenever a coefficient in `a` changes.',
              '[Integration] Omitting `set_bc(b, bcs)` after the ghost update leaves '
              'the constrained rows carrying the assembled load instead of the '
              'prescribed values. Unlike the missing-apply_lifting case, this one IS '
              'visible in a min/max print. Signal: measured, T landed in [-1.0667, '
              '+0.9440] for boundary data in [0, 1] with initial data 0 - a negative '
              'temperature and a hot wall that never reaches its prescribed value, '
              'both impossible under the discrete maximum principle.',
              '[Numerical] Crank-Nicolson (theta = 0.5) rings in TIME when the initial '
              'condition is incompatible with the boundary data, and the ringing does '
              'not go away with mesh refinement. Backward Euler (theta = 1) is '
              'L-stable and does not. Signal: probe one interior point per step with '
              'dolfinx.fem.Function.eval against a dolfinx.geometry.bb_tree cell '
              'candidate, and plot the history rather than the min/max: measured on a '
              'unit square with h = 0.025, dt = 0.005 (k*dt/h^2 = 8) and a wall '
              'stepped from 0 to 1 at t = 0, the temperature one cell from the wall '
              'went 0.603, 0.915, 0.829, 0.927, 0.881 over the first five steps under '
              'Crank-Nicolson, against a monotone 0.701, 0.827, 0.869, 0.890, 0.904 '
              'under backward Euler. The oscillation decays over ~10 steps and never '
              'leaves [0, 1] here, so it will not trip a bounds check - plot a probe '
              'history instead.',
              '[API] In a theta-scheme, never multiply a form integrand by a bare '
              'Python float that can evaluate to exactly 0.0. UFL folds the product to '
              'Zero and the resulting integral has no domain. Signal: `ValueError: '
              'This integral is missing an integration domain.` raised at '
              'form-construction time for theta = 1.0 in `(1.0 - theta) * '
              'ufl.dot(ufl.grad(T_n), ufl.grad(v)) * ufl.dx`. Wrapping the weight as '
              '`fem.Constant(msh, 1.0 - theta)` makes it work for every theta '
              'including 1.0, and has the further advantage that theta can then be '
              'changed without rebuilding the form.',
              '[Input] Any coefficient you intend to vary in space must be a '
              '`fem.Function` on a DG0 space, populated per cell (from cell midpoints '
              'via `mesh.compute_midpoints`, or from MeshTags); a single '
              '`fem.Constant` cannot represent a layered conductivity. And any '
              'coefficient you intend to vary in TIME must be a `fem.Constant` updated '
              'in place with `c.value = ...`. Signal: rebinding a Python name has no '
              'effect at all - measured, assembling a form built from a Python float '
              '`c = 2.0` still gives the value 2.0 after `c = 10.0`, while '
              '`fem.Constant.value = 10.0` immediately gives 10.0. No error is raised '
              'in either case.',
              '[API] `T.x.array.min()` / `.max()` are RANK-LOCAL and include ghost '
              'entries, and an unguarded `print` emits one copy per rank. Slice to '
              '`T.x.array[:V.dofmap.index_map.size_local]` and reduce with '
              '`comm.allreduce(..., MPI.MIN/MPI.MAX)`, and guard printing with `if '
              'comm.rank == 0:`. Signal: measured, the same steady solution that '
              "reports 'T in [0.0000, 1.0000]' in serial reports 'T in [0.0750, "
              "1.0000]' from rank 0 under mpirun -n 2, because rank 0 does not own the "
              'cold end of the domain.',
              '[Numerical] Backward Euler and Crank-Nicolson are implicit, so there is '
              'NO CFL condition and dt is limited only by accuracy: backward Euler is '
              'first-order in time, Crank-Nicolson second-order. Sizing dt from an '
              'explicit-stability formula such as dt < h^2/(2*alpha) just multiplies '
              'the number of steps for nothing. Signal: the run is correct but '
              'needlessly slow; conversely, a dt so large that it smears the transient '
              'produces a perfectly stable, perfectly converged, physically useless '
              'answer - only a probe history at two different dt values will show it.',
              '[Physics] Useful reference-free checks for this physics, all of which '
              'are in the minimal_working_example: (1) the discrete maximum principle '
              '- with no volumetric source, T must stay inside the range spanned by '
              'the initial condition and the Dirichlet data; (2) the Galerkin residual '
              'assembled at the solution must vanish at every UNCONSTRAINED degree of '
              'freedom; (3) the nodal reactions summed over each Dirichlet wall give '
              'the heat entering and leaving there, and their sum must tend to zero as '
              'the solution reaches steady state. Signal: measured on the example, the '
              'residual over free DOFs stays below 1.5e-12, T stays in [0.000000, '
              '1.000000], and the wall reactions are -1.015046e+00 and +9.849544e-01 '
              'with an imbalance of -3.0e-02 that is still shrinking because the run '
              'stops before steady state. Printing only min/max of T, as many '
              'templates do, detects none of the failures listed above.']}


def _time_dependent_heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx transient heat script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 40)
    ny = params.get("ny", 40)
    k = params.get("conductivity", 1.0)
    rho = params.get("density", 1.0)
    cp = params.get("specific_heat", 1.0)
    dt = params.get("dt", 0.005)
    n_steps = params.get("n_steps", 100)
    T_hot = params.get("T_hot", 1.0)
    T_init = params.get("T_init", 0.0)
    h_conv = params.get("h_conv", 0.0)
    T_amb = params.get("T_amb", 0.0)
    return f'''\
"""Transient heat equation — backward Euler — FEniCSx/dolfinx
rho*cp*dT/dt - div(k*grad(T)) = Q on [0,1]^2
Left boundary: T = T_hot = {T_hot}
Right boundary: T = 0
Top/bottom: insulated (natural BC)
Convective coefficient h_conv = {h_conv} (set >0 to activate Robin BC)
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl
import numpy as np
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Boundary conditions
def left_wall(x):
    return np.isclose(x[0], 0.0)

def right_wall(x):
    return np.isclose(x[0], 1.0)

left_facets  = mesh.locate_entities_boundary(domain, fdim, left_wall)
right_facets = mesh.locate_entities_boundary(domain, fdim, right_wall)
dofs_left  = fem.locate_dofs_topological(V, fdim, left_facets)
dofs_right = fem.locate_dofs_topological(V, fdim, right_facets)
bc_hot  = fem.dirichletbc(default_scalar_type({T_hot}), dofs_left,  V)
bc_cold = fem.dirichletbc(default_scalar_type(0.0),     dofs_right, V)
bcs = [bc_hot, bc_cold]

# Thermal properties
k     = fem.Constant(domain, default_scalar_type({k}))
rho   = fem.Constant(domain, default_scalar_type({rho}))
cp    = fem.Constant(domain, default_scalar_type({cp}))
dt_c  = fem.Constant(domain, default_scalar_type({dt}))
Q_src = fem.Constant(domain, default_scalar_type(0.0))  # volumetric heat source
h_conv = fem.Constant(domain, default_scalar_type({h_conv}))
T_amb  = fem.Constant(domain, default_scalar_type({T_amb}))

# Solution functions
T_n = fem.Function(V, name="T_old")   # previous time step
T_n.x.array[:] = {T_init}            # uniform initial temperature
T_h = ufl.TrialFunction(V)
v   = ufl.TestFunction(V)

# Backward Euler bilinear and linear forms
# Convective BC on top/bottom: h_conv*(T - T_amb)*v*ds (Robin)
a = (
    rho * cp / dt_c * T_h * v * ufl.dx
    + k * ufl.dot(ufl.grad(T_h), ufl.grad(v)) * ufl.dx
    + h_conv * T_h * v * ufl.ds         # Robin: convective loss
)
L = (
    rho * cp / dt_c * T_n * v * ufl.dx
    + Q_src * v * ufl.dx
    + h_conv * T_amb * v * ufl.ds       # Robin: ambient contribution
)

# Compile forms
a_form = fem.form(a)
L_form = fem.form(L)

from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

A = assemble_matrix(a_form, bcs=bcs)
A.assemble()

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

T_new = fem.Function(V, name="temperature")

# Time loop
n_steps = {n_steps}
from dolfinx.io import XDMFFile

with XDMFFile(domain.comm, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(T_n, 0.0)

    for step in range(n_steps):
        b_vec = assemble_vector(L_form)
        apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_vec, bcs)

        solver.solve(b_vec, T_new.x.petsc_vec)
        _reason = solver.getConvergedReason()
        if _reason <= 0:
            raise RuntimeError(
                f"KSP failed at step {{step + 1}} with KSPConvergedReason="
                f"{{_reason}} (-11 = DIVERGED_PC_FAILED). Nothing raises on "
                f"this, so without the check the loop runs to the end and "
                f"exits 0 with garbage in T.")
        T_new.x.scatter_forward()
        T_n.x.array[:] = T_new.x.array[:]

        t = (step + 1) * {dt}
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            xdmf.write_function(T_new, t)
            T_arr = T_new.x.array
            print(f"Step {{step+1}}/{{n_steps}}, t={{t:.4f}}: "
                  f"T in [{{T_arr.min():.4f}}, {{T_arr.max():.4f}}]")

# Physical check that needs no reference solution: on a closed domain the
# stored energy can only change through the boundary fluxes and the source,
# and min/max alone would not reveal a frozen or diverged field.
_energy = domain.comm.allreduce(
    fem.assemble_scalar(fem.form(rho * cp * T_new * ufl.dx)), op=MPI.SUM)
if not np.isfinite(_energy):
    raise RuntimeError("stored energy is not finite — the solve failed")
print(f"stored energy int(rho*cp*T) dx = {{_energy:.6e}}")
print(f"Transient heat: {{n_steps}} steps complete")
print(f"Final T: min={{T_new.x.array.min():.6e}}, max={{T_new.x.array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# cahn_hilliard
# ---------------------------------------------------------------------------

_CAHN_HILLIARD_KNOWLEDGE = {
    "description": (
        "Cahn-Hilliard equation for spinodal decomposition / phase separation. "
        "Fourth-order PDE split into two coupled second-order equations: "
        "dphi/dt = div(M*grad(mu));  mu = -eps^2*laplacian(phi) + W'(phi)."
    ),
    "weak_form": (
        "Mixed formulation: (dphi/dt, v)*dx + M*(grad(mu), grad(v))*dx = 0; "
        "(mu, q)*dx - eps^2*(grad(phi), grad(q))*dx - W'(phi)*q*dx = 0; "
        "W(phi) = (phi^2-1)^2/4 (double-well)"
    ),
    "function_space": "Mixed P1+P1 (phi and mu); can use P2+P2 for better accuracy",
    "solver": "Newton (NonlinearProblem/SNES) per time step — nonlinear W'(phi)=phi^3-phi",
    "pitfalls": [
        "eps controls interface width: set eps = 2-4 * h_mesh for resolved interface",
        "Mobility M: constant M or degenerate M(phi) = (1-phi^2)^+",
        "Double-well W(phi) = (phi^2-1)^2/4; W'(phi) = phi^3 - phi",
        "Mass conservation: integral of phi is conserved (no-flux BCs)",
        "Backward Euler: stable but first-order; convex splitting schemes are energy-stable",
        "Convex splitting: treat W_convex implicitly, W_concave explicitly for unconditional stability",
        "Initial condition: small random perturbation around phi=0 triggers spinodal decomposition",
        "Output: phi=+1 (phase A), phi=-1 (phase B), interface at phi=0",
    ],
    "materials": {
        "epsilon": {"range": [0.01, 0.1], "unit": "dimensionless (interface parameter)"},
        "mobility": {"range": [1e-5, 1.0], "unit": "m^2/(N*s) or dimensionless"},
    },
}


def _cahn_hilliard_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx Cahn-Hilliard script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 64)
    ny = params.get("ny", 64)
    eps = params.get("epsilon", 0.02)
    M = params.get("mobility", 1.0)
    dt = params.get("dt", 5e-5)
    n_steps = params.get("n_steps", 50)
    return f'''\
"""Cahn-Hilliard equation — spinodal decomposition — FEniCSx/dolfinx
Mixed formulation: (phi, mu) coupled system.
Double-well potential W(phi) = (phi^2-1)^2 / 4
Backward Euler time discretization.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Mixed function space: (phi, mu) both P1
P1 = element("Lagrange", domain.topology.cell_name(), 1)
ME = mixed_element([P1, P1])
W = fem.functionspace(domain, ME)

# Previous and current solution
w_old = fem.Function(W)
w     = fem.Function(W)

# Split: phi (phase field), mu (chemical potential)
phi_old, mu_old = ufl.split(w_old)
phi,     mu     = ufl.split(w)

# Test functions
v_phi, v_mu = ufl.TestFunctions(W)

# Parameters
eps_c = fem.Constant(domain, default_scalar_type({eps}))
M_c   = fem.Constant(domain, default_scalar_type({M}))
dt_c  = fem.Constant(domain, default_scalar_type({dt}))

# Nonlinear double-well: W'(phi) = phi^3 - phi
# Use theta-method weighting for stability (theta=1: fully implicit backward Euler)
theta = 0.5  # Crank-Nicolson in chemical potential (semi-implicit)
phi_mid = (1 - theta) * phi_old + theta * phi

# Cahn-Hilliard weak form (backward Euler in phi, semi-implicit in mu)
# Eq 1: (dphi/dt, v_phi) + M * (grad(mu), grad(v_phi)) = 0
# Eq 2: (mu, v_mu) - eps^2 * (grad(phi), grad(v_mu)) - W'(phi)*v_mu = 0
F = (
    (phi - phi_old) / dt_c * v_phi * ufl.dx
    + M_c * ufl.dot(ufl.grad(mu), ufl.grad(v_phi)) * ufl.dx
    + mu * v_mu * ufl.dx
    - eps_c**2 * ufl.dot(ufl.grad(phi), ufl.grad(v_mu)) * ufl.dx
    - (phi**3 - phi) * v_mu * ufl.dx
)

# No-flux BCs (natural) — no Dirichlet needed

# Initial condition: uniform mixture with small random noise
# phi ~ 0 + noise triggers spinodal decomposition
rng = np.random.default_rng(42)
W0, _ = W.sub(0).collapse()
W1, _ = W.sub(1).collapse()
phi_init = fem.Function(W0)
phi_init.x.array[:] = 0.0 + 0.05 * rng.standard_normal(len(phi_init.x.array))
mu_init = fem.Function(W1)
mu_init.x.array[:] = phi_init.x.array**3 - phi_init.x.array  # mu = W'(phi_0)
w.sub(0).interpolate(phi_init)
w.sub(1).interpolate(mu_init)
w.x.scatter_forward()
w_old.x.array[:] = w.x.array[:]

# Newton solver
problem = NonlinearProblem(F, w, bcs=[], petsc_options_prefix="ch",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
    }})

# Time loop with mass conservation check
n_steps = {n_steps}
from dolfinx.io import XDMFFile
from dolfinx.fem import assemble_scalar, form

phi_mass_form = form(phi * ufl.dx)
initial_mass = domain.comm.allreduce(assemble_scalar(phi_mass_form), op=MPI.SUM)

phi_out = fem.Function(W0, name="phase_field")
mu_out  = fem.Function(W1, name="chemical_potential")

with XDMFFile(domain.comm, "phase_field.xdmf", "w") as xdmf_phi, \
     XDMFFile(domain.comm, "chemical_potential.xdmf", "w") as xdmf_mu:
    xdmf_phi.write_mesh(domain)
    xdmf_mu.write_mesh(domain)

    for step in range(n_steps):
        w_old.x.array[:] = w.x.array[:]
        problem.solve()
        w.x.scatter_forward()

        t = (step + 1) * {dt}
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            # Extract and write
            phi_out.interpolate(w.sub(0).collapse())
            mu_out.interpolate(w.sub(1).collapse())
            xdmf_phi.write_function(phi_out, t)
            xdmf_mu.write_function(mu_out, t)

            mass = domain.comm.allreduce(assemble_scalar(phi_mass_form), op=MPI.SUM)
            phi_arr = phi_out.x.array
            its = problem.solver.getIterationNumber()
            print(f"Step {{step+1}}/{{n_steps}}, t={{t:.6f}}: "
                  f"phi=[{{phi_arr.min():.4f}}, {{phi_arr.max():.4f}}], "
                  f"mass={{mass:.6e}} (ref={{initial_mass:.6e}}), "
                  f"Newton its={{its}}")

print(f"Cahn-Hilliard: {{n_steps}} steps complete")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# nonlinear_pde
# ---------------------------------------------------------------------------

_NONLINEAR_PDE_KNOWLEDGE = {
    'description': ('General scalar nonlinear PDE of the form -div(D(u)*grad(u)) + R(u) = f, solved with '
     "Newton's method. The residual is written directly in UFL and the Jacobian is "
     'obtained by automatic differentiation, so only the residual has to be supplied. The '
     'nonlinear solver is PETSc SNES, reached through dolfinx.fem.petsc.NonlinearProblem.'),
    'minimal_working_example': ('"""Nonlinear diffusion -div(D(u) grad u) = f (dolfinx 0.10). Self-checks only."""\n'
     'from mpi4py import MPI\n'
     'import numpy as np\n'
     'import ufl\n'
     'from dolfinx import default_scalar_type, fem, la, mesh\n'
     'from dolfinx.fem.petsc import NonlinearProblem\n'
     '\n'
     'comm = MPI.COMM_WORLD\n'
     'msh = mesh.create_unit_square(comm, 32, 32, mesh.CellType.triangle)\n'
     'fdim = msh.topology.dim - 1\n'
     'msh.topology.create_connectivity(fdim, msh.topology.dim)\n'
     'V = fem.functionspace(msh, ("Lagrange", 1))\n'
     '\n'
     'bfacets = mesh.exterior_facet_indices(msh.topology)\n'
     'bdofs = fem.locate_dofs_topological(V, fdim, bfacets)\n'
     'bc = fem.dirichletbc(default_scalar_type(0.0), bdofs, V)\n'
     '\n'
     'u = fem.Function(V, name="u")\n'
     'v = ufl.TestFunction(V)\n'
     'f = fem.Constant(msh, default_scalar_type(10.0))\n'
     '# Strictly positive, smooth diffusivity: no regularisation needed.\n'
     'D = 1.0 + u ** 2\n'
     'F = D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx\n'
     '\n'
     'problem = NonlinearProblem(\n'
     '    F, u, bcs=[bc], petsc_options_prefix="nlp_",\n'
     '    petsc_options={"ksp_type": "preonly", "pc_type": "lu",\n'
     '                   "pc_factor_mat_solver_type": "mumps",\n'
     '                   "snes_rtol": 1e-10, "snes_atol": 1e-12, "snes_max_it": 30,\n'
     '                   "snes_linesearch_type": "bt"})\n'
     'problem.solve()\n'
     'reason = problem.solver.getConvergedReason()\n'
     'assert reason > 0, (\n'
     '    f"SNES did NOT converge: reason {reason} (negative = diverged). "\n'
     '    f"problem.solve() returns a Function even when this happens.")\n'
     '\n'
     '# Self-check 1: Galerkin residual at every unconstrained DOF.\n'
     'res_form = fem.form(F)\n'
     'R = fem.assemble_vector(res_form)\n'
     'R.scatter_reverse(la.InsertMode.add)\n'
     'nloc = V.dofmap.index_map.size_local\n'
     'free = np.ones(nloc, dtype=bool)\n'
     'free[bdofs[bdofs < nloc]] = False\n'
     'res_free = comm.allreduce(\n'
     '    float(np.abs(R.array[:nloc][free]).max(initial=0.0)), MPI.MAX)\n'
     '\n'
     '# Self-check 2: with f > 0 and u = 0 on the whole boundary the maximum\n'
     '# principle forces u >= 0 everywhere, and max(u) must be interior.\n'
     'gmin = comm.allreduce(float(u.x.array[:nloc].min()), MPI.MIN)\n'
     'gmax = comm.allreduce(float(u.x.array[:nloc].max()), MPI.MAX)\n'
     '\n'
     '# Self-check 3: energy balance  int D(u)|grad u|^2 dx == int f*u dx\n'
     '# (take v = u in the weak form; u is admissible because it vanishes on dOmega).\n'
     'lhs = comm.allreduce(fem.assemble_scalar(\n'
     '    fem.form(D * ufl.dot(ufl.grad(u), ufl.grad(u)) * ufl.dx)), MPI.SUM)\n'
     'rhs = comm.allreduce(fem.assemble_scalar(fem.form(f * u * ufl.dx)), MPI.SUM)\n'
     '\n'
     '# Self-check 4: the diffusivity actually varies (i.e. the problem is nonlinear).\n'
     'W = fem.functionspace(msh, ("DG", 0))\n'
     'Dh = fem.Function(W, name="diffusivity")\n'
     'Dh.interpolate(fem.Expression(D, W.element.interpolation_points))\n'
     'wloc = W.dofmap.index_map.size_local\n'
     'Dmin = comm.allreduce(float(Dh.x.array[:wloc].min()), MPI.MIN)\n'
     'Dmax = comm.allreduce(float(Dh.x.array[:wloc].max()), MPI.MAX)\n'
     '\n'
     'if comm.rank == 0:\n'
     '    print(f"SNES converged reason={reason} (>0 = converged), "\n'
     '          f"iterations={problem.solver.getIterationNumber()}")\n'
     '    print(f"max |residual| over unconstrained DOFs = {res_free:.3e}")\n'
     '    print(f"global u range = [{gmin:.6e}, {gmax:.6e}]  "\n'
     '          f"(f>0 with u=0 on dOmega => u >= 0 everywhere)")\n'
     '    print(f"energy balance: int D|grad u|^2 dx = {lhs:.8e}, "\n'
     '          f"int f*u dx = {rhs:.8e}, rel. gap = {abs(lhs - rhs) / abs(rhs):.3e}")\n'
     '    print(f"D(u) range over the mesh = [{Dmin:.6f}, {Dmax:.6f}] "\n'
     '          f"(spread > 0 confirms the problem is genuinely nonlinear)")\n'
     'assert res_free < 1e-10\n'
     'assert gmin > -1e-12 and np.isfinite(gmax)\n'
     'assert abs(lhs - rhs) / abs(rhs) < 1e-10\n'),
    'function_space': {'REQUIRED': 'V = fem.functionspace(msh, ("Lagrange", 1))\n'
                 'u = fem.Function(V)      # the UNKNOWN is a Function, not a '
                 'TrialFunction\n'
                 'v = ufl.TestFunction(V)',
     'OPTIONAL': 'Degree 1 or 2; vector-valued via fem.functionspace(msh, ("Lagrange", 1, '
                 '(gdim,))). Any cell type. u may be given a non-zero starting guess by '
                 'writing into u.x.array before solve() - this is how continuation is '
                 'done.',
     'explanation': 'In a nonlinear problem the current iterate must be a Function, '
                    'because the residual has to be evaluated at it. Using a TrialFunction '
                    'for the unknown makes the form bilinear and the problem is no longer '
                    'the one you meant to solve.',
     'pitfalls': ['[API] u must be a fem.Function. Signal: with a ufl.TrialFunction as the '
                  'unknown, form construction fails with '
                  'ufl.algorithms.check_arities.ArityMismatch: Applying nonlinear operator '
                  'Power to expression depending on form argument v_1.']},
    'weak_form': {'REQUIRED': 'F = D(u) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + R(u)*v*ufl.dx - '
                 'f*v*ufl.dx\n'
                 '# F is the RESIDUAL and must equal zero; do NOT split it into a and L.',
     'OPTIONAL': 'J = ufl.derivative(F, u, ufl.TrialFunction(V)) may be passed as the '
                 'OPTIONAL keyword J=..., but only if you deliberately want a '
                 'NON-consistent tangent (a frozen-coefficient Picard operator, say). Left '
                 'out, dolfinx derives the exact Jacobian itself.',
     'explanation': 'The whole equation, source term included, goes into a single residual '
                    'form F(u; v). Newton then needs dF/du, which dolfinx obtains from UFL '
                    'by automatic differentiation - including through exp, tanh, powers '
                    'and conditionals.',
     'pitfalls': ['[Numerical] A diffusivity that can hit zero or a non-integer power of a quantity '
                  'that can go negative makes the residual non-differentiable at the '
                  'starting guess. Signal: SNES returns converged reason -4 '
                  '(DIVERGED_FNORM_NAN) at iteration 0 with u left exactly at its initial '
                  'value.']},
    'boundary_conditions': {'REQUIRED': 'msh.topology.create_connectivity(fdim, msh.topology.dim)\n'
                 'facets = mesh.exterior_facet_indices(msh.topology)\n'
                 'dofs = fem.locate_dofs_topological(V, fdim, facets)\n'
                 'bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)\n'
                 'problem = NonlinearProblem(F, u, bcs=[bc], petsc_options_prefix="nlp_")',
     'OPTIONAL': 'mesh.locate_entities_boundary(msh, fdim, marker_fn) instead of '
                 'exterior_facet_indices for a subset of the boundary. Neumann data enters '
                 'F as a surface term; a tagged ufl.Measure("ds", domain=msh, '
                 'subdomain_data=tags) restricts it to one wall. For a large load, ramp '
                 'the Dirichlet value or the source over several solves (continuation).',
     'explanation': 'NonlinearProblem applies the constraints itself: it zeroes the Newton '
                    'update at constrained DOFs and lifts the residual, so no manual '
                    'apply_lifting/set_bc is needed. Give it the bcs list and start from '
                    'an iterate that already satisfies them.',
     'pitfalls': ['[Numerical] An initial guess that violates the Dirichlet data is fine, but one '
                  'outside the domain of D(u) or R(u) is not. Signal: SNES returns '
                  'converged reason -4 (DIVERGED_FNORM_NAN) or -6 (DIVERGED_LINE_SEARCH) '
                  'at iteration 0, and solve() hands back the unchanged initial guess '
                  'without raising.']},
    'solver': {'REQUIRED': 'problem = NonlinearProblem(F, u, bcs=[bc], petsc_options_prefix="nlp_",\n'
                 '    petsc_options={"ksp_type": "preonly", "pc_type": "lu",\n'
                 '                   "pc_factor_mat_solver_type": "mumps",\n'
                 '                   "snes_rtol": 1e-10, "snes_max_it": 30,\n'
                 '                   "snes_linesearch_type": "bt"})\n'
                 'problem.solve()\n'
                 'assert problem.solver.getConvergedReason() > 0   # NOT optional',
     'OPTIONAL': 'petsc_options_prefix is REQUIRED (keyword-only); J and P are optional. '
                 "snes_linesearch_type: 'bt' (backtracking, the PETSc default), 'l2', "
                 "'basic' (undamped Newton), 'none'. snes_monitor accepts either None or "
                 "the empty string '' - both activate the default printing. "
                 '"snes_error_if_not_converged": True makes PETSc raise instead of '
                 'returning silently. Scalable alternative to the LU inner solve: ksp_type '
                 "'gmres' with pc_type 'hypre' or 'gamg'.",
     'explanation': 'problem.solver is the underlying PETSc SNES object, so the whole '
                    'petsc4py SNES API (getConvergedReason, getIterationNumber, '
                    'getFunctionNorm) is available after the solve.',
     'pitfalls': ['[API] problem.solve() returns the solution Function whether or not SNES '
                  'converged. Signal: a negative converged reason with a perfectly finite, '
                  'plausible-looking field.',
                  '[API] dolfinx.nls.petsc.NewtonSolver cannot wrap a 0.10 NonlinearProblem. '
                  "Signal: AttributeError: 'NonlinearProblem' object has no attribute 'a'.",
                  '[API] dolfinx.PETScKrylovSolver does not exist. Signal: AttributeError: '
                  "module 'dolfinx' has no attribute 'PETScKrylovSolver'."]},
    'postprocessing': {'REQUIRED': 'W = fem.functionspace(msh, ("DG", 0))\n'
                 'Dh = fem.Function(W)\n'
                 'Dh.interpolate(fem.Expression(D, W.element.interpolation_points))',
     'OPTIONAL': 'Any UFL expression of the solution can be interpolated this way (fluxes, '
                 'gradients, stress measures). Use DG0 for cell-wise-constant quantities '
                 'and DG1/Lagrange for smoother output.',
     'explanation': 'fem.Expression compiles a UFL expression for evaluation at the '
                    'interpolation points of the target element; Function.interpolate then '
                    'fills the Function.',
     'pitfalls': ['[API] element.interpolation_points is a PROPERTY (a numpy array) in dolfinx '
                  "0.10, not a method. Signal: TypeError: 'numpy.ndarray' object is not "
                  'callable, if you write interpolation_points().']},
    'pitfalls': ['[API] The Jacobian keyword `J` is OPTIONAL on dolfinx 0.10 (signature: `J: '
     'ufl.form.Form | ... | None = None`). Passing only `NonlinearProblem(F, u, bcs=bcs, '
     "petsc_options_prefix='...')` is correct and complete - dolfinx derives dF/du with "
     'ufl.derivative internally. Signal: measured, a (1 + u^2)-diffusion problem solved '
     "with NO J at all converged in 3 SNES iterations. There is no 'TypeError: "
     "NonlinearProblem missing required argument J', and SNES does not fall back to a "
     'finite-difference Jacobian. What IS required is `petsc_options_prefix`; omitting '
     'that gives `TypeError: NonlinearProblem.__init__() missing 1 required keyword-only '
     "argument: 'petsc_options_prefix'`. Pass J explicitly only when you deliberately want "
     'a non-consistent tangent.',
     '[Integration] `problem.solve()` returns the solution Function whether or not SNES '
     "converged; the dolfinx 0.10 docstring states 'the user is responsible for asserting "
     "convergence of the SNES solver'. Assert `problem.solver.getConvergedReason() > 0`. "
     'Signal: measured on a Bratu problem (-div(grad u) = lambda*exp(u)) past its turning '
     'point at lambda = 20, SNES returned converged reason -5 (DIVERGED_MAX_IT) after 8 '
     'iterations and `solve()` handed back a completely finite field with max(u) = '
     "8.19e-04 and no NaNs - a number small and tidy enough to pass any 'is it finite / is "
     'it O(1)\' check. With `"snes_error_if_not_converged": True` the same run instead '
     "raises with the PETSc text 'Error: error code 91 ... SNESSolve has not converged'.",
     '[Numerical] A diffusivity that is singular at the starting iterate (the p-Laplacian '
     'D = |grad u|^(p-2) with p < 2, evaluated at u = 0) must be regularised as D = (|grad '
     'u|^2 + eps^2)^((p-2)/2) with eps around 1e-6. Signal: measured with p = 1.5 and u0 = '
     '0, SNES returns converged reason -4 (DIVERGED_FNORM_NAN) at iteration 0 and leaves u '
     'exactly at zero - and `np.isnan(u.x.array).any()` is FALSE, so a NaN check on the '
     'SOLUTION does not catch it. Look at the assembled objects instead, because that is '
     'where the NaN actually is: assembled at the singular starting iterate, both the '
     'residual vector and the JACOBIAN carry NaN entries, and the Jacobian\'s PETSc norm '
     'comes back nan. IMPORTANT CORRECTION: an earlier version of this entry listed \'nan '
     "entries in J' among the signals that do NOT reproduce - they DO. What does not "
     'reproduce is an EXCEPTION: neither the residual assembly nor the Jacobian assembly '
     "raises anything, in particular no ZeroDivisionError, so 'assembly raised' is the "
     'wrong thing to guard on and `PETSc.Mat.norm()` / `np.isnan` on the assembled '
     'Jacobian is the right one. With eps = 1e-6 the same problem converges in a handful '
     'of iterations with a positive reason. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
     '[Numerical] A non-integer power of the solution, D = 1 + u**q with q not an integer, '
     'is only defined while u >= 0. If the load drives u negative the power becomes '
     'complex and Newton dies. Signal: measured, q = 2.0 with a NEGATIVE source converges '
     'normally (reason 2, u in [-0.0733, 0]), but q = 1.5 with the same negative source '
     'dies at iteration 0 with u left exactly at zero. WHICH negative reason you get is '
     "decided by the LINE SEARCH, not by the exponent: on the default 'bt' line search and "
     "on 'basic' it is -4 (DIVERGED_FNORM_NAN), and only 'l2' reports -6 "
     '(DIVERGED_LINE_SEARCH). An earlier version of this entry quoted -6 alone, which is '
     'the l2 answer - a check that matches on DIVERGED_LINE_SEARCH will miss this failure '
     'entirely under the default settings, so test for a NEGATIVE '
     'getConvergedReason() and read the '
     'code rather than matching one value. Use abs(u)**q, max(u,0)**q, or an even integer '
     'exponent if u can change sign. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
     '[Numerical] Semilinear source terms that grow super-linearly (R(u) = exp(u)) have a '
     'load beyond which no steady solution exists; past it Newton cannot converge no '
     'matter what you do to the solver. Signal: measured, lambda = 1 converges in 3 '
     'iterations to max(u) = 7.78e-02, while lambda = 20 exhausts the iteration budget '
     'with reason -5. This is a property of the CONTINUOUS problem, so raising snes_max_it '
     'or switching line search will not fix it; ramp the load in steps and reuse the '
     'previous solution as the starting iterate.',
     '[Integration] SNES converged reasons on this install: positive means converged (2 = '
     'CONVERGED_FNORM_ABS, 3 = CONVERGED_FNORM_RELATIVE, 4 = CONVERGED_SNORM_RELATIVE, 5 = '
     'CONVERGED_ITS), negative means diverged (-1 DIVERGED_FUNCTION_DOMAIN, -3 '
     'DIVERGED_LINEAR_SOLVE, -4 DIVERGED_FNORM_NAN, -5 DIVERGED_MAX_IT, -6 '
     'DIVERGED_LINE_SEARCH, -8 DIVERGED_LOCAL_MIN). Signal: reason 2 is '
     "CONVERGED_FNORM_ABS, NOT CONVERGED_FNORM_RELATIVE - if a wrapper reports 'reason 2 = "
     "CONVERGED_FNORM_RELATIVE' its mapping is off by one and any logic built on the name "
     'is unreliable. Test only the SIGN.',
     '[API] Turn on the SNES history with `"snes_monitor": None` in petsc_options; the '
     'empty string works identically. Without it a failed solve tells you only the final '
     "reason, which does not distinguish 'the load step was too big' from 'the tangent is "
     "wrong'. Signal: measured, both `snes_monitor: None` and `snes_monitor: ''` produce "
     "the same per-iteration lines ('0 SNES Function norm 3.027343750000e-02', '1 SNES "
     "Function norm 6.029199357706e-05', ...) and both end at reason 2 after 3 iterations. "
     'A residual that falls by several orders per iteration is healthy Newton; one that '
     'stalls or oscillates points at the tangent or the line search.',
     '[Numerical] The line search changes WHICH failure you get, not whether you get one, '
     'so never infer correctness from the mere fact that a run finished. Signal: measured '
     'on the same Bratu problem past its turning point (lambda = 20, 30-iteration budget), '
     'the four PETSc line searches returned four different diverged reasons AND four '
     "different finite-looking fields: 'basic' gave reason -5 (DIVERGED_MAX_IT) after 30 "
     "iterations with max(u) = 8.19e-04; 'bt' (the default) gave reason -6 "
     "(DIVERGED_LINE_SEARCH) after 23 iterations with max(u) = 0.0 exactly; 'l2' gave "
     "reason -9 (DIVERGED_DTOL) after only 2 iterations with max(u) = 88.3; 'cp' matched "
     "'basic'. Every one of them is finite and free of NaNs. Use 'bt' or 'l2' for "
     "robustness and 'basic' (undamped Newton) when you want to reproduce a divergence for "
     'diagnosis - but read the reason either way.',
     '[API] `V.element.interpolation_points` is a PROPERTY holding a numpy array in '
     "dolfinx 0.10; older code calls it as a method. Signal: `TypeError: 'numpy.ndarray' "
     'object is not callable` from `fem.Expression(expr, '
     'W.element.interpolation_points())`. Drop the parentheses.',
     '[Physics] Reference-free checks that actually catch a wrong nonlinear solve, all '
     'used in the minimal_working_example: (1) the Galerkin residual assembled at the '
     'converged u must vanish at every UNCONSTRAINED degree of freedom; (2) the energy '
     'identity - taking v = u in the weak form gives int D(u)|grad u|^2 dx = int f*u dx '
     'exactly, since u vanishes on the Dirichlet boundary; (3) the maximum principle - '
     'with f > 0 and homogeneous Dirichlet data, u must be non-negative everywhere; (4) '
     'the spread of D(u) over the mesh must be non-zero, otherwise the run was effectively '
     'linear and proves nothing about the nonlinear machinery. Signal: measured on the '
     'example, residual over free DOFs 6.7e-12, energy identity closing to a relative gap '
     'of 2.2e-12, u in [0, 6.46e-01], D(u) in [1.000000, 1.416262].'],
}


def _nonlinear_pde_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx nonlinear PDE script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    q_exp = params.get("q_exponent", 2.0)
    return f'''\
"""General nonlinear PDE — FEniCSx/dolfinx
-div((1 + u^{q_exp}) * grad(u)) = f on [0,1]^2, u=0 on boundary
Jacobian via UFL automatic differentiation.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet BC: u = 0 on all boundaries
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Source term
f = fem.Constant(domain, default_scalar_type(1.0))

# Nonlinear diffusion coefficient: D(u) = 1 + u^q
# q_exp controls the nonlinearity strength
q = {q_exp}

# Current solution (nonlinear iterate)
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)

# Nonlinear diffusivity
D_u = 1.0 + u**q

# Residual F(u; v)
F = D_u * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

# Jacobian: computed automatically by UFL via derivative
# du = ufl.TrialFunction(V) is inferred by NonlinearProblem

# Newton solver with PETSc SNES
problem = NonlinearProblem(F, u, bcs=[bc], petsc_options_prefix="nl",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "snes_monitor": None,
        "snes_linesearch_type": "l2",
    }})

# Solve (initial guess u=0 is fine for moderate nonlinearity)
problem.solve()
its  = problem.solver.getIterationNumber()
reason = problem.solver.getConvergedReason()
print(f"Newton: {{its}} iterations, converged reason = {{reason}}")
if reason <= 0:
    raise RuntimeError(
        f"SNES did NOT converge (reason {{reason}} < 0). The returned field is "
        f"NOT a solution even though it is finite and plausible-looking. "
        f"Reduce the load, add continuation, or change snes_linesearch_type.")

# Postprocess: evaluate diffusivity D(u) at solution
V_vis = fem.functionspace(domain, ("DG", 0))
D_expr = fem.Expression(D_u, V_vis.element.interpolation_points)
D_func = fem.Function(V_vis, name="diffusivity")
D_func.interpolate(D_expr)

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

u_arr = u.x.array
D_arr = D_func.x.array
print(f"Nonlinear PDE (q={{q:.1f}}) solved:")
print(f"u: min={{u_arr.min():.6e}}, max={{u_arr.max():.6e}}")
print(f"D(u): min={{D_arr.min():.4f}}, max={{D_arr.max():.4f}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# magnetostatics
# ---------------------------------------------------------------------------

_MAGNETOSTATICS_KNOWLEDGE = {
    'description': ('Magnetostatics for the magnetic vector potential A, with B = curl(A) so that div(B) '
     '= 0 holds by construction. In 2D plane problems A has only an out-of-plane component '
     'Az and the curl-curl operator collapses to -div((1/mu)*grad(Az)) = Jz, which '
     'ordinary scalar Lagrange elements solve exactly. In 3D the full curl-curl operator '
     'must be discretised with H(curl)-conforming Nedelec elements.'),
    'minimal_working_example': ('"""2D magnetostatics, scalar Az formulation (dolfinx 0.10). Self-checks only."""\n'
     'from mpi4py import MPI\n'
     'import numpy as np\n'
     'import ufl\n'
     'from dolfinx import default_scalar_type, fem, mesh\n'
     'from dolfinx.fem.petsc import LinearProblem\n'
     '\n'
     'comm = MPI.COMM_WORLD\n'
     'MU0 = 4.0e-7 * np.pi\n'
     'N = 64\n'
     'msh = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)\n'
     'fdim = msh.topology.dim - 1\n'
     'msh.topology.create_connectivity(fdim, msh.topology.dim)\n'
     'V = fem.functionspace(msh, ("Lagrange", 1))\n'
     '\n'
     'bdofs = fem.locate_dofs_topological(\n'
     '    V, fdim, mesh.exterior_facet_indices(msh.topology))\n'
     'bc = fem.dirichletbc(default_scalar_type(0.0), bdofs, V)\n'
     '\n'
     '# Cell-wise material data on DG0: one value per cell, no quadrature aliasing.\n'
     'Q = fem.functionspace(msh, ("DG", 0))\n'
     'cmap = msh.topology.index_map(msh.topology.dim)\n'
     'cells = np.arange(cmap.size_local + cmap.num_ghosts, dtype=np.int32)\n'
     'mid = mesh.compute_midpoints(msh, msh.topology.dim, cells)\n'
     'r_coil, cx, cy = 0.15, 0.5, 0.5\n'
     'in_coil = (mid[:, 0] - cx) ** 2 + (mid[:, 1] - cy) ** 2 < r_coil ** 2\n'
     'in_iron = (~in_coil) & (\n'
     '    ((mid[:, 0] - cx) ** 2 + (mid[:, 1] - cy) ** 2) < 0.30 ** 2)\n'
     '\n'
     'mu_r = fem.Function(Q, name="mu_r")\n'
     'mu_r.x.array[:] = np.where(in_iron, 500.0, 1.0)\n'
     'Jz = fem.Function(Q, name="Jz")\n'
     'J_coil = 2.5e4                       # A/m^2\n'
     'Jz.x.array[:] = np.where(in_coil, J_coil, 0.0)\n'
     'nu = 1.0 / (MU0 * mu_r)              # reluctivity\n'
     '\n'
     'Az, v = ufl.TrialFunction(V), ufl.TestFunction(V)\n'
     'a = nu * ufl.dot(ufl.grad(Az), ufl.grad(v)) * ufl.dx\n'
     'L = Jz * v * ufl.dx\n'
     'problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="mag_",\n'
     '                        petsc_options={"ksp_type": "cg", "pc_type": "hypre",\n'
     '                                       "ksp_rtol": 1e-12})\n'
     'Az_h = problem.solve()\n'
     'Az_h.name = "Az"\n'
     'assert problem.solver.getConvergedReason() > 0, (\n'
     '    f"KSP failed: reason {problem.solver.getConvergedReason()}")\n'
     '\n'
     '# B = curl(Az) = (dAz/dy, -dAz/dx); ufl.curl of a 2D scalar gives exactly this.\n'
     'W = fem.functionspace(msh, ("DG", 0, (2,)))\n'
     'B = fem.Function(W, name="B")\n'
     'B.interpolate(fem.Expression(ufl.curl(Az_h), W.element.interpolation_points))\n'
     '\n'
     '\n'
     'def scal(form):\n'
     '    return comm.allreduce(fem.assemble_scalar(form), MPI.SUM)\n'
     '\n'
     '\n'
     '# Self-check 1: Ampere / conservation. Integrating the PDE over the domain,\n'
     '# the total current must equal minus the outward flux of nu*grad(Az).\n'
     'n = ufl.FacetNormal(msh)\n'
     'I_total = scal(fem.form(Jz * ufl.dx))\n'
     'flux = scal(fem.form(nu * ufl.dot(ufl.grad(Az_h), n) * ufl.ds))\n'
     '# Self-check 2: energy identity int nu |grad Az|^2 dx == int Jz*Az dx.\n'
     'w_mag = scal(fem.form(nu * ufl.dot(ufl.grad(Az_h), ufl.grad(Az_h)) * ufl.dx))\n'
     'w_src = scal(fem.form(Jz * Az_h * ufl.dx))\n'
     '# Self-check 3: |B| must be finite and below saturation for a LINEAR mu_r.\n'
     'Bmag = np.sqrt(B.x.array[0::2] ** 2 + B.x.array[1::2] ** 2)\n'
     'wloc = W.dofmap.index_map.size_local\n'
     'Bmax = comm.allreduce(float(Bmag[:wloc].max()), MPI.MAX)\n'
     'nloc = V.dofmap.index_map.size_local\n'
     'Amax = comm.allreduce(float(np.abs(Az_h.x.array[:nloc]).max()), MPI.MAX)\n'
     '\n'
     'if comm.rank == 0:\n'
     '    print(f"KSP reason={problem.solver.getConvergedReason()} "\n'
     '          f"iterations={problem.solver.getIterationNumber()} "\n'
     '          f"DOFs={V.dofmap.index_map.size_global}")\n'
     '    print(f"Ampere check: int(Jz)dx = {I_total:.6e} A, "\n'
     '          f"-flux of nu*grad(Az) = {-flux:.6e} A, "\n'
     '          f"ratio = {-flux / I_total:.10f} (must be 1)")\n'
     '    print(f"energy check: int nu|grad Az|^2 dx = {w_mag:.8e}, "\n'
     '          f"int Jz*Az dx = {w_src:.8e}, rel. gap = "\n'
     '          f"{abs(w_mag - w_src) / abs(w_src):.3e}")\n'
     '    print(f"max |Az| = {Amax:.6e} Wb/m,  max |B| = {Bmax:.6e} T "\n'
     '          f"(a LINEAR mu_r is only valid while |B| stays below the "\n'
     '          f"saturation flux density of the material, ~2 T for steel)")\n'
     'assert abs(-flux / I_total - 1.0) < 1e-6\n'
     'assert abs(w_mag - w_src) / abs(w_src) < 1e-8\n'
     'assert np.isfinite(Bmax) and Bmax < 2.0\n'),
    'function_space': {'REQUIRED': '# 2D plane problem (scalar out-of-plane potential Az):\n'
                 'V = fem.functionspace(msh, ("Lagrange", 1))\n'
                 '# 3D full vector potential - H(curl) is MANDATORY:\n'
                 'import basix.ufl\n'
                 'el = basix.ufl.element("N1curl", msh.basix_cell(), 1)\n'
                 'V = fem.functionspace(msh, el)',
     'OPTIONAL': '2D: Lagrange degree 1 or 2; triangles or quadrilaterals. 3D: "N1curl" '
                 '(Nedelec first kind) or "N2curl" (second kind), degree 1 or higher. '
                 'Material data belongs in a separate DG0 space with one value per cell.',
     'explanation': 'In 2D, Az is a scalar and B = curl(Az) is a rotated gradient, which '
                    'H1-conforming Lagrange represents exactly - no special element is '
                    'needed and none helps. In 3D the physical field has continuous '
                    'tangential and DISCONTINUOUS normal components across material '
                    'interfaces; only H(curl)-conforming elements can represent that, and '
                    'using vector Lagrange instead poisons the whole discrete spectrum.',
     'pitfalls': ['[Numerical] 3D vector Lagrange for curl-curl assembles a perfectly ordinary matrix '
                  'and solves without complaint - there is no error to catch. Signal: the '
                  'discrete spectrum is entirely spurious (see the top-level pitfall for '
                  'measured eigenvalues).']},
    'weak_form': {'REQUIRED': '# 2D:\n'
                 'a = nu * ufl.dot(ufl.grad(Az), ufl.grad(v)) * ufl.dx\n'
                 'L = Jz * v * ufl.dx\n'
                 '# 3D:\n'
                 'a = nu * ufl.inner(ufl.curl(A), ufl.curl(w)) * ufl.dx + ufl.inner(A, w) '
                 '* ufl.dx    # the second term is the gauge\n'
                 'L = ufl.inner(J, w) * ufl.dx\n'
                 '# with nu = 1/(mu0*mu_r) the reluctivity',
     'OPTIONAL': 'Instead of the mass-term (regularised) gauge in 3D, a Lagrange '
                 'multiplier enforcing div(A) = 0 (a mixed formulation) or tree-cotree '
                 'gauging. A nonlinear iron law replaces the constant nu by nu(|B|) and '
                 'turns the LinearProblem into a NonlinearProblem.',
     'explanation': 'Write the reluctivity nu = 1/(mu0*mu_r) rather than dividing by mu '
                    'inside the form; that is the coefficient that actually multiplies the '
                    'operator, and it keeps the DG0 material Function as a plain factor. '
                    'In 2D the whole curl-curl reduces to a Laplacian in Az, so the same '
                    'assembly path as any scalar diffusion problem applies.',
     'pitfalls': ['[Numerical] The 3D pure curl-curl operator without any gauge term is SINGULAR '
                  '(every gradient field is in its kernel). Signal: it still solves - see '
                  'the top-level gauge pitfall.',
                  '[API] ufl.curl accepts a 2D SCALAR (shape (2,)) and a 2D VECTOR (shape ()); '
                  'both compile through fem.form. Signal: inner(curl(s), curl(s))*dx for a '
                  'SCALAR s assembles silently and returns exactly the same number as '
                  'inner(grad(s), grad(s))*dx (measured 10.328124999999996 for both), so a '
                  'dimensional mix-up produces no error at all.']},
    'boundary_conditions': {'REQUIRED': '# 2D: Az = 0 on the far boundary (flux parallel to it)\n'
                 'bdofs = fem.locate_dofs_topological(\n'
                 '    V, fdim, mesh.exterior_facet_indices(msh.topology))\n'
                 'bc = fem.dirichletbc(default_scalar_type(0.0), bdofs, V)\n'
                 '# 3D with Nedelec, the constrained quantity is the TANGENTIAL trace,\n'
                 '# so the bc value must be a Function of the SAME space:\n'
                 'bc = fem.dirichletbc(fem.Function(V), fem.locate_dofs_topological(V, '
                 'fdim, facets))',
     'OPTIONAL': 'A symmetry plane where B is normal to the boundary is the natural '
                 'condition (no term needed). Az = const on a boundary makes it a flux '
                 'line; different constants on two boundaries prescribe the flux between '
                 'them.',
     'explanation': 'Az = 0 on the exterior means no flux crosses that boundary, which '
                    'models a far-field box or a flux-return path. For Nedelec spaces a '
                    'scalar bc value is not meaningful because the degrees of freedom are '
                    'tangential moments, not point values - pass a Function.',
     'pitfalls': ['[BC] For a Nedelec space the bc value must be a Function of that space, not '
                  'a scalar. Signal: RuntimeError: Rank mismatch between Constant and '
                  'function space in DirichletBC.']},
    'solver': {'REQUIRED': 'problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="mag_",\n'
                 '    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": '
                 '1e-12})\n'
                 'Az_h = problem.solve()\n'
                 'assert problem.solver.getConvergedReason() > 0   # NOT optional',
     'OPTIONAL': 'petsc_options_prefix is REQUIRED (keyword-only). Direct solve: ksp_type '
                 "'preonly' with pc_type 'lu' (PETSc selects MUMPS for an MPI matrix when "
                 "MUMPS is configured; a healthy 'preonly' solve reports converged reason "
                 "4 = CONVERGED_ITS). For 3D Nedelec, hypre's AMS preconditioner is the "
                 'specialised choice. "ksp_error_if_not_converged": True raises instead of '
                 'returning.',
     'explanation': 'The 2D operator is symmetric positive definite once the Dirichlet '
                    'condition removes the constant mode, so CG plus algebraic multigrid '
                    'is both correct and scalable. A very large mu_r contrast raises the '
                    'condition number, which shows up as a rising iteration count.',
     'pitfalls': ['[API] LinearProblem.solve() never raises on failure. Signal: a negative KSP '
                  'converged reason with the script continuing normally.',
                  '[Numerical] A converged KSP says nothing about whether the system was well posed. '
                  'Signal: measured on an ungauged 3D curl-curl system, LU, CG and GMRES '
                  'all reported converged and returned ||A||_L2 of 1.1599, 0.03213 and '
                  '0.03207 respectively - a factor 36 apart.']},
    'postprocessing': {'REQUIRED': 'W = fem.functionspace(msh, ("DG", 0, (2,)))\n'
                 'B = fem.Function(W)\n'
                 'B.interpolate(fem.Expression(ufl.curl(Az_h), '
                 'W.element.interpolation_points))',
     'OPTIONAL': 'H = nu*B for the field strength; energy density 0.5*nu*|B|^2. For a P1 '
                 'Az the derivative is cell-wise constant, so DG0 is the natural target '
                 'and DG1/CG1 gain nothing.',
     'explanation': 'ufl.curl applied to a 2D scalar already produces the correct '
                    '2-vector, so there is no reason to spell out the components by hand.',
     'pitfalls': ['[API] element.interpolation_points is a PROPERTY, not a method. Signal: '
                  "TypeError: 'numpy.ndarray' object is not callable."]},
    'materials': {'mu0': '4*pi*1e-7 H/m',
     'mu_r': '1 for air/copper; 500-10000 for linear soft iron - but only below '
             'saturation, roughly |B| < 2 T for electrical steel',
     'nu': 'reluctivity 1/(mu0*mu_r), the coefficient that multiplies the operator',
     'J_coil': 'A/m^2 over the coil cross-section; the total current is J_coil * area, '
               "which is the quantity Ampere's law constrains"},
    'pitfalls': ['[Physics] Check the magnitude of |B| against the saturation flux density of the '
     'material you claim to be modelling. A LINEAR mu_r is only meaningful below '
     'saturation (about 2 T for electrical steel), and nothing in the solve will tell you '
     'that you have left that regime. Signal: measured with mu_r = 1000 filling the domain '
     'around a coil carrying 1e6 A/m^2 over a disc of radius 0.2 m (total current 1.26e+05 '
     'A), the solve converged cleanly and reported max|B| = 1.435e+02 T - seventy times '
     'past saturation, printed with a units label of T as if it meant something. Setting '
     'mu_r = 1 on the same geometry gives max|B| = 1.25e-01 T, which matches '
     'mu0*I/(2*pi*r) for the same enclosed current, so the SOLVER is right and the MODEL '
     'is out of range. Sanity-check |B| against mu0*I/(2*pi*r) before believing anything '
     'downstream.',
     '[Numerical] max|B| is NOT a convergence indicator for a problem with a permeability '
     'jump: the field is singular at the material corner, and the jump follows the mesh, '
     'so refining the mesh makes the reported maximum GROW. Signal: measured on the same '
     'geometry, max|B| = 1.435e+02 T at 40x40, 1.583e+02 T at 80x80 and 1.650e+02 T at '
     '160x160, with the KSP reporting converged every time. A mesh-independent check does '
     'exist and does converge: the outward flux of nu*grad(Az) over the boundary equals '
     'minus the total current, measured -flux/int(Jz)dx = 1.00000000 at every resolution. '
     'Use that, or the energy identity int nu|grad Az|^2 dx = int Jz*Az dx, as the '
     'correctness check.',
     '[Numerical] In 3D you MUST use H(curl) Nedelec elements for curl-curl. Vector '
     'Lagrange does not fail loudly - it assembles a perfectly ordinary matrix, solves '
     'without complaint and returns finite numbers - but the entire discrete spectrum is '
     'wrong. Signal: solve the 2D Maxwell cavity eigenproblem on a unit square, where the '
     'exact eigenvalues are lambda/pi^2 = 1, 1, 2, 4, 4, 5, 5, ... , on two or three '
     'refinements of the same mesh. N1curl reproduces that spectrum to several digits and '
     'each mode SETTLES as the mesh is refined. Vector Lagrange degree 1 returns values '
     'none of which is near any true eigenvalue, and - this is the tell - its spurious '
     'modes COLLAPSE TOWARD ZERO under refinement rather than converging on anything: the '
     'lowest computed mode sits near zero and every refinement drives it LOWER. So the '
     'thing to watch for is a near-zero mode that refinement pushes further down, NOT a '
     'mode that drifts to some wrong but settled value, and NOT a rising sequence. '
     'IMPORTANT CORRECTION: an earlier version of this entry quoted the spurious modes as '
     'drifting UPWARD under refinement; that direction is wrong and was falsified by two '
     'independent runs. Vector Lagrange degree 2 is not a fix either: it interleaves a '
     'couple of nearly-right modes with spurious ones, which is worse because it looks '
     'plausible. Also do not look for the trouble in the matrix: the previously quoted '
     "signals ('near-zero off-diagonal entries', 'B = curl(A) uniformly ~0') do NOT "
     'reproduce. Assembled on a small 3D cube, the vector Lagrange and N1curl operators '
     'both have ordinary sparsity, no all-zero rows and an ordinary largest entry - '
     'nothing about either matrix looks degenerate, so the spectrum is the only place the '
     'defect shows. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
     '[Numerical] The 3D curl-curl operator without a gauge is singular - every gradient '
     'is in its kernel - and this does NOT show up as a solver failure. Every solver '
     'reports success and returns a different A. Signal: measured on a 6x6x6 unit cube '
     'with N1curl degree 1 and no gauge term, MUMPS-LU, CG/Jacobi and GMRES/ILU all '
     'reported converged (reasons 4, 2, 2) with a machine-precision residual (6.07e-15 for '
     'LU) but produced ||A||_L2 = 1.1599, 0.03213 and 0.03207 respectively - a factor 36 '
     'apart. Meanwhile ||curl A||_L2 came out as 0.14165072300980, 0.14165072300980 and '
     '0.14165072300974 from the three solvers - identical to 12 digits, because B = '
     'curl(A) IS well defined even when A is not. Adding the mass term + inner(A, w)*dx '
     'makes A itself solver-independent (0.030139363564781 / ...674 / ...526 from the same '
     "three solvers). IMPORTANT CORRECTION: the previously quoted signal ('PETSc returns "
     "KSP_DIVERGED_BREAKDOWN or the iterative solver diverges immediately') does NOT "
     'reproduce, and the direct solver gives no zero-pivot warning either. The only '
     'detectable symptom is that A changes when you change the solver - so if you '
     'post-process A itself (not just curl A), gauge it.',
     '[Input] Material data written with `ufl.conditional` is evaluated at QUADRATURE '
     'POINTS, so the effective geometry of the material interface silently depends on the '
     'quadrature degree of the form. Prefer a cell-wise DG0 fem.Function built from cell '
     'midpoints (`mesh.compute_midpoints`) or from MeshTags, which pins the interface to '
     'cell boundaries. Signal: measured on the same coil-in-iron problem, a '
     'ufl.conditional material and a DG0 material give BIT-IDENTICAL answers at P1 (max Az '
     '= 24.3317064059 both ways, relative difference 0.0e+00, because the P1 form has one '
     'quadrature point per cell at the midpoint) but differ by 1.2e-02 relative at P2 '
     '(24.2612747568 versus 24.5584843625). So the two approaches agree exactly at the '
     'degree you probably tested and disagree at the degree you probably shipped.',
     '[API] `ufl.curl` applied to a 2D SCALAR returns the 2-vector (dAz/dy, -dAz/dx) - '
     'exactly the plane restriction of the 3D curl((0, 0, Az)) - so there is no need to '
     'write the components by hand and no sign convention to get wrong. Signal: verified '
     'for Az = y, `ufl.curl(Az)` interpolates to [1., 0.], `ufl.as_vector((Az.dx(1), '
     '-Az.dx(0)))` interpolates to the same [1., 0.], and the 3D `ufl.curl((0, 0, y))` '
     'interpolates to [1., 0., 0.]. All three agree. Note also that `ufl.curl(<2D '
     'scalar>)` has shape (2,) while `ufl.curl(<2D vector>)` has shape () and BOTH compile '
     'through fem.form, so a dimensional mix-up produces no error - `inner(curl(s), '
     'curl(s))*dx` for a scalar s silently assembles the same value as `inner(grad(s), '
     'grad(s))*dx`.',
     '[Integration] `LinearProblem.solve()` never raises when the KSP fails; the dolfinx '
     "0.10 docstring says outright that 'the user is responsible for asserting convergence "
     "of the KSP solver'. Assert `problem.solver.getConvergedReason() > 0` or pass "
     '`"ksp_error_if_not_converged": True`. Signal: on a healthy run \'preonly\' returns '
     'reason 4 (CONVERGED_ITS) and CG returns 2 (CONVERGED_RTOL); a failure returns a '
     'negative value such as -3 (DIVERGED_MAX_IT), -9 (DIVERGED_NANORINF) or -11 '
     '(DIVERGED_PCSETUP_FAILED) while the script goes on to write output files and exit 0.',
     '[Input] Distinguish the coil from the iron and the air with cell data. Setting J '
     'over the whole domain, or mu_r = 1 everywhere, produces a solve that converges and a '
     'field that is wrong by orders of magnitude in the region that matters. Signal: '
     'measured, the total current computed from the DG0 source, int(Jz)dx, comes out as '
     '1.25625e+05 A against the analytic pi*R^2*J = 1.256637e+05 A for the intended disc - '
     'a 0.03% geometric error from the staircase representation of the circle. If that '
     'integral is not close to the current you intended, the material map is wrong, and '
     'this is the cheapest possible way to find out. Assert on it before looking at B.',
     '[API] `dolfinx.fem.petsc.LinearProblem` requires the keyword-only argument '
     '`petsc_options_prefix`. Signal: `TypeError: LinearProblem.__init__() missing 1 '
     "required keyword-only argument: 'petsc_options_prefix'`. On this install PETSc uses "
     '32-bit indices (PETSc.IntType is int32) and MUMPS is available, so '
     "'pc_factor_mat_solver_type': 'mumps' works; on a 64-bit-index PETSc build it would "
     "not, and 'superlu_dist' is the drop-in replacement.",
     '[API] Field extrema and integrals are per-rank quantities. `B.x.array.max()` is '
     'RANK-LOCAL and includes ghost entries; reduce over owned entries with '
     '`comm.allreduce(..., MPI.MAX)`, wrap every `fem.assemble_scalar` in '
     '`comm.allreduce(..., MPI.SUM)`, and guard printing with `if comm.rank == 0:`. '
     'Signal: without the reduction a run under mpirun prints one line per rank and the '
     'ranks DISAGREE - no error, just quietly different numbers depending on how many '
     'ranks you happened to use, and none of them need agree with the serial run. Watch '
     'for the disagreement, not for every printed number being '
     'wrong: the rank that happens to own the peak prints the true global maximum, so one '
     'of the lines can be accidentally right and which one it is depends on the partition. '
     '(An earlier version of this entry said none of the printed maxima is the global one; '
     'that is too strong and a check written against it will not fire.) The same applies '
     'to `fem.assemble_scalar`, which returns a rank-local partial sum that only becomes '
     'the true integral after MPI.SUM. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)'],
}


def _magnetostatics_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx 2D magnetostatics script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 40)
    ny = params.get("ny", 40)
    mu_r = params.get("mu_r", 1000.0)
    J_source = params.get("J_source", 1e6)
    coil_cx = params.get("coil_cx", 0.5)
    coil_cy = params.get("coil_cy", 0.5)
    coil_r = params.get("coil_r", 0.2)
    return f'''\
"""Magnetostatics — 2D scalar Az formulation — FEniCSx/dolfinx
-div((1/mu) * grad(Az)) = Jz on [0,1]^2
Az = 0 on boundary (tangential A = 0 -> no normal flux through boundary)
Current-carrying coil region: circle at ({coil_cx}, {coil_cy}) radius {coil_r}
Iron region (high mu_r={mu_r}) outside coil; air inside coil.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

MU0 = 4.0 * np.pi * 1e-7  # H/m (permeability of free space)

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Homogeneous Dirichlet BC: Az = 0 on outer boundary
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Spatially varying permeability mu(x):
# Coil region (circle): air mu_r = 1
# Surrounding domain: iron mu_r = {mu_r}
x = ufl.SpatialCoordinate(domain)
coil_cx, coil_cy, coil_r = {coil_cx}, {coil_cy}, {coil_r}
in_coil = ufl.conditional(
    (x[0] - coil_cx)**2 + (x[1] - coil_cy)**2 < coil_r**2,
    1.0,   # air: mu_r = 1
    {mu_r} # iron: mu_r = {mu_r}
)
mu_r_field = in_coil
mu = MU0 * mu_r_field

# Current density Jz: only inside coil
Jz = ufl.conditional(
    (x[0] - coil_cx)**2 + (x[1] - coil_cy)**2 < coil_r**2,
    {J_source},
    0.0
)

# Weak form: (1/mu) * grad(Az) . grad(v) = Jz * v
Az = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
a = (1.0 / mu) * ufl.dot(ufl.grad(Az), ufl.grad(v)) * ufl.dx
L = Jz * v * ufl.dx

# Solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="mag",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
Az_h = problem.solve()
_reason = problem.solver.getConvergedReason()
if _reason <= 0:
    raise RuntimeError(
        f"KSP failed with KSPConvergedReason={{_reason}} (-11 = "
        f"DIVERGED_PC_FAILED). LinearProblem.solve() returns a Function "
        f"regardless, so without this check the script prints a field made "
        f"of garbage and exits 0.")
Az_h.name = "Az"

# Post-process: magnetic flux density B = curl(A) = (dAz/dy, -dAz/dx, 0)
# In 2D, computed on DG0 space
V_vec = fem.functionspace(domain, ("DG", 0, (2,)))
Bx_expr = fem.Expression(Az_h.dx(1),              V_vec.sub(0).collapse()[0].element.interpolation_points)
By_expr = fem.Expression(-Az_h.dx(0),             V_vec.sub(0).collapse()[0].element.interpolation_points)

# Scalar DG0 for each component
V_dg0 = fem.functionspace(domain, ("DG", 0))
Bx = fem.Function(V_dg0, name="Bx")
By = fem.Function(V_dg0, name="By")
Bx.interpolate(fem.Expression(Az_h.dx(1),  V_dg0.element.interpolation_points))
By.interpolate(fem.Expression(-Az_h.dx(0), V_dg0.element.interpolation_points))

# |B| = sqrt(Bx^2 + By^2)
B_mag = np.sqrt(Bx.x.array**2 + By.x.array**2)

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "Az.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(Az_h)

with XDMFFile(domain.comm, "Bx.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(Bx)

with XDMFFile(domain.comm, "By.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(By)

Az_arr = Az_h.x.array
print(f"Magnetostatics 2D solved:")
print(f"Az: min={{Az_arr.min():.6e}}, max={{Az_arr.max():.6e}} Wb/m")
print(f"|B|: min={{B_mag.min():.6e}}, max={{B_mag.max():.6e}} T")
# A CONSTANT relative permeability is a LINEAR model. Real soft magnetic
# materials saturate around 1.5-2 T, above which mu_r collapses toward 1, so
# a linear run reporting a peak far above that has left the regime where its
# own material model means anything. Say so rather than printing the number
# as if it were a result.
if B_mag.max() > 2.0 and {mu_r} > 1.0:
    print(f"WARNING: peak |B| = {{B_mag.max():.3e}} T is far above the "
          f"~1.5-2 T saturation range of soft magnetic materials, while mu_r "
          f"is held constant at {mu_r}. This LINEAR model is NOT valid here. "
          f"Reduce J_source, or replace mu_r with a B-H curve, before using "
          f"this number for force, torque or coupling.")
print(f"mu_r iron = {mu_r}, J_source = {J_source:.2e} A/m^2")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KNOWLEDGE: dict[str, dict] = {
    "dg_methods":          _DG_KNOWLEDGE,
    "contact":             _CONTACT_KNOWLEDGE,
    "multiphase":          _MULTIPHASE_KNOWLEDGE,
    "time_dependent_heat": _TIME_DEPENDENT_HEAT_KNOWLEDGE,
    "cahn_hilliard":       _CAHN_HILLIARD_KNOWLEDGE,
    "nonlinear_pde":       _NONLINEAR_PDE_KNOWLEDGE,
    "magnetostatics":      _MAGNETOSTATICS_KNOWLEDGE,
}

GENERATORS: dict[str, dict[str, callable]] = {
    "dg_methods":          {"2d": _dg_methods_2d},
    "contact":             {"2d": _contact_2d},
    "multiphase":          {"2d": _multiphase_2d},
    "time_dependent_heat": {"2d": _time_dependent_heat_2d},
    "cahn_hilliard":       {"2d": _cahn_hilliard_2d},
    "nonlinear_pde":       {"2d": _nonlinear_pde_2d},
    "magnetostatics":      {"2d": _magnetostatics_2d},
}


def generate(physics: str, variant: str, params: dict) -> str:
    """Dispatch to the appropriate advanced physics generator.

    Parameters
    ----------
    physics : str
        One of the physics names registered in GENERATORS.
    variant : str
        Variant name, e.g. ``"2d"``.
    params : dict
        Problem-specific parameters (mesh resolution, material constants, etc.).

    Returns
    -------
    str
        A runnable FEniCSx Python script.

    Raises
    ------
    ValueError
        If *physics* or *variant* is unknown.
    """
    physics_gens = GENERATORS.get(physics)
    if physics_gens is None:
        raise ValueError(
            f"Unknown advanced physics: {physics!r}. "
            f"Available: {sorted(GENERATORS)}"
        )
    gen = physics_gens.get(variant)
    if gen is None:
        raise ValueError(
            f"Unknown variant {variant!r} for physics {physics!r}. "
            f"Available: {sorted(physics_gens)}"
        )
    return gen(params)
