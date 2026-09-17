"""Mixed Poisson / Darcy flow generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    "description": (
        "Mixed (dual) Poisson / Darcy flow: solve for the FLUX sigma and the "
        "pressure u at the same time, sigma = -grad(u) with div(sigma) = f. "
        "Discretised with H(div)-conforming Raviart-Thomas flux and "
        "discontinuous pressure, which makes the discrete mass balance hold "
        "cell by cell."
    ),
    "minimal_working_example": (
        "# COMPLETE runnable script. Darcy flow on the unit square with a\n"
        "# unit source: prescribed pressure on the left/right walls (a\n"
        "# NATURAL condition in this formulation) and no-flow on top/bottom\n"
        "# (an ESSENTIAL condition on sigma.n). Verified by executing it on\n"
        "# dolfinx 0.10.0.\n"
        "from mpi4py import MPI\n"
        "from dolfinx import mesh, fem\n"
        "from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,\n"
        "                               apply_lifting, set_bc)\n"
        "from basix.ufl import element, mixed_element\n"
        "from petsc4py import PETSc\n"
        "import ufl\n"
        "import numpy as np\n"
        "\n"
        "k = 1\n"
        "domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32,\n"
        "                                 mesh.CellType.triangle)\n"
        "fdim = domain.topology.dim - 1\n"
        "domain.topology.create_connectivity(fdim, domain.topology.dim)\n"
        "\n"
        "RT = element('RT', domain.basix_cell(), k)\n"
        "DG = element('DG', domain.basix_cell(), k - 1)\n"
        "W = fem.functionspace(domain, mixed_element([RT, DG]))\n"
        "(sigma, u) = ufl.TrialFunctions(W)\n"
        "(tau, v) = ufl.TestFunctions(W)\n"
        "\n"
        "n = ufl.FacetNormal(domain)\n"
        "f = fem.Constant(domain, 1.0)      # source, div(sigma) = f\n"
        "u_D = fem.Constant(domain, 0.0)    # prescribed pressure, natural BC\n"
        "\n"
        "# facet tag 1 = the walls where the PRESSURE is prescribed\n"
        "sides = mesh.locate_entities_boundary(\n"
        "    domain, fdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))\n"
        "mt = mesh.meshtags(domain, fdim, np.sort(sides),\n"
        "                   np.full(len(sides), 1, dtype=np.int32))\n"
        "ds = ufl.Measure('ds', domain=domain, subdomain_data=mt)\n"
        "\n"
        "a = (ufl.inner(sigma, tau) - u * ufl.div(tau) + ufl.div(sigma) * v) * ufl.dx\n"
        "L = f * v * ufl.dx - u_D * ufl.dot(tau, n) * ds(1)\n"
        "\n"
        "# essential BC: sigma.n = 0 on the remaining walls\n"
        "flat = mesh.locate_entities_boundary(\n"
        "    domain, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))\n"
        "W0 = W.sub(0)\n"
        "V0, _ = W0.collapse()\n"
        "g = fem.Function(V0)\n"
        "g.x.array[:] = 0.0\n"
        "bc = fem.dirichletbc(g, fem.locate_dofs_topological((W0, V0), fdim, flat), W0)\n"
        "\n"
        "a_form, L_form = fem.form(a), fem.form(L)\n"
        "A = assemble_matrix(a_form, bcs=[bc])\n"
        "A.assemble()\n"
        "b = assemble_vector(L_form)\n"
        "apply_lifting(b, [a_form], bcs=[[bc]])\n"
        "b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)\n"
        "set_bc(b, [bc])\n"
        "\n"
        "opts = PETSc.Options()\n"
        "opts['mat_mumps_icntl_14'] = 200   # MUMPS working-space headroom\n"
        "ksp = PETSc.KSP().create(domain.comm)\n"
        "ksp.setOperators(A)\n"
        "ksp.setType('preonly')\n"
        "ksp.getPC().setType('lu')\n"
        "ksp.getPC().setFactorSolverType('mumps')\n"
        "ksp.setFromOptions()\n"
        "ksp.getPC().setFromOptions()\n"
        "wh = fem.Function(W)\n"
        "ksp.solve(b, wh.x.petsc_vec)\n"
        "wh.x.scatter_forward()\n"
        "if ksp.getConvergedReason() <= 0:\n"
        "    raise RuntimeError(f'KSP failed, reason {ksp.getConvergedReason()}')\n"
        "\n"
        "# physical check: cellwise mass balance div(sigma_h) == f\n"
        "sig_h, u_h = wh.sub(0), wh.sub(1)\n"
        "res = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(\n"
        "    fem.form((ufl.div(sig_h) - f) ** 2 * ufl.dx)), op=MPI.SUM))\n"
        "scale = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(\n"
        "    fem.form(f ** 2 * ufl.dx)), op=MPI.SUM))\n"
        "print('relative mass-balance residual:', res / scale)\n"
        "p = wh.sub(1).collapse().x.array\n"
        "print('pressure range:', p.min(), p.max())\n"
    ),
    "function_space": {
        "REQUIRED": (
            "from basix.ufl import element, mixed_element\n"
            "RT = element('RT', domain.basix_cell(), k)       # flux,  H(div)\n"
            "DG = element('DG', domain.basix_cell(), k - 1)   # pressure, L2\n"
            "W = fem.functionspace(domain, mixed_element([RT, DG]))\n"
            "(sigma, u) = ufl.TrialFunctions(W)\n"
            "(tau, v) = ufl.TestFunctions(W)"
        ),
        "OPTIONAL": (
            "k = 1, 2, 3 (all verified on triangles and quadrilaterals). "
            "'BDM' may replace 'RT' — same stability, more DoFs, one order "
            "higher in the flux. fem.functionspace(domain, ('RT', k)) builds "
            "the same space as the basix.ufl.element spelling."
        ),
        "explanation": (
            "REQUIRED pairing: the pressure degree must be exactly one lower "
            "than the flux degree. RT(k) with DG(k) is not inf-sup stable."
        ),
        "pitfalls": [
            "[Numerical] Pair RT(k) strictly with DG(k-1). Signal: with "
            "RT(k)+DG(k) the assembled saddle-point matrix acquires extra "
            "numerical null vectors and the LU factorisation reports a zero "
            "pivot.",
        ],
    },
    "weak_form": {
        "REQUIRED": (
            "a = (ufl.inner(sigma, tau) - u * ufl.div(tau)\n"
            "     + ufl.div(sigma) * v) * ufl.dx\n"
            "L = f * v * ufl.dx - u_D * ufl.dot(tau, n) * ds(pressure_tag)"
        ),
        "OPTIONAL": (
            "Heterogeneous permeability: weight the first term, "
            "ufl.inner(Kinv * sigma, tau), where Kinv is a fem.Function or a "
            "ufl expression for 1/K."
        ),
        "explanation": (
            "This is sigma = -grad(u) with div(sigma) = f. The pressure "
            "boundary value u_D enters the RIGHT-HAND SIDE as a facet "
            "integral against tau.n — it is a NATURAL condition here, the "
            "reverse of the primal formulation."
        ),
    },
    "boundary_conditions": {
        "REQUIRED": (
            "# ESSENTIAL: prescribed normal flux sigma.n, on the flux subspace\n"
            "W0 = W.sub(0)\n"
            "V0, _ = W0.collapse()\n"
            "g = fem.Function(V0)          # a Function on the COLLAPSED space\n"
            "g.x.array[:] = 0.0            # sigma.n = 0  (no flow)\n"
            "bc = fem.dirichletbc(g, fem.locate_dofs_topological((W0, V0), fdim, facets), W0)\n"
            "\n"
            "# NATURAL: prescribed pressure, a term in L, NOT a dirichletbc\n"
            "L = f * v * ufl.dx - u_D * ufl.dot(tau, n) * ds(pressure_tag)"
        ),
        "OPTIONAL": (
            "Non-zero flux: interpolate the wanted vector field into g "
            "instead of zeroing it."
        ),
        "explanation": (
            "REQUIRED: at least part of the boundary must carry the natural "
            "pressure condition. Prescribing sigma.n on the WHOLE boundary "
            "leaves the pressure undetermined up to a constant and, unless "
            "the prescribed fluxes happen to balance the source exactly, "
            "makes the problem unsolvable."
        ),
        "pitfalls": [
            "[Numerical] Do not put sigma.n on the whole boundary. Signal: "
            "the run exits 0 but reports min(p) == max(p) at magnitude ~1e13 "
            "— a constant garbage pressure from an LU solve of a singular "
            "system.",
            "[API] The essential BC value must be a Function on the "
            "COLLAPSED flux subspace. Signal: passing a raw constant with "
            "the (W0, V0) dof tuple raises TypeError: __init__(): "
            "incompatible function arguments.",
        ],
    },
    "solver": {
        "REQUIRED": (
            "opts = PETSc.Options()\n"
            "opts['mat_mumps_icntl_14'] = 200\n"
            "ksp = PETSc.KSP().create(domain.comm)\n"
            "ksp.setOperators(A)\n"
            "ksp.setType('preonly')\n"
            "ksp.getPC().setType('lu')\n"
            "ksp.getPC().setFactorSolverType('mumps')\n"
            "ksp.setFromOptions(); ksp.getPC().setFromOptions()\n"
            "ksp.solve(b, wh.x.petsc_vec)\n"
            "if ksp.getConvergedReason() <= 0:            # REQUIRED check\n"
            "    raise RuntimeError(ksp.getConvergedReason())"
        ),
        "OPTIONAL": (
            "'umfpack' is the alternative factor solver that was clean on "
            "every (cell type, degree, mesh) combination tried here. "
            "'superlu_dist' is NOT a safe blind fallback — see the pitfall "
            "below. At scale, replace the direct solve with a fieldsplit "
            "Schur-complement preconditioner."
        ),
        "explanation": (
            "The system is INDEFINITE (a saddle point), so Krylov methods "
            "that assume positive definiteness fail on it. BOTH checks are "
            "REQUIRED: the converged reason catches a failed factorisation "
            "(which otherwise leaves inf in the solution while the script "
            "exits 0), and the mass-balance check catches a factorisation "
            "that reports success and returns a wrong answer."
        ),
        "pitfalls": [
            "[Numerical] Always test ksp.getConvergedReason() > 0. Signal: "
            "reason -11 (KSP_DIVERGED_PC_FAILED) with the solution vector "
            "full of inf, while the process exit status is still 0.",
            "[Numerical] The converged reason is NOT sufficient on its own. "
            "Signal: superlu_dist returns reason 4 (converged) on this "
            "saddle point while ||div(sigma_h) - f|| / ||f|| comes back at "
            "order 1-100 instead of machine epsilon. Only the mass-balance "
            "check catches it.",
            "[Numerical] MUMPS can run out of factorisation workspace on "
            "some meshes. Signal: 'MUMPS error in numerical factorization: "
            "INFOG(1)=-9'. Setting mat_mumps_icntl_14 to a few hundred "
            "recovers those cases.",
        ],
    },
    "verification": (
        "The mixed method is locally conservative by construction: when the "
        "source lies in the pressure space (e.g. a constant f with DG0), "
        "div(sigma_h) equals f to machine precision. Assemble "
        "((div(sigma_h) - f)**2 * dx)**0.5 and compare it against the norm of "
        "f. A relative residual near machine epsilon is a genuine physical "
        "check that needs no reference solution; O(1) means the solve failed."
    ),
    "pitfalls": [
        "[Numerical] Prescribing sigma.n on the ENTIRE boundary is not a "
        "well-posed problem. The pressure is then determined only up to a "
        "constant, and unless the prescribed normal fluxes integrate to "
        "exactly the source, no solution exists at all. Signal: the script "
        "exits 0 and prints a single constant pressure of enormous magnitude "
        "— min(p) == max(p) at order 1e13 — because LU on the singular "
        "saddle-point system returns garbage rather than failing. Fix: leave "
        "part of the boundary to the NATURAL pressure condition "
        "(- u_D * dot(tau, n) * ds(tag) in L), or, if the flux really is "
        "prescribed everywhere, attach a constant nullspace to the pressure "
        "block and make the data compatible. (Verified by execution "
        "2026-08-03, dolfinx 0.10.0 — this is the defect the shipped "
        "template used to have.)",
        "[Numerical] A direct solve of this saddle point can fail while the "
        "script still exits 0. Signal: ksp.getConvergedReason() returns -11 "
        "(KSP_DIVERGED_PC_FAILED) and the solution vector is filled with "
        "inf, but nothing is printed and the return code is 0. Setting "
        "ksp.setErrorIfNotConverged(True) surfaces the underlying cause: "
        "'MUMPS error in numerical factorization: INFOG(1)=-9' for the MUMPS "
        "factoriser, and '[0] Zero pivot row 0 value 0. tolerance "
        "2.22045e-14' for PETSc's built-in LU. ALWAYS test "
        "getConvergedReason() > 0 — and do not stop there, because a "
        "converged reason is not proof of a correct answer here (next "
        "pitfall). (Verified by execution 2026-08-03.)",
        "[Numerical] No direct solver was robust across every "
        "(cell type, RT degree, mesh size) combination here, and the "
        "failures are SPORADIC — they do not follow a monotone rule in "
        "degree or mesh size, so you cannot predict them, you have to check "
        "for them. Signal: over a clean-process sweep of triangles and "
        "quadrilaterals at RT degree 1/2/3 and several mesh sizes, MUMPS "
        "returned KSPConvergedReason -11 on a few isolated triangle cases "
        "(underlying cause 'MUMPS error in numerical factorization: "
        "INFOG(1)=-9', a workspace shortage, which setting "
        "mat_mumps_icntl_14 to a few hundred fixed on exactly those cases); "
        "superlu_dist returned -11 on a different, non-overlapping set; and "
        "umfpack was clean on all of them but is sequential. WORST CASE, and "
        "the reason the mass-balance check is mandatory: superlu_dist also "
        "produced several quadrilateral cases with KSPConvergedReason 4 "
        "(converged) and a mass-balance residual of order 1-100 with a "
        "visibly wrong pressure range — a silent wrong answer that no solver "
        "status reveals. Do not use superlu_dist as a blind fallback. "
        "(Verified by execution 2026-08-03, each configuration in its own "
        "process. NOTE: an adversarial re-check that swept different mesh "
        "sizes saw MUMPS succeed everywhere and concluded the workspace "
        "option was inert — it is not inert, it is only exercised on the "
        "configurations that fail, which is the point of this pitfall.)",
        "[Numerical] The system is INDEFINITE (saddle point): use a direct "
        "solver or a Schur-complement block preconditioner. Signal: PETSc CG "
        "returns KSP_DIVERGED_INDEFINITE_PC and GMRES stagnates with the "
        "residual failing to drop, while the same matrix factorises fine "
        "under LU. (Audit 2026-06-02.)",
        "[Numerical] RT(k) pairs with DG(k-1), never DG(k). Signal: count "
        "the DoFs first - DG(k) carries more pressure DoFs than the RT(k) "
        "flux space can constrain, so the divergence block cannot have full "
        "row rank, and the assembled saddle-point matrix then has extra "
        "numerical null vectors (measure the numerical null dimension of "
        "the assembled matrix; RT(k)+DG(k-1) does not have them). TWO "
        "CORRECTIONS to the way this used to be diagnosed. First, 'the "
        "pressure error does not converge under refinement' is NOT "
        "observable: the system is singular, so there is no solution to "
        "take an error from and no rate to plot - do not wait for a bad "
        "convergence order, it never arrives. Second, a zero pivot only "
        "discriminates under a PIVOTING factorisation (a dense LAPACK LU "
        "of the same matrix stops with 'Zero pivot in LU factorization'); "
        "PETSc's SPARSE LU does no pivoting and trips over the zero "
        "pressure block of BOTH pairs, so a zero-pivot failure from the "
        "default sparse direct solver tells you nothing about which pair "
        "you used. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)",
        "[Numerical] BDM(k) + DG(k-1) is the alternative H(div) pair, with "
        "the FULL polynomial space rather than RT's subspace. Signal: BDM "
        "gives the same inf-sup stability at roughly twice the k=1 flux DoF "
        "count in 2D, and buys one extra order in the flux, not in the "
        "pressure — choosing BDM while measuring only the pressure error "
        "shows the same rate as RT at higher cost. (Audit 2026-06-02.)",
        "[API] The essential condition is on sigma.n (the flux), NOT on the "
        "pressure — the pressure is the NATURAL one here, imposed by the "
        "facet term - u_D * dot(tau, n) * ds(tag) in the linear form. "
        "Signal: putting a dolfinx dirichletbc on the pressure subspace "
        "instead constrains the discontinuous pressure DoFs directly, giving "
        "a pressure that is pinned to the prescribed value only in the cells "
        "touching the wall and a normal flux that does not respond to the "
        "imposed pressure difference at all. (Audit 2026-06-02.)",
        "[Numerical] For heterogeneous permeability K(x), weight the flux "
        "term by K^-1: ufl.inner(Kinv * sigma, tau) * dx. Signal: with the "
        "weighting omitted in a layered domain the computed flux is "
        "continuous across the layer interface instead of jumping with the "
        "permeability contrast, and the mass-balance residual at the "
        "interface is O(1). Putting K only into a post-processed "
        "sigma = -K grad(p) does not fix the solve. (Audit 2026-06-02.)",
        "[API] Both spellings work for Raviart-Thomas: "
        "basix.ufl.element('RT', domain.basix_cell(), k) and the tuple "
        "shorthand fem.functionspace(domain, ('RT', k)) build the SAME "
        "space. Signal: both give identical global dof counts at k=1 on the "
        "same mesh and neither raises. IMPORTANT CORRECTION: the previously "
        "quoted signal (AttributeError: module 'dolfinx.fem' has no "
        "attribute 'FiniteElement' from passing 'RT' to FunctionSpace) is "
        "not reproducible — 'RT' IS the registered basix family name. The "
        "names that DO raise ValueError 'Unknown element family: ...' are "
        "the old DOLFIN degree-suffixed spellings such as 'P1'; and "
        "ufl.FiniteElement itself no longer exists at all (AttributeError on "
        "the ufl module). (Verified by execution 2026-08-03.)",
    ],
    "materials": {
        "permeability": {"range": [1e-15, 1.0], "unit": "m^2 (Darcy permeability)"},
    },
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate mixed Poisson variant."""
    generators = {
        "2d": _mixed_poisson_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown mixed_poisson variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _mixed_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    rt_order = params.get("rt_order", 1)
    source = params.get("source", 1.0)
    pressure_bc = params.get("pressure_bc", 0.0)
    return f'''\
"""Mixed Poisson / Darcy: Raviart-Thomas flux + DG pressure — FEniCSx/dolfinx

    sigma = -grad(u)        (Darcy law, flux)
    div(sigma) = f          (mass balance)
    sigma in H(div), u in L2.

BOUNDARY CONDITIONS — the part that decides well-posedness:
  * ESSENTIAL here is the NORMAL FLUX sigma.n, imposed with a dirichletbc on
    the flux subspace.  Applied on y=0 and y=1 (no-flow walls).
  * NATURAL here is the PRESSURE, imposed by the facet term
    -u_D*dot(tau, n)*ds(1) in the linear form.  Applied on x=0 and x=1.
  Prescribing sigma.n on the WHOLE boundary is NOT well posed: the pressure
  is then fixed only up to a constant, and unless the fluxes balance the
  source exactly there is no solution at all — LU returns a constant garbage
  pressure of order 1e13 and the script still exits 0.

The run is checked by the cellwise mass balance div(sigma_h) = f, which the
RT/DG pair satisfies to machine precision when f lies in the pressure space.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl
import numpy as np
from basix.ufl import element, mixed_element
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Mixed function space: Raviart-Thomas flux + one-degree-lower DG pressure
RT = element("RT", domain.basix_cell(), {rt_order})
DG = element("DG", domain.basix_cell(), {rt_order - 1})
W = fem.functionspace(domain, mixed_element([RT, DG]))

(sigma, u) = ufl.TrialFunctions(W)
(tau, v) = ufl.TestFunctions(W)

n = ufl.FacetNormal(domain)
f = fem.Constant(domain, default_scalar_type({source}))        # source term
u_D = fem.Constant(domain, default_scalar_type({pressure_bc}))  # prescribed pressure

# Facet tag 1 = the walls carrying the natural (pressure) condition
pressure_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
marker = mesh.meshtags(domain, fdim, np.sort(pressure_facets),
                       np.full(len(pressure_facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=marker)

# Weak form for sigma = -grad(u), div(sigma) = f
a = (ufl.inner(sigma, tau) - u * ufl.div(tau) + ufl.div(sigma) * v) * ufl.dx
L = f * v * ufl.dx - u_D * ufl.dot(tau, n) * ds(1)

# Essential BC: sigma.n = 0 on the no-flow walls (flux subspace, collapsed Function)
noflow_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
W0 = W.sub(0)
V0, _ = W0.collapse()
sigma_bc = fem.Function(V0)
sigma_bc.x.array[:] = 0.0
bc = fem.dirichletbc(sigma_bc,
                     fem.locate_dofs_topological((W0, V0), fdim, noflow_facets), W0)

# Assemble
a_form = fem.form(a)
L_form = fem.form(L)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

A = assemble_matrix(a_form, bcs=[bc])
A.assemble()
b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Solve — the system is INDEFINITE (saddle point), so use a direct solver.
# mat_mumps_icntl_14 raises MUMPS' working-space headroom; without it the
# factorisation of higher-order RT systems aborts with INFOG(1)=-9 and the
# KSP returns reason -11 while the script still exits 0.
opts = PETSc.Options()
opts["mat_mumps_icntl_14"] = 200
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")
solver.setFromOptions()
solver.getPC().setFromOptions()

wh = fem.Function(W)
solver.solve(b, wh.x.petsc_vec)
wh.x.scatter_forward()

reason = solver.getConvergedReason()
if reason <= 0:
    raise RuntimeError(
        f"KSP failed with KSPConvergedReason={{reason}} (-11 = "
        f"DIVERGED_PC_FAILED). Try a larger mat_mumps_icntl_14, or "
        f"setFactorSolverType('superlu_dist') / ('umfpack').")

# ---- physical check: cellwise mass balance, no reference solution needed ---
sigma_h = wh.sub(0)
u_h = wh.sub(1)
res = np.sqrt(domain.comm.allreduce(
    fem.assemble_scalar(fem.form((ufl.div(sigma_h) - f) ** 2 * ufl.dx)), op=MPI.SUM))
scale = np.sqrt(domain.comm.allreduce(
    fem.assemble_scalar(fem.form(f ** 2 * ufl.dx)), op=MPI.SUM))
rel_balance = res / scale if scale > 0 else res

p_arr = wh.sub(1).collapse().x.array
if not np.all(np.isfinite(p_arr)):
    raise RuntimeError("pressure contains non-finite values — the solve failed")
# The RT/DG pair is locally conservative by construction: when the source lies
# in the pressure space this residual is at machine epsilon. Anything larger
# means the factorisation returned a wrong answer WITHOUT reporting it — the
# converged reason alone does not catch that (superlu_dist has been observed
# returning reason 4 with a residual of order 1-100 on this system).
if rel_balance > 1e-8:
    raise RuntimeError(
        f"mass balance violated: ||div(sigma_h)-f||/||f|| = {{rel_balance:.3e}} "
        f"but the RT/DG pair is exactly conservative for a source in the "
        f"pressure space. KSPConvergedReason was {{reason}}, so the solver "
        f"did not report the failure. Try another "
        f"setFactorSolverType(...) — 'umfpack' was clean on every case "
        f"tested — or check that the source really lies in the DG space.")
if p_arr.max() - p_arr.min() < 1e-14 * max(abs(p_arr).max(), 1.0):
    raise RuntimeError(
        "pressure is constant to round-off: the flux BCs probably cover the "
        "whole boundary, leaving the pressure undetermined.")

# Export pressure (interpolate the DG field to P1 for XDMF)
P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(wh.sub(1).collapse())

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

print(f"Mixed Poisson solved (KSPConvergedReason={{reason}})")
print(f"pressure: min={{p_arr.min():.6e}}, max={{p_arr.max():.6e}}")
print(f"relative mass-balance residual ||div(sigma_h)-f||/||f||: {{rel_balance:.3e}}")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''
