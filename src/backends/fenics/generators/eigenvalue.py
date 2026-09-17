"""Eigenvalue problem generator for FEniCSx/dolfinx + SLEPc.

Variants: 2d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['eigenvalue'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": (
        "Generalised eigenvalue problems A x = lambda M x via SLEPc "
        "(the eigenvalue counterpart of PETSc). A is the stiffness matrix, "
        "M the mass matrix. Used for vibration modes, buckling loads and "
        "waveguide modes."
    ),
    "minimal_working_example": (
        "# COMPLETE runnable script. Dirichlet Laplacian eigenmodes on the\n"
        "# unit square. Verified by executing it on dolfinx 0.10.0 /\n"
        "# slepc4py 3.24.3.\n"
        "from mpi4py import MPI\n"
        "from dolfinx import mesh, fem, default_scalar_type\n"
        "from dolfinx.fem.petsc import assemble_matrix\n"
        "from slepc4py import SLEPc\n"
        "import ufl\n"
        "import numpy as np\n"
        "\n"
        "N_EIGS = 5\n"
        "domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32,\n"
        "                                 mesh.CellType.triangle)\n"
        "tdim = domain.topology.dim\n"
        "fdim = tdim - 1\n"
        "domain.topology.create_connectivity(fdim, tdim)\n"
        "V = fem.functionspace(domain, ('Lagrange', 1))\n"
        "\n"
        "def boundary(x):\n"
        "    return (np.isclose(x[0], 0) | np.isclose(x[0], 1)\n"
        "            | np.isclose(x[1], 0) | np.isclose(x[1], 1))\n"
        "facets = mesh.locate_entities_boundary(domain, fdim, boundary)\n"
        "bc = fem.dirichletbc(default_scalar_type(0),\n"
        "                     fem.locate_dofs_topological(V, fdim, facets), V)\n"
        "\n"
        "u = ufl.TrialFunction(V)\n"
        "v = ufl.TestFunction(V)\n"
        "a = fem.form(ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)\n"
        "m = fem.form(u * v * ufl.dx)\n"
        "\n"
        "A = assemble_matrix(a, bcs=[bc])          # diag defaults to 1\n"
        "A.assemble()\n"
        "M = assemble_matrix(m, bcs=[bc], diag=0.0)\n"
        "M.assemble()\n"
        "\n"
        "eps = SLEPc.EPS().create(MPI.COMM_WORLD)\n"
        "eps.setOperators(A, M)\n"
        "eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)\n"
        "eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)\n"
        "eps.setDimensions(N_EIGS, max(4 * N_EIGS, 20))\n"
        "eps.setTolerances(tol=1e-9, max_it=2000)\n"
        "eps.getST().setType(SLEPc.ST.Type.SINVERT)\n"
        "eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)\n"
        "eps.setTarget(0.0)\n"
        "eps.solve()\n"
        "\n"
        "n_conv = eps.getConverged()\n"
        "if n_conv < N_EIGS:\n"
        "    raise RuntimeError(f'converged {n_conv} of {N_EIGS}, '\n"
        "                       f'reason {eps.getConvergedReason()}')\n"
        "xr, xi = A.createVecs()\n"
        "Ax = A.createVecLeft()\n"
        "Mx = A.createVecLeft()\n"
        "eigenvalues, residuals = [], []\n"
        "for i in range(N_EIGS):\n"
        "    lam = eps.getEigenvalue(i).real\n"
        "    eps.getEigenvector(i, xr, xi)\n"
        "    A.mult(xr, Ax)\n"
        "    M.mult(xr, Mx)\n"
        "    Ax.axpy(-lam, Mx)          # residual A x - lambda M x\n"
        "    eigenvalues.append(lam)\n"
        "    residuals.append(Ax.norm() / max(xr.norm(), 1e-300))\n"
        "if max(residuals) > 1e-6 or any(l <= 0 for l in eigenvalues):\n"
        "    raise RuntimeError(f'bad eigenpairs: {eigenvalues} {residuals}')\n"
        "print('eigenvalues:', eigenvalues)\n"
        "print('max residual:', max(residuals))\n"
    ),
    "function_space": {
        "REQUIRED": "V = fem.functionspace(domain, ('Lagrange', degree))",
        "OPTIONAL": (
            "degree 1, 2, 3, ...; any cell type. For electromagnetic modes "
            "use ('N1curl', degree); for structural modes a blocked space, "
            "fem.functionspace(domain, ('Lagrange', degree, (gdim,)))."
        ),
        "explanation": (
            "Both matrices must be assembled on the SAME space. Verified on "
            "dolfinx 0.10.0 for Lagrange degree 1/2/3 on triangles and for "
            "degree 1 on quadrilaterals."
        ),
    },
    "weak_form": {
        "REQUIRED": (
            "a = fem.form(ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)  "
            "# stiffness -> A\n"
            "m = fem.form(u * v * ufl.dx)                              "
            "# mass      -> M"
        ),
        "OPTIONAL": (
            "Weight either form with a material fem.Constant / Function "
            "(k * grad-grad for conductivity, rho * u * v for density)."
        ),
        "explanation": (
            "Two SEPARATE forms are required: SLEPc needs both operators. "
            "eps.setOperators(A) alone solves the STANDARD problem "
            "A x = lambda x, whose eigenvalues are not the physical ones."
        ),
        "pitfalls": [
            "[Numerical] Pass BOTH matrices: eps.setOperators(A, M). "
            "Signal: with setOperators(A) alone the returned eigenvalues "
            "scale with the mesh size instead of converging to a "
            "mesh-independent limit, because the mass weighting is missing."
        ],
    },
    "boundary_conditions": {
        "REQUIRED": (
            "bc = fem.dirichletbc(default_scalar_type(0),\n"
            "                     fem.locate_dofs_topological(V, fdim, facets), V)\n"
            "A = assemble_matrix(a, bcs=[bc])           # REQUIRED: diag left at 1\n"
            "A.assemble()\n"
            "M = assemble_matrix(m, bcs=[bc], diag=0.0) # REQUIRED: diag=0.0 on the MASS matrix\n"
            "M.assemble()"
        ),
        "OPTIONAL": (
            "Natural (Neumann) boundaries need no BC at all; then assemble "
            "both matrices with bcs=[] and expect a zero eigenvalue with a "
            "constant eigenvector."
        ),
        "explanation": (
            "The constrained rows must NOT get the same diagonal in A and in "
            "M. diag=0.0 on M sends the constrained modes to infinity, which "
            "leaves exactly the constrained discrete spectrum in the finite "
            "part. This combination requires the shift-and-invert spectral "
            "transform below, because the default transform factorises M."
        ),
        "pitfalls": [
            "[Numerical] Do NOT leave the mass matrix at the default "
            "diag=1 when A also has diag=1. Signal: the lowest returned "
            "eigenvalues come back as exactly 1.0 (1.0000000000003, "
            "1.0000000000031, ...), one per constrained DoF, and the "
            "physical spectrum starts only after them.",
            "[API] The kwarg is `diag`, not `diagonal`. Signal: TypeError: "
            "assemble_matrix() got an unexpected keyword argument 'diagonal'.",
        ],
    },
    "solver": {
        "REQUIRED": (
            "eps = SLEPc.EPS().create(MPI.COMM_WORLD)\n"
            "eps.setOperators(A, M)\n"
            "eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)\n"
            "eps.getST().setType(SLEPc.ST.Type.SINVERT)   # REQUIRED with diag=0.0 on M\n"
            "eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)\n"
            "eps.setTarget(0.0)\n"
            "eps.solve()"
        ),
        "OPTIONAL": (
            "eps.setType(...): KRYLOVSCHUR (default, recommended), ARNOLDI, "
            "LANCZOS, POWER, JD. eps.setDimensions(nev, ncv) with ncv >= 2*nev. "
            "eps.setTarget(sigma) anywhere in the spectrum for interior modes."
        ),
        "explanation": (
            "Shift-and-invert is REQUIRED here, not an optimisation: with "
            "diag=0.0 the mass matrix is singular, and the default spectral "
            "transform factorises it directly."
        ),
        "pitfalls": [
            "[Numerical] diag=0.0 on M with the DEFAULT spectral transform "
            "aborts. Signal: '[0] Zero pivot row 0 value 0. tolerance "
            "2.22045e-14' from MatPivotCheck_none inside STSetUp_Shift, "
            "surfacing in Python as 'SystemError: <cyfunction EPS.solve at "
            "0x...> returned a result with an exception set'. Adding "
            "eps.getST().setType(SLEPc.ST.Type.SINVERT) removes it.",
        ],
    },
    "verification": (
        "Check each eigenpair by its own residual, which needs no reference "
        "solution: A.mult(x, Ax); M.mult(x, Mx); Ax.axpy(-lam, Mx); then "
        "Ax.norm() / x.norm() must be at the solver tolerance. Also assert "
        "every eigenvalue is positive for a Dirichlet Laplacian. Return code "
        "0 proves nothing here — the known failure mode is a run that exits 0 "
        "and reports spurious constraint modes as if they were physical."
    ),
    "pitfalls": [
        "[Numerical] For A x = lambda M x with Dirichlet BCs, the "
        "constrained rows of A and of M must NOT carry the same diagonal "
        "value, or every constrained DoF contributes a spurious eigenvalue "
        "at lambda = A_ii / M_ii. Assembling A with bcs (default diag=1) and "
        "M ALSO with bcs (default diag=1) puts those spurious modes at "
        "exactly lambda = 1 and, since they are smaller than the physical "
        "fundamental for this problem, SMALLEST_REAL returns them FIRST. "
        "Signal: the reported spectrum begins with values equal to 1 to "
        "within round-off (1.0000000000003, 1.0000000000031) and the "
        "physical fundamental appears only in third place. Correct recipe: "
        "assemble M with bcs=[bc], diag=0.0 AND switch the spectral "
        "transform to SINVERT. (Verified by execution 2026-08-03, "
        "dolfinx 0.10.0 / slepc4py 3.24.3, Lagrange degree 1/2/3 on "
        "triangles and degree 1 on quadrilaterals.)",
        "[Numerical] diag=0.0 on the mass matrix REQUIRES shift-and-invert. "
        "The default EPS spectral transform is a plain shift, which "
        "factorises M itself; with the constrained rows zeroed that "
        "factorisation has an exact zero pivot. Signal: '[0] Zero pivot in "
        "LU factorization: https://petsc.org/release/faq/#zeropivot' "
        "followed by '[0] Zero pivot row 0 value 0. tolerance 2.22045e-14', "
        "raised through the frames STSetUp_Shift -> KSPSetUp -> PCSetUp_LU "
        "-> MatLUFactorNumeric_SeqAIJ, and reaching Python as 'SystemError: "
        "<cyfunction EPS.solve at 0x...> returned a result with an exception "
        "set'. Adding eps.getST().setType(SLEPc.ST.Type.SINVERT) makes the "
        "same run succeed. (Verified by execution 2026-08-03 — this "
        "corrects an earlier catalog claim that diag=0.0 works out of the "
        "box.)",
        "[Numerical] Assembling M with bcs=[] (no BC at all) does NOT give "
        "the constrained discrete spectrum when M is the CONSISTENT mass "
        "matrix: its boundary-interior blocks are non-zero, so it is a "
        "different pencil whose eigenvalues merely approach the constrained "
        "ones as the mesh is refined. Signal: on a coarse mesh the "
        "eigenvalues from M with bcs=[] differ from those of the "
        "exactly-reduced interior-DoF problem in the second significant "
        "digit, and the gap closes under refinement — which is why the error "
        "is easy to miss on a fine mesh — while M with diag=0.0 reproduces "
        "the reduced problem to machine precision at EVERY mesh and returns "
        "exactly as many finite eigenvalues as there are unconstrained DoFs. "
        "SCOPE: with a mass-LUMPED M the boundary-interior blocks vanish and "
        "bcs=[] is exact, but the constrained modes then reappear at "
        "lambda = 1/M_ii and interleave with the physical spectrum. "
        "diag=0.0 + SINVERT is correct in both cases. (Verified by execution "
        "2026-08-03 against a dense scipy eigensolve of the interior block — "
        "this corrects an earlier catalog claim that bcs=[] is a clean "
        "recipe.)",
        "[API] dolfinx 0.10's assemble_matrix takes `diag`, not `diagonal`. "
        "Signal: TypeError: assemble_matrix() got an unexpected keyword "
        "argument 'diagonal'. The 0.10 signature is (a, bcs=None, diag=1, "
        "constants=None, coeffs=None, kind=None). (Verified by execution "
        "2026-08-03.)",
        "[Numerical] eps.setOperators(A) alone solves the STANDARD problem "
        "A x = lambda x, not the physical generalised one. Signal: the "
        "returned eigenvalues track the mesh spacing instead of converging "
        "to mesh-independent values under refinement; setOperators(A, M) "
        "restores mesh-independent limits. (Verified by execution "
        "2026-08-03.)",
        "[Integration] slepc4py is a package separate from petsc4py and can "
        "be missing. Signal: `from slepc4py import SLEPc` raises "
        "ModuleNotFoundError: No module named 'slepc4py'. On the reference "
        "install it is present (slepc4py 3.24.3) and SLEPc.EPS resolves. "
        "(Verified by execution 2026-08-03.)",
        "[Numerical] Always verify eigenpairs by their own residual rather "
        "than by the return code. Signal: a script that only checks the exit "
        "status reports spurious constraint modes as physical eigenvalues; "
        "computing ||A x - lambda M x|| / ||x|| for each returned pair, and "
        "requiring lambda > 0 for a Dirichlet Laplacian, catches it without "
        "any reference solution. (Verified by execution 2026-08-03.)",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate eigenvalue variant."""
    generators = {
        "2d": _eigenvalue_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown eigenvalue variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    n_eigs = params.get("n_eigenvalues", 5)
    degree = params.get("degree", 1)
    return f'''\
"""Eigenvalue problem: Laplace on [0,1]² — FEniCSx + SLEPc

Generalised problem A x = lambda M x with homogeneous Dirichlet BCs.

BC handling (this is the whole difficulty of the problem):
  A is assembled WITH the BCs and the default unit diagonal on the
  constrained rows.  M is assembled WITH the BCs and diag=0.0, which sends
  every constrained mode to infinity and leaves exactly the constrained
  discrete spectrum in the finite part.  Giving BOTH matrices the same
  diagonal instead puts one spurious eigenvalue per constrained DoF at
  lambda = A_ii / M_ii = 1, and those are returned ahead of the physical
  modes.  A singular M forces the shift-and-invert spectral transform: the
  default transform factorises M directly and hits an exact zero pivot.

The run is checked by the eigenpair residual ||A x - lambda M x|| / ||x||,
which needs no reference solution, and by the positivity of the spectrum.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl
import numpy as np
import json

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", {degree}))
def boundary(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0) | np.isclose(x[1], 1)
bc_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
bc = fem.dirichletbc(default_scalar_type(0),
    fem.locate_dofs_topological(V, fdim, bc_facets), V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = fem.form(ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
m = fem.form(u * v * ufl.dx)

from dolfinx.fem.petsc import assemble_matrix
A = assemble_matrix(a, bcs=[bc])          # diag defaults to 1
A.assemble()
M = assemble_matrix(m, bcs=[bc], diag=0.0)
M.assemble()

n_eigs = {n_eigs}
try:
    from slepc4py import SLEPc
except ImportError:
    print("SLEPc not available — eigenvalue solve skipped")
    with open("results_summary.json", "w") as f:
        json.dump({{"note": "slepc4py not installed",
                   "n_dofs": V.dofmap.index_map.size_global}}, f, indent=2)
    raise SystemExit(0)

eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setOperators(A, M)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
eigensolver.setDimensions(n_eigs, max(4 * n_eigs, 20))
eigensolver.setTolerances(tol=1e-9, max_it=2000)
eigensolver.getST().setType(SLEPc.ST.Type.SINVERT)
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eigensolver.setTarget(0.0)
eigensolver.solve()

n_conv = eigensolver.getConverged()
reason = eigensolver.getConvergedReason()
if n_conv < n_eigs:
    raise RuntimeError(
        f"SLEPc converged {{n_conv}} of {{n_eigs}} requested eigenpairs "
        f"(EPSConvergedReason={{reason}}). Increase ncv via setDimensions "
        f"or loosen setTolerances.")

xr, xi = A.createVecs()
Ax = A.createVecLeft()
Mx = A.createVecLeft()
eigenvalues = []
residuals = []
for i in range(n_eigs):
    lam = eigensolver.getEigenvalue(i).real
    eigensolver.getEigenvector(i, xr, xi)
    A.mult(xr, Ax)
    M.mult(xr, Mx)
    Ax.axpy(-lam, Mx)                     # A x - lambda M x
    eigenvalues.append(lam)
    residuals.append(Ax.norm() / max(xr.norm(), 1e-300))

# ---- physical checks (NOT the return code) -------------------------------
max_res = max(residuals)
if not np.isfinite(max_res) or max_res > 1e-6:
    raise RuntimeError(f"eigenpair residual too large: {{residuals}}")
if any(lam <= 0.0 for lam in eigenvalues):
    raise RuntimeError(
        f"non-positive eigenvalue for an SPD Dirichlet problem: {{eigenvalues}}")

print(f"Eigenvalues: {{eigenvalues}}")
print(f"Max relative eigen-residual ||A x - lam M x||/||x||: {{max_res:.3e}}")
print(f"Converged: {{n_conv}} (reason {{reason}}), DOFs: {{V.dofmap.index_map.size_global}}")

summary = {{
    "eigenvalues": eigenvalues,
    "eigen_residuals": residuals,
    "max_eigen_residual": max_res,
    "n_converged": n_conv,
    "eps_converged_reason": reason,
    "n_dofs": V.dofmap.index_map.size_global,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Eigenvalue analysis complete.")
'''
