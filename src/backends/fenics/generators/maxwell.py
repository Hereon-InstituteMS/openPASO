"""Maxwell / curl-curl generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['maxwell'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Maxwell curl-curl: ∇×(μ⁻¹ ∇×E) - k₀²εᵣE = J. H(curl) (Nédélec / N1E) elements.",
    "weak_form": "inner(curl(E), curl(v))*dx - k0**2 * eps_r * inner(E, v)*dx = inner(J, v)*dx",
    "function_space": (
        "Nédélec 1st kind (N1E) — H(curl) conforming, tangential "
        "continuity across element interfaces. Build via "
        "basix.ufl.element with family=basix.ElementFamily.N1E."
    ),
    "solver": {
        "real": "GMRES + AMS preconditioner (hypre) for H(curl)",
        "complex": "GMRES + ILU (complex PETSc build required)",
    },
    "pitfalls": [
        "[API] Use Nédélec H(curl), NOT Lagrange. Standard Lagrange P1/P2 "
        "for vector fields gives spurious modes near corners (Maxwell "
        "is unstable on H1). "
        "Signal: solution has parasitic high-frequency oscillations at "
        "geometric corners, growing with mesh refinement instead of decaying.",
        "[Syntax] basix N1E element constructor: basix.ufl.element("
        "basix.ElementFamily.N1E, domain.basix_cell(), order). "
        "Signal: passing 'Nedelec' as a string raises 'unknown family'.",
        "[Numerical] curl-curl is INDEFINITE (like Helmholtz). CG fails. "
        "Use GMRES + AMS (hypre) preconditioner for large problems. "
        "Signal: SolverCG breakdown / oscillating residual on Maxwell.",
        "[Integration] BCs in H(curl): tangential component is "
        "constrained (perfect electric conductor: n × E = 0). Use "
        "fem.locate_dofs_topological on the Nedelec space — Dirichlet "
        "on E projects the tangential component to zero. "
        "Signal: using the wrong dolfinx dirichletbc type (e.g. "
        "fixing all components of the Nedelec Function) produces "
        "an over-constrained solution with wrong eigenmodes.",
        "[Output] Nédélec / H(curl) Functions CANNOT be written "
        "directly to VTX or XDMF for visualization — both writers "
        "raise 'Cannot interpolate function ... to the VTX output "
        "basis' / 'ADIOS2 VTX only supports Lagrange elements' or "
        "the equivalent XDMF check. ParaView/VisIt also can't "
        "render H(curl) DOFs sensibly because they're edge "
        "tangents, not nodal point values. Workaround (canonical, "
        "from cpp/demo/interpolation-io/main.cpp:interpolate_nedelec): "
        "create a vector-valued DISCONTINUOUS Lagrange space of "
        "the SAME OR HIGHER degree, interpolate the Nedelec "
        "Function into it via `u_dg.interpolate(u_nedelec)`, then "
        "VTX/XDMF-write u_dg. The 6th positional arg of "
        "basix.create_element / basix.ufl.element is the "
        "`discontinuous` bool — set True. Example: for N1E degree "
        "2, build a degree-2 vector DG Lagrange space (it contains "
        "N1E degree 2). Signal: opening the resulting .bp file in "
        "ParaView, the tangential component is continuous across "
        "edges while the normal component visibly jumps at "
        "interfaces — that's the correct H(curl) signature, not "
        "a bug. (File walk cpp/demo/interpolation-io/main.cpp "
        "2026-06-03.)",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Maxwell variant."""
    generators = {
        "2d": _maxwell_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(
            f"Unknown Maxwell variant: {variant!r}. "
            f"Available: {list(generators)}")
    return gen(params)


def _maxwell_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Maxwell time-harmonic on a unit square
    with PEC (tangential E = 0) on the boundary. Source is a
    centered current density. H(curl) discretization via Nédélec
    first-kind elements. Real-valued (no complex source); for
    complex problems switch scalar_type=np.complex128."""
    nx = params.get("nx", 32)
    k0_val = params.get("k0", 3.0)
    return f'''\
"""Maxwell curl-curl — FEniCSx/dolfinx (Nédélec H(curl))"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix
import basix.ufl
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx},
                                 mesh.CellType.triangle)

# H(curl) — Nédélec 1st kind, order 1
V = fem.functionspace(
    domain,
    basix.ufl.element(basix.ElementFamily.N1E,
                       domain.basix_cell(), 1))

k0 = fem.Constant(domain, default_scalar_type({k0_val}))

E = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
# Localized current source: J = exp(-r²/sigma²) * e_x
J = ufl.as_vector([
    ufl.exp(-20.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)),
    0.0,
])
a = (ufl.inner(ufl.curl(E), ufl.curl(v)) * ufl.dx
     - k0 * k0 * ufl.inner(E, v) * ufl.dx)
L = ufl.inner(J, v) * ufl.dx

# PEC: n × E = 0 on boundary (tangential component zero)
def boundary(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, boundary)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(
    fem.Function(V), dofs)  # zero by default

problem = LinearProblem(
    a, L, bcs=[bc],
    petsc_options_prefix="maxwell_",
    petsc_options={{"ksp_type": "gmres",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps"}})
Eh = problem.solve()
print(f"||E||_L2 = {{np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(Eh, Eh) * ufl.dx))))}}")
'''
