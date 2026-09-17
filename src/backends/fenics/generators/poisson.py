"""Poisson equation generators for FEniCSx/dolfinx.

Variants: 2d, 3d, l_domain, rectangle
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['poisson'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Poisson equation -kappa * laplacian(u) = f solved with FEniCSx/dolfinx",
    "weak_form": "kappa * (grad(u), grad(v)) * dx = (f, v) * dx",
    "function_space": "Lagrange order 1 or 2",
    "solver": {"ksp_type": "preonly/cg", "pc_type": "lu/hypre"},
    "pitfalls": [
        "[API] In recent dolfinx, domain.topology.create_connectivity(fdim, tdim) is no longer required as a manual step before locate_entities_boundary / locate_dofs_topological — the connectivity is built lazily on first need. Calling it eagerly is harmless and is the safer default in tutorial code, but its ABSENCE no longer triggers an exception. Signal: in older dolfinx (pre-0.7), locate_dofs_topological raised RuntimeError mentioning 'connectivity has not been computed'; current dolfinx returns the correct dof indices without that step. (Verified empirically 2026-06-01 — locate_entities_boundary + locate_dofs_topological both succeed without an explicit create_connectivity call.)",
        "[API] Use dolfinx.default_scalar_type for Constants and Function arrays so the dtype matches the underlying PETSc build (float64 if PETSc is real, complex128 if PETSc is complex). Signal: in a COMPLEX build, a Python float reaches C++ as a float64 Constant and fem.form dies on the nanobind overload resolution — 'TypeError: __init__(): incompatible function arguments' listing only the Constant_complex128 signatures; the same mismatch through fem.petsc.assemble_matrix gives 'TypeError: create_matrix(): incompatible function arguments ... Invoked with types: dolfinx.cpp.fem.Form_float64'. In a REAL build the reverse direction is much quieter than this entry used to claim: 0j assigned ELEMENTWISE raises 'TypeError: float() argument must be a string or a real number, not \\'complex\\'' (numpy, not dolfinx), while a whole-array assignment f.x.array[:] = <complex array> and Function.interpolate of a complex callable do NOT raise at all — they drop the imaginary part behind a numpy ComplexWarning, and fem.Constant(msh, 0j) is accepted outright. The previously quoted \"ValueError 'cannot convert complex to float'\" is wrong in its exception type, wrong in its text, and wrong about the failure being loud. Assert the dtype, do not wait for a raise. (Verified by execution 2026-08-07 in both conda envs, dolfinx 0.10.0 / numpy 1.26.4.)",
        "[API] VTXWriter (ADIOS2 backend) supports only Lagrange / DG element families. Signal: constructing the writer on a Function defined on N1curl, RT or BDM raises 'RuntimeError: Only (discontinuous) Lagrange functions are supported. Interpolate Functions before output.' — and it raises in the VTXWriter CONSTRUCTOR, before any write() call, so a guard wrapped around write() alone never sees it. Both strings this entry used to quote — 'Cannot interpolate function ... to the VTX output basis' and 'ADIOS2 VTX only supports Lagrange elements' — are absent from dolfinx, ADIOS2 and every library FEniCSx links, and neither reproduces. Interpolate into a ('Lagrange', k, (gdim,)) space first, or use XDMFFile. (Verified by execution 2026-08-07 for all three element families, dolfinx 0.10.0 / ADIOS2 2.11.)",
    ],
    "materials": {"kappa": {"range": [0.001, 1e6], "unit": "W/(m*K) or dimensionless"}},
}

VARIANTS = ["2d", "3d", "l_domain", "rectangle"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Poisson variant."""
    generators = {
        "2d": _poisson_2d,
        "3d": _poisson_3d,
        "l_domain": _poisson_l_domain,
        "rectangle": _poisson_rectangle,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown Poisson variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    kappa = params.get("kappa", 1.0)
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    return f'''\
"""Poisson equation on unit square — FEniCSx/dolfinx
-kappa * laplacian(u) = f on [0,1]^2
u = 0 on boundary
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Boundary condition: u = 0 on all boundaries
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Source term
x = ufl.SpatialCoordinate(domain)
f = fem.Constant(domain, default_scalar_type(1.0))

# Weak form: kappa * (grad u, grad v) = (f, v)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
kappa = fem.Constant(domain, default_scalar_type({kappa}))
a = kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solve", petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
uh = problem.solve()
uh.name = "u"

# Output
with io.VTXWriter(domain.comm, "result.bp", [uh]) as vtx:
    vtx.write(0.0)

# Also write VTU for PyVista compatibility
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# Summary
u_array = uh.x.array
print(f"Poisson solved: min(u)={{u_array.min():.6e}}, max(u)={{u_array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


def _poisson_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    kappa = params.get("kappa", 1.0)
    nx = params.get("nx", 16)
    ny = params.get("ny", 16)
    nz = params.get("nz", 16)
    return f'''\
"""Poisson equation on unit cube — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

domain = mesh.create_unit_cube(MPI.COMM_WORLD, {nx}, {ny}, {nz}, mesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 1))

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

f = fem.Constant(domain, default_scalar_type(1.0))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
kappa = fem.Constant(domain, default_scalar_type({kappa}))
a = kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solve", petsc_options={{"ksp_type": "cg", "pc_type": "hypre"}})
uh = problem.solve()
uh.name = "u"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_array = uh.x.array
print(f"Poisson 3D solved: min(u)={{u_array.min():.6e}}, max(u)={{u_array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


def _poisson_l_domain(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script. Requires Gmsh.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    mesh_size = params.get("mesh_size", 0.03)
    return f'''\
"""Poisson on L-shaped domain — FEniCSx + Gmsh
Re-entrant corner singularity benchmark. Non-trivial geometry.
-\u0394u = 1 on L-domain, u = 0 on boundary.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

# Generate L-domain mesh with Gmsh
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("L-domain")
ms = {mesh_size}
# L-domain: [-1,1]^2 minus [0,1]x[-1,0] — matches deal.II hyper_L(-1,1)
p1 = gmsh.model.geo.addPoint(-1, -1, 0, ms)
p2 = gmsh.model.geo.addPoint(0, -1, 0, ms)
p3 = gmsh.model.geo.addPoint(0, 0, 0, ms * 0.3)  # fine at re-entrant corner
p4 = gmsh.model.geo.addPoint(1, 0, 0, ms)
p5 = gmsh.model.geo.addPoint(1, 1, 0, ms)
p6 = gmsh.model.geo.addPoint(-1, 1, 0, ms)
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p5)
l5 = gmsh.model.geo.addLine(p5, p6)
l6 = gmsh.model.geo.addLine(p6, p1)
cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
s = gmsh.model.geo.addPlaneSurface([cl])
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [s], tag=1)
gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4, l5, l6], tag=1)
gmsh.model.mesh.generate(2)
gmsh.write("l_domain.msh")
gmsh.finalize()

# Read mesh into dolfinx
from dolfinx.io.gmsh import read_from_msh
mesh_data = read_from_msh("l_domain.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh_data.mesh
V = fem.functionspace(domain, ("Lagrange", 1))

# BCs: u=0 on entire boundary
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type(1.0))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = LinearProblem(a, L, bcs=[bc],
    petsc_options_prefix="poisson", petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
uh = problem.solve()
uh.name = "u"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_arr = uh.x.array
print(f"L-domain Poisson: min(u)={{u_arr.min():.6e}}, max(u)={{u_arr.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
print(f"Cells: {{domain.topology.index_map(tdim).size_global}}")
'''


def _poisson_rectangle(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    nx = params.get("nx", 64)
    ny = params.get("ny", 32)
    return f'''\
"""Poisson on [{lx}x{ly}] rectangle — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [{lx}, {ly}]], [{nx}, {ny}], mesh.CellType.triangle
)
V = fem.functionspace(domain, ("Lagrange", 1))

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

f = fem.Constant(domain, default_scalar_type(1.0))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solve",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
uh = problem.solve()
uh.name = "u"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_array = uh.x.array
print(f"Poisson rectangle solved: min(u)={{u_array.min():.6e}}, max(u)={{u_array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''
