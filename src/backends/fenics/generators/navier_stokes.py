"""Navier-Stokes generators for FEniCSx/dolfinx.

Variants: 2d, 3d, channel_cylinder
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['navier_stokes'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Incompressible Navier-Stokes with Taylor-Hood elements (P2/P1)",
    "weak_form": "nu*(grad(u),grad(v))*dx + (grad(u)*u,v)*dx - p*div(v)*dx - q*div(u)*dx = (f,v)*dx",
    "function_space": "Mixed: P2 velocity + P1 pressure (Taylor-Hood, inf-sup stable)",
    "solver": "Newton iteration with LU (MUMPS) for direct solve",
    "pitfalls": [
        # Promoted from one-line tips to [Category] + Signal: pitfalls
        # (2026-06-02 audit pass 20). Each names a real dolfinx symptom.
        "[Numerical] Must use inf-sup stable element pair "
        "(Taylor-Hood P2/P1 — P2 velocity + P1 pressure). Equal-order "
        "P1/P1 violates LBB. Signal: a dolfinx mixed_element with "
        "matching basix.ufl.element('Lagrange', 1) on both velocity "
        "and pressure yields a checkerboard pressure pattern; the "
        "XDMFFile output shows oscillations between adjacent DOFs with "
        "amplitude ~ 100% of mean. (Audit 2026-06-02.)",
        "[Input] Pressure needs pinning (dirichletbc at one point) "
        "for enclosed flows — pressure is determined only up to a "
        "constant. Signal: the dolfinx NewtonSolver on a closed-domain "
        "NS problem without pressure pinning either reports "
        "DIVERGED_BREAKDOWN from KSPSolve or returns a Function with a "
        "huge additive offset (the pressure component drifts by O(1e6) "
        "between solver runs). (Audit 2026-06-02.)",
        "[Numerical] High Re requires stabilization (SUPG) or a "
        "finer mesh — the Galerkin form is unstable for "
        "convection-dominated NS. Signal: a Taylor-Hood mixed-element "
        "solve at Re > 500 with no SUPG produces visible wiggles in "
        "the velocity Function near the inflow / obstacle; the "
        "XDMFFile output shows oscillations growing with each Newton "
        "iteration. (Audit 2026-06-02.)",
        "[Numerical] Newton may not converge for Re > 500 without "
        "load-step continuation (ramp Re from 1 to target). Signal: "
        "the dolfinx NewtonSolver raises NoConvergence or the residual "
        "ratio oscillates between two values; ramping nu via "
        "fem.Constant(domain, nu_initial) then updating in a loop "
        "produces monotonic convergence. (Audit 2026-06-02.)",
        "[API] Use basix.ufl.element() and basix.ufl.mixed_element() "
        "in modern dolfinx — the old VectorElement/MixedElement from "
        "ufl alone is deprecated. Signal: passing ufl.MixedElement to "
        "fem.functionspace raises DeprecationWarning or "
        "AttributeError; the correct pattern is "
        "basix.ufl.mixed_element([P2_vec, P1]). (Audit 2026-06-02.)",
    ],
    "materials": {
        "Re": {"range": [1, 10000], "unit": "dimensionless (Reynolds number)"},
        "nu": {"range": [1e-6, 1.0], "unit": "m^2/s (kinematic viscosity = 1/Re)"},
    },
}

VARIANTS = ["2d", "3d", "channel_cylinder"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Navier-Stokes variant."""
    generators = {
        "2d": _navier_stokes_2d,
        "3d": _navier_stokes_cavity_3d,
        "channel_cylinder": _navier_stokes_channel_cylinder,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown Navier-Stokes variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    Re = params.get("Re", 100)
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    return f'''\
"""Incompressible Navier-Stokes — FEniCSx/dolfinx
Taylor-Hood P2/P1 elements, Newton iteration.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
gdim = domain.geometry.dim

# Taylor-Hood elements: P2 velocity + P1 pressure
P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

# Boundary conditions
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# No-slip on all walls
def walls(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

def lid(x):
    return np.isclose(x[1], 1.0)

# Velocity sub-space (must collapse for BC application)
V, V_map = W.sub(0).collapse()

# No-slip walls: use Function (not constant) for sub-space BC
noslip = fem.Function(V)
noslip.x.array[:] = 0.0
wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
bc_walls = fem.dirichletbc(noslip, wall_dofs, W.sub(0))

# Lid velocity u=(1,0)
lid_velocity = fem.Function(V)
lid_velocity.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[0])))
lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
lid_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
bc_lid = fem.dirichletbc(lid_velocity, lid_dofs, W.sub(0))

# Pressure pin (single point)
Q, Q_map = W.sub(1).collapse()
zero_p = fem.Function(Q)
zero_p.x.array[:] = 0.0
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
bc_pressure = fem.dirichletbc(zero_p, p_dofs, W.sub(1))

bcs = [bc_walls, bc_lid, bc_pressure]

# Weak form: Navier-Stokes
w = fem.Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)

nu = fem.Constant(domain, default_scalar_type(1.0 / {Re}))
f = fem.Constant(domain, default_scalar_type((0.0, 0.0)))

F = (
    nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
    - p * ufl.div(v) * ufl.dx
    - q * ufl.div(u) * ufl.dx
    - ufl.dot(f, v) * ufl.dx
)

# Newton solver
problem = NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-6, "snes_max_it": 50, "snes_monitor": None}})
problem.solve()
reason = problem.solver.getConvergedReason()
its = problem.solver.getIterationNumber()
print(f"Newton solver: {{its}} iterations, converged reason={{reason}}")

# Extract velocity and pressure
(u_sol, p_sol) = w.split()

# Interpolate P2 velocity to P1 for XDMF output
V_out = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_out = fem.Function(V_out, name="velocity")
u_out.interpolate(u_sol)

P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(p_sol)

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

# Statistics
u_arr = u_out.x.array.reshape(-1, gdim)
u_mag = np.linalg.norm(u_arr, axis=1)
print(f"Velocity: max |u| = {{u_mag.max():.6e}}")
print(f"Pressure: min(p) = {{p_out.x.array.min():.6e}}, max(p) = {{p_out.x.array.max():.6e}}")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''


def _navier_stokes_cavity_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    Re = params.get("Re", 100)
    n = params.get("n", 12)
    return f'''\
"""3D Lid-driven cavity — incompressible Navier-Stokes"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element
from petsc4py import PETSc

domain = mesh.create_unit_cube(MPI.COMM_WORLD, {n}, {n}, {n}, mesh.CellType.tetrahedron)
gdim = 3

P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

def walls(x):
    return (np.isclose(x[0], 0) | np.isclose(x[0], 1) |
            np.isclose(x[1], 0) |
            np.isclose(x[2], 0) | np.isclose(x[2], 1))

def lid(x):
    return np.isclose(x[1], 1.0)

V, _ = W.sub(0).collapse()
noslip = fem.Function(V)
noslip.x.array[:] = 0.0
wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
bc_walls = fem.dirichletbc(noslip, wall_dofs, W.sub(0))

lid_vel = fem.Function(V)
lid_vel.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
lid_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
bc_lid = fem.dirichletbc(lid_vel, lid_dofs, W.sub(0))

Q, _ = W.sub(1).collapse()
zero_p = fem.Function(Q)
zero_p.x.array[:] = 0.0
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0) & np.isclose(x[2], 0))
bc_p = fem.dirichletbc(zero_p, p_dofs, W.sub(1))
bcs = [bc_walls, bc_lid, bc_p]

w = fem.Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
nu = fem.Constant(domain, default_scalar_type(1.0 / {Re}))
f = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))

F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
     - p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx
     - ufl.dot(f, v) * ufl.dx)

problem = NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns3d",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-5, "snes_max_it": 30, "snes_monitor": None}})
problem.solve()
its = problem.solver.getIterationNumber()
print(f"Newton: {{its}} iterations")
(u_sol, p_sol) = w.split()

V_out = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_out = fem.Function(V_out, name="velocity")
u_out.interpolate(u_sol)
P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(p_sol)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

u_arr = u_out.x.array.reshape(-1, gdim)
print(f"max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''


def _navier_stokes_channel_cylinder(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script. Requires Gmsh.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    Re = params.get("Re", 20)
    mesh_size = params.get("mesh_size", 0.02)
    return f'''\
"""Navier-Stokes: channel flow around cylinder — FEniCSx + Gmsh
Parabolic inlet, no-slip walls, cylinder obstacle.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element

# Generate channel with cylinder using Gmsh
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("channel-cyl")
L, H = 2.2, 0.41
cx, cy, r = 0.2, 0.2, 0.05
rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
cyl = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
gmsh.model.occ.cut([(2, rect)], [(2, cyl)])
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(2)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=1)
curves = gmsh.model.getEntities(1)
for i, c in enumerate(curves):
    gmsh.model.addPhysicalGroup(1, [c[1]], tag=i+1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", {mesh_size})
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", {mesh_size} * 0.15)
gmsh.model.mesh.generate(2)
gmsh.write("channel_cyl.msh")
gmsh.finalize()

from dolfinx.io.gmsh import read_from_msh
mesh_data = read_from_msh("channel_cyl.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh_data.mesh
gdim = domain.geometry.dim

# Taylor-Hood P2/P1
P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# BCs
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

# No-slip on walls + cylinder
def walls(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], H)

def cylinder_surf(x):
    return ((x[0] - cx)**2 + (x[1] - cy)**2) < (r * 1.5)**2

def inlet(x):
    return np.isclose(x[0], 0.0)

noslip = fem.Function(V)
noslip.x.array[:] = 0.0

wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
bc_walls = fem.dirichletbc(noslip, wall_dofs, W.sub(0))

cyl_facets = mesh.locate_entities_boundary(domain, fdim, cylinder_surf)
cyl_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, cyl_facets)
bc_cyl = fem.dirichletbc(noslip, cyl_dofs, W.sub(0))

# Parabolic inlet: u_x = 4*U_m*y*(H-y)/H^2, u_y = 0
U_m = 0.3  # max inlet velocity
inlet_vel = fem.Function(V)
inlet_vel.interpolate(lambda x: (4 * U_m * x[1] * (H - x[1]) / H**2, np.zeros_like(x[0])))
inlet_facets = mesh.locate_entities_boundary(domain, fdim, inlet)
inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
bc_inlet = fem.dirichletbc(inlet_vel, inlet_dofs, W.sub(0))

# Pressure pin at outlet
def outlet_corner(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0.0, atol=0.05)
zero_p = fem.Function(Q)
zero_p.x.array[:] = 0.0
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), outlet_corner)
bc_p = fem.dirichletbc(zero_p, p_dofs, W.sub(1))

bcs = [bc_walls, bc_cyl, bc_inlet, bc_p]

# NS weak form
w = fem.Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
nu = fem.Constant(domain, default_scalar_type(U_m * 2 * r / {Re}))  # nu = U_m * D / Re
f = fem.Constant(domain, default_scalar_type((0.0, 0.0)))
F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
     - p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx
     - ufl.dot(f, v) * ufl.dx)

problem = NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-6, "snes_max_it": 50, "snes_monitor": None}})
problem.solve()
its = problem.solver.getIterationNumber()
print(f"Newton: {{its}} iterations")

(u_sol, p_sol) = w.split()
V_out = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_out = fem.Function(V_out, name="velocity")
u_out.interpolate(u_sol)
P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(p_sol)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

u_arr = u_out.x.array.reshape(-1, gdim)
print(f"max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"Re = {Re}, DOFs = {{W.dofmap.index_map.size_global}}")
'''
