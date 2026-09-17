"""DUNE-fem steady SIPG advection-diffusion generator and knowledge."""


def _dg_advection_diffusion_2d(params: dict) -> str:
    """Return a complete generic SIPG solve on a triangular ALUGrid."""
    nx = int(params.get("nx", 8))
    ny = int(params.get("ny", nx))
    degree = int(params.get("order", 2))
    epsilon = float(params.get("diffusion", 1.0))
    velocity_x = float(params.get("bx", 2.0))
    velocity_y = float(params.get("by", 1.0))
    beta = float(params.get("beta", 20.0))
    source = float(params.get("source", 1.0))
    return f'''\
"""Steady SIPG advection-diffusion on triangles -- DUNE-fem.

Solves -epsilon*laplace(u) + velocity . grad(u) = source with u=0
imposed weakly by SIPG/Nitsche boundary terms. Adapt ``source_expr`` and
``boundary_value`` for the problem at hand; do not add a strong DirichletBC to
this discontinuous space.
"""
import json
import numpy as np
from dune.alugrid import aluConformGrid
from dune.fem.scheme import galerkin
from dune.fem.space import dglagrange
from dune.fem.utility import pointSample
from dune.grid import cartesianDomain
from dune.ufl import Constant, DuneCellDiameter as CellDiameter
from ufl import (FacetNormal, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, avg, dS, dot, ds, dx, grad, inner, jump)

nx, ny = {nx}, {ny}
degree = {degree}
epsilon = {epsilon!r}
velocity = as_vector([{velocity_x!r}, {velocity_y!r}])
beta = {beta!r}

# ALUGrid triangulates every logical rectangle: exactly 2*nx*ny triangles.
grid = aluConformGrid(
    cartesianDomain([0, 0], [1, 1], [nx, ny]), dimgrid=2)
if grid.size(0) != 2 * nx * ny:
    raise RuntimeError(
        f"expected {{2*nx*ny}} triangles, got {{grid.size(0)}}")
space = dglagrange(grid, order=degree)

u = TrialFunction(space)
v = TestFunction(space)
x = SpatialCoordinate(space)
normal = FacetNormal(space)
h = CellDiameter(space)
h_avg = (h("+") + h("-")) / 2.0

# Replace this scalar by the task's UFL expression in x[0], x[1].
source_expr = Constant({source!r}, name="source")
boundary_value = Constant(0.0, name="boundary_value")

normal_velocity = dot(velocity, normal)
upwind = (
    (normal_velocity("+") + abs(normal_velocity("+"))) / 2.0 * u("+")
    + (normal_velocity("+") - abs(normal_velocity("+"))) / 2.0 * u("-"))

# Symmetric interior-penalty diffusion. The requested convention is exactly
# beta*p*p/h, where p is the polynomial degree (not degree+1).
interior_diffusion = (
    epsilon * inner(grad(u), grad(v)) * dx
    - epsilon * inner(avg(grad(u)), jump(v, normal)) * dS
    - epsilon * inner(jump(u, normal), avg(grad(v))) * dS
    + beta * degree * degree / h_avg * epsilon
      * inner(jump(u, normal), jump(v, normal)) * dS)

# Conservative upwind form for divergence-free, constant velocity.
advection = (
    -inner(u * velocity, grad(v)) * dx
    + upwind * jump(v) * dS
    + (normal_velocity + abs(normal_velocity)) / 2.0 * u * v * ds)

# Weak Dirichlet data for diffusion. For nonzero boundary data, retain these
# operator terms and add the matching linear terms shown below.
boundary_diffusion = (
    -epsilon * dot(grad(u), normal) * v * ds
    -epsilon * dot(grad(v), normal) * u * ds
    + beta * degree * degree / h * epsilon * u * v * ds)
boundary_rhs = (
    -epsilon * dot(grad(v), normal) * boundary_value * ds
    + beta * degree * degree / h * epsilon * boundary_value * v * ds
    -(normal_velocity - abs(normal_velocity)) / 2.0
      * boundary_value * v * ds)

scheme = galerkin(
    [interior_diffusion + advection + boundary_diffusion
     == source_expr * v * dx + boundary_rhs],
    solver="gmres",
    parameters={{"linear.tolerance": 1e-10,
                "linear.maxiterations": 20000,
                "linear.preconditioning.method": "ssor",
                # DUNE is near-silent once the JIT cache is warm; without
                # this the run's captured log carries no solver output at
                # all on a re-run (measured: first run logs the JIT build,
                # second run ~nothing).
                "linear.verbose": True}})
solution = space.interpolate(0, name="u")
info = scheme.solve(target=solution)
values = np.asarray(solution.as_numpy)
if not info["converged"]:
    raise RuntimeError(f"DUNE linear solve did not converge: {{info}}")
if not np.isfinite(values).all():
    raise RuntimeError("DUNE returned non-finite solution coefficients")

center = float(pointSample(solution, [0.5, 0.5]))
print("DUNE_SIPG_OK")
print(f"NDOF = {{int(space.size)}}")
print(f"triangles = {{grid.size(0)}}")
print(f"linear_iterations = {{info.get('linear_iterations')}}")
print(f"u(0.5,0.5) = {{center:.12e}}")
grid.writeVTK("result", pointdata={{"u": solution}})
with open("results_summary.json", "w") as stream:
    json.dump({{"ndof": int(space.size), "triangles": int(grid.size(0)),
               "center": center, "converged": bool(info["converged"]),
               "degree": degree, "beta": beta}}, stream, indent=2)
# TERMINAL SENTINEL, last line, after every other print.
#
# Every other DUNE generator in this backend ends with it and a suite-wide test
# requires it, because a DUNE run can exit 134 AFTER printing every correct
# result — so "the numbers looked right" does not establish that the template
# finished. A bespoke marker earlier in the output cannot carry that meaning:
# it only says the solve reached that line, which is why the check is on the
# LAST line and why this file failed the contract while passing its own test.
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "dg_advection_diffusion": {
        "description": (
            "Steady advection-diffusion with discontinuous Lagrange elements, "
            "upwind advection, symmetric interior-penalty diffusion, and "
            "weak SIPG/Nitsche Dirichlet boundary data on ALUGrid triangles."),
        "discretisation": (
            "dglagrange(aluConformGrid(cartesianDomain(...), dimgrid=2), "
            "order=p); the ALUGrid has 2*N*N triangles for N cells per side."),
        "penalty": (
            "Use the convention the problem states. For beta*p*p/h, use the "
            "polynomial degree p itself, not p+1, on both interior and boundary "
            "facets."),
        "required_api": (
            "Use dune.ufl.DuneCellDiameter, not ufl.CellDiameter: current "
            "DUNE rejects the latter because its historical value was wrong."),
        "boundary_conditions": (
            "A strong DirichletBC is not the requested method on a DG space. "
            "The template includes the symmetric consistency and penalty terms "
            "on ds, plus the upwind inflow contribution."),
        "probe_evaluation": (
            "dune.fem.utility.pointSample(solution, [x, y]) evaluates the "
            "computed DG field at a prescribed off-node point."),
        "pitfalls": [
            "[API] Import DuneCellDiameter from dune.ufl. Signal: importing "
            "CellDiameter from ufl reaches model generation and then raises a "
            "TypeError explicitly asking for DuneCellDiameter.",
            "[Numerical] Keep both SIPG consistency terms and the penalty on "
            "interior facets, and the matching Nitsche terms on the boundary. "
            "Signal: the solve converges but the boundary defect stalls under "
            "refinement when the boundary block is absent.",
            "[Numerical] Advection makes the matrix nonsymmetric; use GMRES, "
            "not CG. Signal: CG stagnates or reports failure on an otherwise "
            "well-posed form.",
            "[Performance] Build one scheme per polynomial form and reuse it "
            "across mesh resolutions where possible; each unseen DUNE form "
            "JIT-compiles C++ before the first solve. Signal: DUNE prints "
            "'Compiling <module> (new)', where <module> is the generated "
            "module name, and the first level then takes minutes while later "
            "resolutions of the same form start without another compile. The "
            "wording is f\"Compiling {pythonName} (new)\" at "
            "dune/generator/cmakebuilder.py:378, with (updated), (rebuilding) "
            "and (loading) as its siblings -- so match on 'Compiling' and "
            "'(new)', not on a fixed module name.",
        ],
    }
}


GENERATORS = {
    "dg_advection_diffusion_2d": _dg_advection_diffusion_2d,
}