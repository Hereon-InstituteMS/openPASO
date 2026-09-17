"""DUNE-fem reaction-diffusion generators and knowledge."""


def _reaction_diffusion_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Reaction-diffusion with time-stepping — DUNE-fem."""
    nx = params.get("nx", 64)
    dt = params.get("dt", 0.01)
    T_end = params.get("T_end", 1.0)
    return f'''\
"""Reaction-diffusion: Fisher-KPP equation — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx, conditional, lt
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)

u_n = space.interpolate(conditional(lt((x[0]-0.5)**2 + (x[1]-0.5)**2, 0.04), 1.0, 0.0), name="u")

u = TrialFunction(space)
v = TestFunction(space)
dt = {dt}
D = 0.01

# Implicit Euler: (u - u_n)/dt - D*laplacian(u) + u*(u-1) = 0
# Linearized: (u/dt + D*grad(u)*grad(v)) = u_n/dt + source
a = (u * v / dt + D * dot(grad(u), grad(v))) * dx
b = (u_n * v / dt + u_n * (1.0 - u_n) * v) * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")

t = 0.0
for step in range(int({T_end}/{dt})):
    scheme.solve(target=u_n)
    t += dt

vals = np.array(u_n.as_numpy)
print(f"Reaction-diffusion: t={{t:.4f}}, max={{vals.max():.6f}}")
gridView.writeVTK("result", pointdata={{"concentration": u_n}})
summary = {{"time": t, "max_value": float(vals.max()), "n_dofs": len(vals)}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "reaction_diffusion": {
        "description": "Reaction-diffusion (Fisher-KPP and other reaction-diffusion systems)",
        "solver": "Implicit Euler time-stepping with Newton linearization",
        "pitfalls": [
            (
                "[API] Nonlinear reaction term is "
                "AUTOMATICALLY linearised by the dune.fem "
                "galerkin scheme's internal Newton solver. "
                "Signal: writing the form with explicit "
                "u^n+1 inside R(u) (e.g. R(u) = u*(1-u)) "
                "is accepted by the dune.fem lagrange-"
                "space galerkin scheme; UFL automatic "
                "differentiation builds the Jacobian and "
                "Newton iterates per time step. Manual "
                "linearisation is not needed. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] For STIFF reactions: use "
                "IMPLICIT time stepping (backward Euler or "
                "DIRK). Signal: explicit time stepping at "
                "dt > 2/lambda_max (where lambda_max ~ "
                "reaction rate) gives NaN within ~10 "
                "steps; for Da > 100 the explicit dt is "
                "infeasibly small. Switch to an implicit "
                "scheme via the time-stepper selection. The "
                "accepted names are exactly ImplicitEuler, "
                "CrankNicolson, DIRK23, DIRK34 and SDIRK22 "
                "-- the std::string array in dune/fem/solver/"
                "rungekutta/timestepcontrol.hh:154. There is "
                "no DIRK22 scheme; an earlier version of this "
                "entry offered it beside SDIRK22 and it "
                "appears in no dune header. (Audit "
                "2026-06-02; names read from the installed "
                "headers 2026-08-09.)"
            ),
            (
                "[API] Multi-component systems (e.g. "
                "spiral waves) require 2+ COUPLED fields. "
                "Signal: a Schnakenberg / Gray-Scott "
                "implementation written as a single "
                "scalar problem ignores the cross-"
                "diffusion; use lagrange(gridView, order=k"
                ", dimRange=2) for a 2-species mixed "
                "space, with the reaction terms coupling "
                "the components inside the UFL form. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "reaction_diffusion_2d": _reaction_diffusion_2d,
}
