"""NGSolve Navier-Stokes generators and knowledge."""


def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Incompressible Navier-Stokes with IMEX time-stepping."""
    Re = params.get("Re", 100)
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 1.0)
    nu = 1.0 / Re
    maxh = 1.0 / nx
    return f'''\
"""Navier-Stokes — IMEX time-stepping — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
V = VectorH1(mesh, order=2, dirichlet="bottom|right|top|left")
Q = H1(mesh, order=1)
X = V * Q
(u, p), (v, q) = X.TnT()

nu = {nu}
dt = {dt}
# Stokes part (implicit)
stokes = nu*InnerProduct(Grad(u), Grad(v))*dx + div(u)*q*dx + div(v)*p*dx
# Mass
mass = InnerProduct(u, v)*dx
mstar = BilinearForm(X)
mstar += mass + dt*stokes
mstar.Assemble()

gfu = GridFunction(X)
gfu.components[0].Set(CoefficientFunction((1, 0)), definedon=mesh.Boundaries("top"))

velocity = gfu.components[0]

inv = mstar.mat.Inverse(X.FreeDofs(), "umfpack")

t = 0.0
for step in range(int({T_end}/{dt})):
    # Convection (explicit)
    conv = LinearForm(X)
    conv += InnerProduct(Grad(velocity)*velocity, v)*dx
    conv.Assemble()

    rhs = LinearForm(X)
    rhs.Assemble()
    rhs.vec.data = mstar.mat * gfu.vec - dt*conv.vec
    gfu.vec.data = inv * rhs.vec
    t += dt

print(f"t={{t:.4f}}, Re={Re}")
vtk = VTKOutput(mesh, coefs=[gfu.components[0], gfu.components[1]],
                names=["velocity", "pressure"], filename="result", subdivision=1)
vtk.Do()
summary = {{"Re": {Re}, "time": t, "n_dofs": X.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Navier-Stokes solve complete.")
'''


KNOWLEDGE = {
    "navier_stokes": {
        "description": "Incompressible Navier-Stokes with IMEX (convection explicit, Stokes implicit)",
        "spaces": "VectorH1(order=2) * H1(order=1)",
        "solver": "IMEX: factor Stokes operator once, explicit convection each step",
        "pitfalls": [
            (
                "[Numerical] CFL condition for explicit "
                "convection: dt < h / max(velocity). Signal: "
                "the VectorH1 GridFunction velocity field "
                "reaches NaN within the first 10 steps when "
                "violation_ratio dt*max|u|/h > ~0.5; per-step "
                "max_u of the gfu diverges geometrically. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Convection term: Grad(velocity)*"
                "velocity for standard, or skew-symmetric "
                "form. Signal: in a closed periodic box the "
                "non_conservative form of the BilinearForm "
                "drifts the total kinetic_energy by ~1% over "
                "1000 steps; the skew_symmetric variant of "
                "the GridFunction velocity preserves it to "
                "machine precision. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Re > ~500 needs finer mesh or "
                "stabilization (SUPG). Signal: spurious "
                "wiggles in the GridFunction velocity "
                "upstream of obstacles; energy spectrum has "
                "high-frequency content not present in a "
                "reference DNS; on Schafer-Turek drag "
                "coefficient computed via BilinearForm "
                "boundary integrals differs > 10% from the "
                "5.57 (Re=20) / 3.22 (Re=100) reference. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Validation] Benchmark: Schafer-Turek DFG "
                "channel with cylinder at Re=20 (steady) and "
                "Re=100 (periodic vortex shedding). Signal: a "
                "Taylor-Hood VectorH1 + H1 implementation, "
                "with the drag/lift integrals computed via "
                "Integrate over the BoundaryFromVolumeCF / "
                "BND patch on the cylinder, should produce "
                "drag coefficient Cd around 5.57 and lift Cl "
                "in [0.0104, 0.0110] (Schafer-Turek published "
                "bounds); values outside this envelope "
                "expose either pressure-pin / boundary-"
                "condition errors or insufficient mesh "
                "refinement around the cylinder. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Validation] The unsteady DFG 2D-2 case "
                "(Re=100) published set, which the Re=20 "
                "entry above does not cover: Cd_max 3.22-"
                "3.24, Cl_max 0.98-1.02, Strouhal 0.295-"
                "0.305, pressure difference across the "
                "cylinder 2.46-2.50. Umax=1.5 gives "
                "Ubar=1.0 and Re = Ubar*D/nu = 100. "
                "(Measured 2026-09-16.)"
            ),
            (
                "[Validation] TWO DIFFERENT STROUHAL NUMBERS "
                "EXIST FOR 'Re=100 flow past a cylinder' and "
                "picking the wrong one makes a correct run "
                "look wrong. The confined DFG channel "
                "(2.2 x 0.41, blockage 0.1/0.41) sheds at St "
                "around 0.30. An UNBOUNDED cylinder at the "
                "same Re sheds at St around 0.164 "
                "(Williamson 1996). Signal: a measured St "
                "near 0.30 reported as a 45% error against "
                "0.164, or vice versa. Check which geometry "
                "the reference belongs to before comparing. "
                "(Measured 2026-09-16.)"
            ),
            (
                "[Numerical] ON DFG 2D-2 THE STROUHAL NUMBER "
                "CONVERGES MUCH FASTER THAN THE DRAG, so one "
                "quantity landing inside its published band "
                "is not evidence the run is resolved. "
                "Measured on this install with Taylor-Hood "
                "P2/P1 and IMEX at three refinements: "
                "NDOF 4172 -> St 0.2695, Cd_max 2.13; "
                "NDOF 11611 -> St 0.2950 (already inside the "
                "published 0.295-0.305), Cd_max 2.39, still "
                "26% below the published 3.22. Signal: a "
                "report that cites St agreement as proof of "
                "mesh independence. Check every reported "
                "quantity against refinement, not one. "
                "(Measured 2026-09-16.)"
            ),
            (
                "[Numerical] AN EXPLICIT-CONVECTION IMEX STEP "
                "THAT IS STABLE ON A COARSE MESH DIVERGES "
                "WHEN THE MESH IS REFINED, because the CFL "
                "limit follows the smallest cell. Measured "
                "on this install: dt=1e-3 runs to completion "
                "at cylinder spacing 0.01 and blows up at "
                "t=1.86 at spacing 0.005. Signal: a run that "
                "worked at one refinement returning NaN or "
                "non-finite forces at the next. Reduce dt "
                "with the mesh; 4e-4 was stable at 0.005. "
                "(Measured 2026-09-16.)"
            ),
        ],
    },
}

GENERATORS = {
    "navier_stokes_2d": _navier_stokes_2d,
}
