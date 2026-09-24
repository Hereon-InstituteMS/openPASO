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
                "reference DNS; on a channel-with-cylinder "
                "case the drag coefficient computed via "
                "BilinearForm boundary integrals differs by "
                "more than 10% from the published reference "
                "YOU RETRIEVED for that geometry and Re. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Validation] HOW TO TAKE DRAG AND LIFT ON A "
                "CYLINDER IN NGSOLVE: with a Taylor-Hood "
                "VectorH1 + H1 discretisation, integrate via "
                "Integrate over the BoundaryFromVolumeCF / "
                "BND patch on the cylinder -- getting the "
                "patch wrong is the usual reason a converged "
                "solve reports a wrong force. Retrieve the "
                "published envelope for your geometry and Re "
                "and compare against that; a value outside it "
                "exposes a pressure-pin error, a boundary-"
                "condition error, or insufficient mesh "
                "refinement around the cylinder, in that "
                "order of likelihood. The reference numbers "
                "are deliberately not reproduced here: they "
                "are the answer, and an answer is only "
                "evidence when you fetched it yourself and "
                "can say where it came from. Signal: a force that is "
                "smooth, steady and plainly wrong -- taking the integral "
                "over the wrong BND patch gives a clean number from the "
                "wrong surface, with nothing in the solve to flag it. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] THE DFG CYLINDER CASES ARE "
                "NON-DIMENSIONALISED ON THE MEAN INLET "
                "SPEED, NOT THE PEAK, and getting that wrong "
                "puts you at the wrong Reynolds number while "
                "every line of your setup looks right. "
                "THE MEAN-TO-PEAK FACTOR IS NOT THE SAME IN "
                "2-D AND 3-D, which is the trap inside the "
                "trap. For the 2-D parabolic inlet "
                "4*Um*y*(H-y)/H^2 the mean is TWO THIRDS of "
                "the peak, so a peak of 1.5 gives a mean of "
                "1.0. For the 3-D inlet "
                "16*Um*y*z*(H-y)*(H-z)/H^4 used by the 3D-1Z "
                "and 3D-2Z cases the peak is still Um but the "
                "mean is FOUR NINTHS of it, not two thirds -- "
                "integrate the profile over the cross-section "
                "and see. Carrying the 2-D factor into a 3-D "
                "case puts you 50% off in Re. Re = Ubar*D/nu "
                "with D the cylinder diameter. Check which convention "
                "the source you are comparing against used "
                "before you conclude anything from a "
                "mismatch. Signal: a run whose forces are wrong by a "
                "consistent factor with no other symptom -- the peak/mean "
                "confusion is a factor of 3/2 in the speed and so 9/4 in "
                "anything quadratic in it."
            ),
            (
                "[Validation] A REFERENCE VALUE BELONGS TO A "
                "GEOMETRY, AND 'FLOW PAST A CYLINDER AT "
                "Re=100' NAMES AT LEAST TWO DIFFERENT "
                "PROBLEMS. A cylinder confined in a channel "
                "and a cylinder in an unbounded stream shed "
                "at substantially different Strouhal "
                "numbers at the same Reynolds number, "
                "because blockage changes the wake. Signal: "
                "a correct run reported as a large error, or "
                "a wrong run reported as agreement, because "
                "the value it was compared against came from "
                "the other configuration. Before you compare "
                "anything, establish WHICH geometry your "
                "retrieved reference describes -- its "
                "blockage ratio, and whether the walls are "
                "there at all -- and say so when you report "
                "the comparison."
            ),
            (
                "[Numerical] ON DFG 2D-2 THE STROUHAL NUMBER "
                "CONVERGES MUCH FASTER THAN THE DRAG, so one "
                "quantity landing inside its published band "
                "is not evidence the run is resolved. "
                "Measured on this install with Taylor-Hood "
                "P2/P1 and IMEX: across a refinement "
                "sequence the shedding frequency entered its "
                "published band while the drag was still "
                "more than twenty per cent short of its own, "
                "and kept moving under further refinement. "
                "Signal: a report that cites agreement on "
                "one quantity as proof of mesh independence. "
                "Run your own ladder and check EVERY "
                "quantity you intend to report against "
                "refinement, not the one that agreed first. "
                "(Measured 2026-09-16; the numbers are not "
                "reproduced because a resolution study is "
                "only evidence when the numbers are yours.)"
            ),
            (
                "[Numerical] AN EXPLICIT-CONVECTION IMEX STEP "
                "THAT IS STABLE ON A COARSE MESH DIVERGES "
                "WHEN THE MESH IS REFINED, because the CFL "
                "limit follows the smallest cell, which on "
                "a graded mesh is the one hugging the "
                "obstacle and not the one you sized. "
                "Measured on this install: halving the "
                "spacing around the cylinder turned a run "
                "that completed into one that blew up "
                "part-way through, at an unchanged time "
                "step. Signal: a run that worked at one "
                "refinement returning NaN or non-finite "
                "forces at the next. Reduce dt with the "
                "mesh -- roughly in proportion to the "
                "smallest cell -- or use an implicit "
                "treatment of the convective term. "
                "(Measured 2026-09-16.)"
            ),
        ],
    },
}

GENERATORS = {
    "navier_stokes_2d": _navier_stokes_2d,
}
