"""NGSolve coupled thermal-structural generators and knowledge."""


def _thermal_structural_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Coupled thermal-structural: heat -> elasticity with thermal strain."""
    E = params.get("E", 200e3)
    nu = params.get("nu", 0.3)
    alpha = params.get("alpha", 12e-6)
    T_hot = params.get("T_hot", 100.0)
    T_cold = params.get("T_cold", 0.0)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Coupled thermal-structural: heat -> thermal expansion — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh=0.03))

# Step 1: Heat conduction
V_T = H1(mesh, order=2, dirichlet="left|right")
uT, vT = V_T.TnT()
aT = BilinearForm(grad(uT)*grad(vT)*dx).Assemble()
# LinearForm(0*vT*dx) collapses symbolically before
# construction, so the resulting form has no TestFunction
# and Assemble() raises NgException 'Linearform must have
# TestFunction'. Build with no integrand and Assemble()
# directly — matches the fix shipped for ngsolve::heat.
fT = LinearForm(V_T); fT.Assemble()
gfT = GridFunction(V_T)
gfT.Set(CoefficientFunction({T_hot}), definedon=mesh.Boundaries("left"))
gfT.Set(CoefficientFunction({T_cold}), definedon=mesh.Boundaries("right"))
fT.vec.data -= aT.mat * gfT.vec
gfT.vec.data += aT.mat.Inverse(V_T.FreeDofs()) * fT.vec
# gfT.name is read-only in this NGSolve build (property
# has no setter). Pass the name string to VTKOutput or
# similar consumers instead of attempting to assign.
print(f"Temperature: [{{min(gfT.vec):.2f}}, {{max(gfT.vec):.2f}}]")

# Step 2: Elasticity with thermal strain
V_u = VectorH1(mesh, order=2, dirichlet="left")
u, v = V_u.TnT()
mu, lam, alpha = {mu}, {lam}, {alpha}
def Strain(u): return 0.5*(Grad(u) + Grad(u).trans)
# Thermal strain
eps_th = alpha * gfT * Id(2)
a_u = BilinearForm(InnerProduct(2*mu*Strain(u) + lam*Trace(Strain(u))*Id(2), Strain(v))*dx).Assemble()
f_u = LinearForm(InnerProduct((3*lam+2*mu)*alpha*gfT*Id(2), Strain(v))*dx).Assemble()

gfu = GridFunction(V_u)
gfu.vec.data = a_u.mat.Inverse(V_u.FreeDofs()) * f_u.vec
# gfu.name property has no setter — see pitfall in the
# heat template; supply the name via the consumer (e.g.
# VTKOutput(names=[...])) instead.

disp_arr = [gfu.components[0](mesh(1,0.5)), gfu.components[1](mesh(1,0.5))]
print(f"Displacement at (1,0.5): u_x={{disp_arr[0]:.6e}}, u_y={{disp_arr[1]:.6e}}")

vtk = VTKOutput(mesh, coefs=[gfT, gfu], names=["temperature", "displacement"],
                filename="result", subdivision=1)
vtk.Do()
summary = {{
    "T_min": float(min(gfT.vec)), "T_max": float(max(gfT.vec)),
    "disp_x_at_tip": disp_arr[0], "disp_y_at_tip": disp_arr[1],
    "n_dofs_thermal": V_T.ndof, "n_dofs_structural": V_u.ndof,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Coupled thermal-structural analysis complete.")
'''


KNOWLEDGE = {
    "thermal_structural": {
        "description": "Coupled thermal-structural: sequential heat -> elasticity with thermal strain",
        "spaces": "H1 (thermal) + VectorH1 (structural)",
        "solver": "Two sequential solves (one-way coupling)",
        "pitfalls": [
            (
                "[Numerical] Thermal strain eps_th = alpha * T * "
                "Id(dim) is isotropic — equal expansion in all "
                "directions. Applying alpha as a non-Id tensor "
                "(e.g. CF((1,0,0,0), dims=(2,2)), expansion only "
                "along x) makes a freely heated specimen expand "
                "along x and CONTRACT transversely. Diagnose it on "
                "the STRAIN, not the stress: 'zero Stress' does "
                "NOT discriminate, because a spatially uniform "
                "eigenstrain on a free body is always compatible, "
                "so the residual stress relaxes to round-off for "
                "the correct Id tensor and the wrong anisotropic "
                "one alike. Signal: with rigid modes removed (a "
                "NumberSpace multiplier) and no traction applied, "
                "Integrate the components of Strain(gfu) over a "
                "uniformly heated free specimen — the Id(dim) "
                "form gives eps_xx = eps_yy equal to the "
                "closed-form free expansion for the plane state "
                "you are in, while the x-only tensor gives an "
                "eps_xx well above it and a NEGATIVE eps_yy. The "
                "stress norm is round-off in BOTH cases, so a "
                "stress-based check passes the bug. (Verified "
                "2026-08-03 on NGSolve 6.2.2604 — Tier-2 fixture "
                "thermal_structural_isotropic_eps_th.)"
            ),
            (
                "[API] GAP FILLED 2026-08-03: GridFunction.Set() "
                "is a WHOLE-VECTOR overwrite, not an "
                "accumulate-into-a-region. Every call zeroes all "
                "DOFs outside its `definedon` region first, so "
                "the common two-call idiom "
                "gfT.Set(100.0, definedon=mesh.Boundaries('left')); "
                "gfT.Set(0.0, definedon=mesh.Boundaries('right')) "
                "silently discards the 100.0. Signal: measured on "
                "unit_square maxh=0.3, H1(order=2, "
                "dirichlet='left|right'): after the first Set the "
                "vector spans [-5.5e-13, 100.0]; after the second "
                "it is exactly [0.0, 0.0]. Swapping the call "
                "order just moves which value survives. Fix: one "
                "Set with a single CoefficientFunction covering "
                "the whole Dirichlet region, e.g. "
                "gfT.Set(IfPos(0.5-x, 100.0, 0.0), "
                "definedon=mesh.Boundaries('left|right')), which "
                "gives the intended [-5.5e-13, 100.0]. This is "
                "what makes the shipped thermal_structural_2d "
                "template print 'Temperature: [0.00, 0.00]' and "
                "a zero displacement while still exiting rc=0. "
                "(Verified empirically 2026-08-03 on NGSolve "
                "6.2.2604.)"
            ),
            (
                "[Numerical] RHS for elasticity: "
                "(3*lam + 2*mu)*alpha*T * Id(dim) contracted "
                "with Strain(v) — the factor is 3K, not 2*mu. "
                "Signal: verified on unit_square, VectorH1(order=2), "
                "rigid modes removed with a 3-component "
                "NumberSpace, E=210 GPa, nu=0.3, alpha=1.2e-5, "
                "dT=100: the (3*lam+2*mu) RHS reproduces the "
                "exact PLANE-STRAIN free expansion "
                "eps_xx = eps_yy = (1+nu)*alpha*dT = 1.560000e-03 "
                "to 6 digits, while the 2*mu RHS gives "
                "4.800000e-04 (0.31x). CORRECTION to the prior "
                "text: the error factor (3*lam+2*mu)/(2*mu) = "
                "(1+nu)/(1-2*nu) is only 2.50 / 3.25 / 4.50 at "
                "nu = 0.25 / 0.30 / 0.35 — i.e. ~3x for typical "
                "metals, NOT '~30-100x'. It reaches 14.5 only at "
                "nu=0.45 and 749.5 at nu=0.499, so the large "
                "numbers belong to rubber-like nearly-"
                "incompressible materials, not metals. (Verified "
                "empirically 2026-08-03 on NGSolve 6.2.2604 — "
                "catalog-drift correction.)"
            ),
            (
                "[Numerical] For two-way coupling: iterate "
                "thermal -> structural -> thermal until both "
                "GridFunction fields converge (typically 3-10 "
                "Picard iterations of BilinearForm assembly). "
                "Signal: doing only the FIRST thermal Solve and "
                "the FIRST structural Solve (one-way) gives a "
                "GridFunction deformation that does not feed "
                "back into the thermal conductivity Coefficient"
                "Function — for problems where deformation "
                "changes effective k(u) or contact-area heat "
                "transfer, the one-way result is wrong by "
                "5-20% on the temperature distribution. Track "
                "||T_new - T_old|| / ||T_new|| and ||u_new - "
                "u_old|| / ||u_new|| < 1e-4 to stop. (Audit "
                "2026-06-02.)"
            ),
            "[Syntax] Symbolic-zero RHS: LinearForm(0*v*dx) "
            "collapses before construction, leaving a form with "
            "no TestFunction. Assemble() then raises NgException "
            "'Linearform must have TestFunction'. Use the no-"
            "integrand constructor LinearForm(V); f.Assemble() "
            "to build an empty RHS — matches the fix shipped for "
            "ngsolve::heat. Signal: NgException text 'Linearform "
            "must have TestFunction' emitted from LinearForm."
            "Assemble. (Verified empirically 2026-06-01 — "
            "Layer F catch.)",
            "[API] GridFunction.name is a read-only property in "
            "current NGSolve builds — there is no name setter. "
            "Code like gfu.name = 'displacement' raises "
            "AttributeError 'property of GridFunction object "
            "has no setter'. Pass the name string to the "
            "consumer instead (e.g. VTKOutput(names=['u'])). "
            "Signal: AttributeError with the literal text "
            "'property of \\'GridFunction\\' object has no "
            "setter' from a direct gfu.name = '...' assignment. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
        ],
    },
}

GENERATORS = {
    "thermal_structural_2d": _thermal_structural_2d,
}
