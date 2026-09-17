"""NGSolve mixed Poisson generators and knowledge."""


def _mixed_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Mixed Poisson with H(div)/L2 (Raviart-Thomas + piecewise constants)."""
    order = params.get("order", 1)
    return f'''\
"""Mixed Poisson: sigma = -grad(u), div(sigma) = f — HDiv/L2 — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))

# Raviart-Thomas for flux, L2 for scalar.
# CRITICAL PAIRING (see pitfall #0): RT_k must be paired with
# L2(order=k), NOT L2(order=k-1). HDiv(order=k, RT=True) x
# L2(order=k-1) assembles and solves without any error but does
# NOT converge under refinement. If you prefer the L2(order=k-1)
# partner, drop RT=True (that is BDM_k, which is the stable
# partner for L2(k-1)).
V = HDiv(mesh, order={order}, RT=True)
Q = L2(mesh, order={order})
X = V * Q
(sigma, u), (tau, v) = X.TnT()

n = specialcf.normal(2)
a = BilinearForm(X)
a += (sigma*tau + div(sigma)*v + div(tau)*u)*dx
a.Assemble()

f = LinearForm(X)
f += -1*v*dx  # source f=1
f.Assemble()

gfu = GridFunction(X)
gfu.vec.data = a.mat.Inverse(X.FreeDofs()) * f.vec

flux = gfu.components[0]
scalar = gfu.components[1]
max_u = Integrate(scalar, mesh) / Integrate(1, mesh)  # mean value
print(f"Mean u: {{max_u:.6f}}")

vtk = VTKOutput(mesh, coefs=[flux, scalar], names=["flux", "potential"],
                filename="result", subdivision=0)
vtk.Do()
summary = {{"n_dofs": X.ndof, "mean_value": float(max_u)}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Mixed Poisson solve complete.")
'''


KNOWLEDGE = {
    "mixed_poisson": {
        "description": "Mixed Poisson with H(div)/L2 (Raviart-Thomas flux recovery)",
        "spaces": (
            "HDiv(mesh, order=k, RT=True) * L2(mesh, order=k)  "
            "[Raviart-Thomas RT_k]  OR  "
            "HDiv(mesh, order=k) * L2(mesh, order=k-1)  "
            "[BDM_k, the RT=False default]. Do NOT cross the two "
            "pairings — see pitfall #0."
        ),
        "solver": "Direct (saddle-point) or iterative with Schur complement",
        "pitfalls": [
            (
                "[Numerical] RT=True gives Raviart-Thomas RT_k; "
                "RT=False (the DEFAULT) gives BDM_k. They need "
                "DIFFERENT L2 partners and crossing them is the "
                "single worst trap in this physics: "
                "HDiv(order=k, RT=True) * L2(order=k-1) "
                "assembles, factorises and returns a plausible "
                "answer that DOES NOT CONVERGE, while "
                "HDiv(order=k, RT=False) * L2(order=k) is "
                "exactly singular and raises. Measured on "
                "unit_square for -Lap u = 1, u=0 (exact "
                "mean(u) = 0.03514425): the RT_1 x L2(0) pairing "
                "gives mean(u) = 0.0101944 / 0.0090554 / "
                "0.0088825 / 0.0088152 at maxh = 0.2 / 0.1 / "
                "0.05 / 0.025 — a 75% error that gets slightly "
                "WORSE under refinement (MMS L2 rates -0.04, "
                "-0.01 for both flux and u). The correct RT_1 x "
                "L2(1) pairing gives 0.0351921 / 0.0351476 / "
                "0.0351445 (errors 4.8e-05 / 3.3e-06 / 2.4e-07), "
                "and BDM order 1 -- HDiv(order=1, RT=False) -- "
                "x L2(0) gives the identical numbers. "
                "MMS rates with correct pairings: RT_k x L2(k) -> "
                "flux and u both k+1 (measured 1.96/2.09 and "
                "2.11/2.17 at k=1; 3.05/3.25 and 3.16/3.16 at "
                "k=2); BDM_k x L2(k-1) -> flux k+1, u k "
                "(measured 2.04/2.09 and 1.04/1.06 at k=1). "
                "ndof check on unit_square maxh=0.3: order=1 "
                "RT=True 132 vs RT=False 84; order=2 270 vs 198; "
                "at order=0 the flag is a no-op (42 either way). "
                "Signal: refine twice and watch a scalar "
                "functional — a mis-paired mixed system is FLAT "
                "or drifting, not converging. (Verified "
                "empirically 2026-08-03 on NGSolve 6.2.2604 — "
                "catalog-drift correction; the shipped template "
                "used the broken RT_k x L2(k-1) pairing and has "
                "been repaired in the same commit. The prior "
                "claim 'if a user forgets RT=True the rate "
                "matches BDM exactly' was WRONG: with the "
                "documented L2(k-1) partner, forgetting RT=True "
                "is what makes it CORRECT.)"
            ),
            (
                "[Numerical] Normal component continuous across "
                "elements; div well-defined, so a normal-flux "
                "jump term is mathematically zero. In practice "
                "you do not get a zero number — you get an "
                "exception: Integrate of "
                "`(gfu*n - gfu.Other()*n)**2 * dx(skeleton=True)` "
                "on an HDiv GridFunction raises "
                "NgException('other mir not set, pls report to "
                "developers') on NGSolve 6.2.2604, because "
                ".Other() is only wired up inside a BilinearForm "
                "skeleton assembly, not inside a standalone "
                "Integrate. Signal: that literal 'other mir not "
                "set' text. Don't try to measure the jump; rely "
                "on the conformity. (Verified empirically "
                "2026-08-03 — signal-text correction; the prior "
                "text said the integrand 'vanishes identically'.)"
            ),
            (
                "[Numerical] Saddle-point system — use a direct "
                "solver or a block preconditioner. Signal: with "
                "a MIS-PAIRED flux/scalar combination the direct "
                "path is the one that talks: "
                "a.mat.Inverse(X.FreeDofs(), inverse='umfpack') "
                "prints 'UMFPACK V5.7.4 ...: WARNING: matrix is "
                "singular' and then raises "
                "NgException('UmfpackInverse: Numeric "
                "factorization failed.'). NOTE: `KSPSolve: "
                "DIVERGED_INDEFINITE_PC` is a PETSc message and "
                "is NOT emitted anywhere in the NGSolve stack — "
                "the prior catalog text imported it from the "
                "wrong solver ecosystem. (Verified empirically "
                "2026-08-03 — signal-text correction.)"
            ),
        ],
    },
}

GENERATORS = {
    "mixed_poisson_2d": _mixed_poisson_2d,
}
