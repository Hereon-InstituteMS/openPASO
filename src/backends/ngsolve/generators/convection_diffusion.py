"""NGSolve convection-diffusion generators and knowledge."""


def _convdiff_dg_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Convection-diffusion with DG upwind."""
    order = params.get("order", 4)
    eps = params.get("diffusion", 0.01)
    maxh = params.get("maxh", 0.05)
    return f'''\
"""Convection-diffusion: DG upwind — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = L2(mesh, order={order}, dgjumps=True)
u, v = fes.TnT()
n = specialcf.normal(2)
h = specialcf.mesh_size
order = {order}
eps = {eps}

b = CoefficientFunction((20, 5))  # convection velocity — set for your problem
# Upwind flux
uup = IfPos(b*n, u, u.Other())

a = BilinearForm(fes)
a += eps*grad(u)*grad(v)*dx
a += -b*u*grad(v)*dx
# Interior facets: upwind + penalty
a += b*n*uup*(v - v.Other())*dx(skeleton=True)
a += 4*order**2/h*eps*(u-u.Other())*(v-v.Other())*dx(skeleton=True)
a += (-eps*0.5*(grad(u)+grad(u.Other()))*n*(v-v.Other()))*dx(skeleton=True)
a += (-eps*0.5*(grad(v)+grad(v.Other()))*n*(u-u.Other()))*dx(skeleton=True)
# Boundary
a += b*n*u*v*ds(skeleton=True)
a += 4*order**2/h*eps*u*v*ds(skeleton=True)
a += (-eps*grad(u)*n*v - eps*grad(v)*n*u)*ds(skeleton=True)
a.Assemble()

f = LinearForm(1*v*dx).Assemble()
gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse() * f.vec

vtk = VTKOutput(mesh, coefs=[gfu], names=["solution"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "diffusion": eps}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("DG convection-diffusion solve complete.")
'''


KNOWLEDGE = {
    "convection_diffusion": {
        "description": "Convection-diffusion with DG upwind stabilization",
        "spaces": "L2(mesh, order=k, dgjumps=True) for DG",
        "solver": "Direct (DG systems are block-diagonal per element for explicit)",
        "pitfalls": [
            (
                "[API] MUST set dgjumps=True on the L2 space to "
                "allocate the cross-element coupling entries. "
                "Signal: assembling a dx(skeleton=True) jump "
                "integrand against L2(mesh, order=k) built "
                "WITHOUT dgjumps=True raises the literal "
                "NgException('SparseMatrixTM::AddElementMatrix: "
                "illegal dnums' + \"in Assemble BilinearForm "
                "'biform_from_py'\"). With dgjumps=True the same "
                "form assembles (nnz=756 on unit_square "
                "maxh=0.3, L2 order 1). NOTE: "
                "`SparseMatrixDynamic: row not allocated` is not "
                "emitted by NGSolve 6.2.2604, and "
                "`Sparse matrix: entry at (i,j) does not exist` "
                "is not emitted either — both prior "
                "catalog signals were wrong, and the failure is "
                "LOUD, not silent. (Verified empirically "
                "2026-08-03 — signal-text correction.)"
            ),
            (
                "[API] u.Other() accesses neighboring element "
                "value. Signal: forgetting .Other() in a "
                "skeleton form gives `u - u = 0` integrands "
                "everywhere — the jump term vanishes and the "
                "DG scheme reduces to a discontinuous "
                "polynomial fit per element with no inter-"
                "element coupling. (Audit 2026-06-02.)"
            ),
            (
                "[API] dx(skeleton=True) for interior facet "
                "integrals. Signal: omitting skeleton=True for a "
                "jump integrand raises the literal "
                "NgException('DG-facet terms need either "
                "skeleton=True or element_boundary=True') — but "
                "it is raised by BilinearForm.__iadd__, at the "
                "`a += ...` line, NOT at .Assemble(), so a "
                "try/except placed around Assemble() never sees "
                "it. Wrap the form construction instead. The "
                "message names a SECOND accepted spelling and it "
                "works: dx(element_boundary=True) assembles and "
                "produces the same cross-element coupling as "
                "dx(skeleton=True). The prior catalog signal "
                "`SymbolicBFI: u.Other() outside skeleton "
                "context` does not exist in NGSolve 6.2.2604, "
                "and the alternative branch in which the jump "
                "integrand is silently integrated against the "
                "bulk dx does not happen either "
                "— the form never reaches a usable state. "
                "(Verified empirically 2026-08-03, failure point "
                "corrected 2026-08-06 on NGSolve 6.2.2604.)"
            ),
            (
                "[Numerical] IfPos(b*n, u, u.Other()) for "
                "upwind flux selection. Signal: swapping the "
                "arguments — IfPos(b*n, u.Other(), u) — gives "
                "DOWNWIND advection that is unconditionally "
                "unstable; concentration field develops "
                "oscillations growing geometrically in the "
                "advection direction. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Penalty parameter: alpha*order^2/h "
                "for interior penalty DG. Signal: alpha too "
                "small gives coercivity loss — solution norm "
                "diverges with mesh refinement, or LU pivots "
                "approach zero; alpha too large gives "
                "cond(K)>1e14 and CG stalls. Rule of thumb: "
                "alpha = 4 * order^2 for symmetric interior "
                "penalty. (Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "convection_diffusion_2d_dg": _convdiff_dg_2d,
}
