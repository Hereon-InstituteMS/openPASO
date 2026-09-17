"""NGSolve linear elasticity generators and knowledge."""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on rectangular domain, fixed left edge, body force."""
    nx = params.get("nx", 40)
    ny = params.get("ny", 4)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    maxh = lx / nx
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Linear elasticity: rectangular domain, fixed left, body force — NGSolve"""
from ngsolve import *
import json
from netgen.geom2d import SplineGeometry

# Domain mesh — set geometry for your problem
geo = SplineGeometry()
pnts = [(0, 0), ({lx}, 0), ({lx}, {ly}), (0, {ly})]
p = [geo.AddPoint(*pnt) for pnt in pnts]
geo.Append(["line", p[0], p[1]], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p[1], p[2]], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p[2], p[3]], leftdomain=1, rightdomain=0, bc="top")
geo.Append(["line", p[3], p[0]], leftdomain=1, rightdomain=0, bc="left")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# Vector FE space
fes = VectorH1(mesh, order=1, dirichlet="left")
u, v = fes.TnT()

mu_val = {mu}
lam_val = {lam}

def Strain(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

def Stress(u):
    return 2 * mu_val * Strain(u) + lam_val * Trace(Strain(u)) * Id(2)

a = BilinearForm(InnerProduct(Stress(u), Strain(v)) * dx).Assemble()
f = LinearForm(CoefficientFunction((0, -1)) * v * dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Max tip displacement
disp = gfu.components
max_uy = 0
for p in mesh.vertices:
    val = disp[1](mesh(*p.point))
    if abs(val) > abs(max_uy):
        max_uy = val

print(f"Max tip displacement (y): {{max_uy:.6f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"],
                filename="result", subdivision=0)
vtk.Do()

summary = {{
    "max_displacement_y": float(max_uy),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Elasticity solve complete.")
'''


def _elasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D linear elasticity with OCC geometry."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 10)
    ly = params.get("ly", 1)
    lz = params.get("lz", 1)
    maxh = params.get("maxh", 0.5)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""3D Linear elasticity — NGSolve + OCC geometry"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry — set dimensions for your problem
box = Box((0,0,0), ({lx},{ly},{lz}))
geo = OCCGeometry(box)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

fes = VectorH1(mesh, order=2, dirichlet=".*")
# Fix left face only
fes = VectorH1(mesh, order=2, dirichlet="face3")  # face numbering depends on OCC
u, v = fes.TnT()

mu_val, lam_val = {mu}, {lam}
def Strain(u): return 0.5*(Grad(u) + Grad(u).trans)
def Stress(u): return 2*mu_val*Strain(u) + lam_val*Trace(Strain(u))*Id(3)

a = BilinearForm(InnerProduct(Stress(u), Strain(v))*dx).Assemble()
f = LinearForm(CoefficientFunction((0, -1, 0))*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
print(f"DOFs: {{fes.ndof}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "n_elements": mesh.ne}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": (
            "Linear elasticity (plane strain/stress, 3D). Both "
            "VectorH1(mesh, order=k) and H1(mesh, order=k, dim=d) "
            "are valid vector spaces and give identical solve "
            "results; they differ only in memory layout (see "
            "pitfall #0)."
        ),
        "spaces": (
            "VectorH1 (flat layout: ndof = d*scalar_ndof, dim=1) "
            "OR H1 with dim parameter (block layout: "
            "ndof = scalar_ndof, dim=d). Same ProxyFunction shape "
            "from TnT(); operationally equivalent."
        ),
        "solver": (
            "Direct for small, CG+AMG for large. Preconditioners: "
            "bddc, multigrid."
        ),
        "pitfalls": [
            "[API] VectorH1(mesh, order=k) and H1(mesh, order=k, "
            "dim=d) BOTH produce a valid d-dimensional vector "
            "FESpace for elasticity — neither is wrong. They "
            "differ only in layout: VectorH1 has dim=1 and "
            "ndof = d * scalar_ndof (flat); H1(dim=d) has dim=d "
            "and ndof = scalar_ndof (block). TnT() returns "
            "ProxyFunction with .dim == d in both cases, and "
            "Grad(u).dims == (d, d). An assembled BilinearForm "
            "of InnerProduct(Stress(u), Strain(v))*dx on either "
            "space gives the same solve to ~1e-16 relative "
            "norm. Signal: type(VectorH1(...)).__name__ == "
            "'VectorH1', type(H1(..., dim=d)).__name__ == 'H1' "
            "(NOT 'CompoundFESpace'); norm of gfu.vec matches "
            "between the two formulations. (Verified empirically "
            "2026-06-01 — Tier-2 fixture vector_h1_vs_h1_dim2_"
            "equivalence in scripts/tier2_fixtures/ngsolve/. "
            "Catalog-drift correction: the previous claim "
            "'NOT H1(dim=2)' was false.)",
            "[Syntax] Body forces are constructed with "
            "CoefficientFunction taking a Python tuple shaped to "
            "match the vector FESpace: "
            "CoefficientFunction((fx, fy)) for 2D, "
            "CoefficientFunction((fx, fy, fz)) for 3D. The rule "
            "really is 'match the space' — a 3-tuple is the "
            "mismatch in 2D and the correct one in 3D, with the "
            "2-tuple swapping roles. "
            "Signal: THREE corrections to where and how the "
            "mismatch shows up. (a) It is raised by "
            "LinearForm.__iadd__, at the `f += ...` line, never "
            "reaching .Assemble() — so a guard wrapped around "
            "Assemble(), which the old wording invited, sees "
            "neither variant. (b) A SCALAR source on a vector "
            "space, the very example the old text gave, produces "
            "NO dimension message at all: the product is "
            "vector-valued and NGSolve says 'SymbolicLFI needs "
            "scalar-valued CoefficientFunction', so grepping for "
            "a dimension mismatch on that case finds nothing. "
            "(c) A wrong-length TUPLE does give a dimension "
            "message, but the literal text is contracted — "
            "\"Dimensions don't match, op = *\" followed by the "
            "two dims — not 'dimensions do not match', so a "
            "substring guard written from the old wording "
            "misses it. Guard on the exception type at form "
            "construction and assert the source's .dim against "
            "the space's dimension before assembling. (Verified "
            "2026-08-06 on NGSolve 6.2.2604 — previously carried "
            "as an inherited claim.)",
            "[API] gfu.components on a VectorH1 GridFunction "
            "returns a tuple of ComponentGridFunction views (NOT "
            "a list, NOT direct vector slices). Each component "
            "i is callable as gfu.components[i](mesh(x,y)) for "
            "pointwise evaluation of the i-th displacement "
            "component. Signal: type(gfu.components).__name__ "
            "== 'tuple' and type(gfu.components[0]).__name__ == "
            "'ComponentGridFunction'. (Verified empirically "
            "2026-06-01 — same Tier-2 fixture as #0.)",
            "[API] Stress visualization uses MatrixValued(H1(mesh, "
            "order=k), symmetric=True). The 'symmetric=True' "
            "kwarg packs only the upper triangle into the dof "
            "vector (ndof reduced from d^2*scalar_ndof to "
            "d*(d+1)/2*scalar_ndof). Without symmetric=True the "
            "MatrixValued space has the full d^2 storage. "
            "Signal: type(MatrixValued(...)).__name__ == "
            "'MatrixValued'; ndof differs by a factor of "
            "d^2/(d*(d+1)/2) between symmetric=False and "
            "symmetric=True. (Verified empirically 2026-06-01 — "
            "for 2D: ndof_full=32, ndof_sym=24 with scalar_ndof=8.)",
            "[Physics] For plane strain use the standard 3D Lame "
            "parameters lambda = E*nu / ((1+nu)*(1-2*nu)) and "
            "mu = E / (2*(1+nu)). For plane stress, lambda is "
            "modified to lambda* = 2*lambda*mu / (lambda + 2*mu) "
            "(or equivalently use the 2D-compatible derivation "
            "from E and nu). Mixing the two silently introduces "
            "a Poisson-ratio-dependent error in the deflection "
            "scaling. Signal: gfu.components[1] tip evaluation on "
            "a VectorH1 cantilever solve produced by the Stress "
            "operator with plane-strain lambda differs from the "
            "plane-stress lambda formulation by a factor on the "
            "order of 1/(1-nu^2). (Advisory pitfall — not "
            "empirically falsified this iteration.)",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
    "linear_elasticity_3d": _elasticity_3d,
}
