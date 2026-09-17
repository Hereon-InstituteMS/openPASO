"""scikit-fem nonlinear PDE generators and knowledge."""


def _nonlinear_2d_skfem(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Nonlinear PDE -div((1+u^2)*grad(u)) = f with manual Newton iteration."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    tol = params.get("newton_tol", 1e-10)
    max_iter = params.get("max_iter", 50)
    return f'''\
"""Nonlinear PDE: -div((1+u^2)*grad(u)) = f — Newton iteration — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import spsolve
import json

# Mesh: structured quad mesh
m = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e = ElementQuad1()
ib = Basis(m, e)

# Boundary DOFs
D = ib.get_dofs().flatten()
I = ib.complement_dofs(D)

# Nonlinear coefficient forms
@BilinearForm
def nonlinear_stiffness(u, v, w):
    # (1 + w_prev^2) * grad(u) . grad(v)
    u_prev = w["u_prev"]
    return (1 + u_prev ** 2) * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

@BilinearForm
def jacobian_extra(u, v, w):
    # 2 * w_prev * grad(w_prev) . grad(v) * u  (linearization of the nonlinear coefficient)
    u_prev = w["u_prev"]
    return 2 * u_prev * (w["u_prev_grad"][0] * v.grad[0] + w["u_prev_grad"][1] * v.grad[1]) * u

@LinearForm
def residual_form(v, w):
    # (1 + w_prev^2) * grad(w_prev) . grad(v) - f * v
    u_prev = w["u_prev"]
    return (1 + u_prev ** 2) * (w["u_prev_grad"][0] * v.grad[0] + w["u_prev_grad"][1] * v.grad[1]) - {f_val} * v

# Initial guess: zero
u = ib.zeros()

# Newton iteration
for it in range({max_iter}):
    # Interpolate current solution to quadrature points
    u_prev = ib.interpolate(u)
    u_prev_grad = u_prev.grad

    # Assemble residual
    R = residual_form.assemble(ib, u_prev=u_prev.value, u_prev_grad=u_prev_grad)

    # Assemble Jacobian: d/du [(1+u^2)*grad(u).grad(v)]
    J1 = nonlinear_stiffness.assemble(ib, u_prev=u_prev.value)
    J2 = jacobian_extra.assemble(ib, u_prev=u_prev.value, u_prev_grad=u_prev_grad)
    J = J1 + J2

    # Apply boundary conditions
    R[D] = 0.0
    du = np.zeros_like(u)
    du[I] = spsolve(J[I][:, I], -R[I])

    u += du
    res_norm = np.linalg.norm(du[I])
    print(f"Newton it {{it+1}}: ||du|| = {{res_norm:.6e}}")
    if res_norm < {tol}:
        print(f"Converged in {{it+1}} iterations")
        break

max_val = u.max()
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{len(u)}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "n_dofs": len(u),
    "n_elements": m.nelements,
    "newton_iterations": it + 1,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Nonlinear PDE solve complete.")
'''


KNOWLEDGE = {
    "nonlinear": {
        "description": "Nonlinear PDE via manual Newton iteration (scikit-fem)",
        "solver": "Manual Newton loop: assemble Jacobian + residual, solve with spsolve",
        "elements": "ElementQuad1, ElementTriP1 (any standard H1 element)",
        "pitfalls": [
            "[API] scikit-fem provides NO built-in Newton "
            "solver — the user must write a manual residual + "
            "Jacobian loop using assemble + scipy.sparse.linalg."
            "spsolve. Signal: neither skfem.* nor skfem.helpers."
            "* has any attribute matching 'newton' or 'nonlin' "
            "(verified via dir() inspection). Don't reach for a "
            "skfem.NewtonSolver — it doesn't exist. (Verified "
            "empirically 2026-06-01.)",
            "[Numerical] The Newton-loop Jacobian is the "
            "linearisation of the nonlinear weak form with "
            "respect to the previous solution iterate. Assemble "
            "as a fresh skfem.BilinearForm each iteration using "
            "basis.interpolate(u_prev) inside the @BilinearForm "
            "body. Signal: the ratio of consecutive "
            "scipy.sparse.linalg.norm(residual) values follows "
            "the O(r^2) Newton convergence rate only when the "
            "BilinearForm-assembled Jacobian matches the exact "
            "linearisation; using a Quasi-Newton secant "
            "Jacobian collapses to linear (constant-ratio) "
            "convergence. (Verified empirically 2026-08-30 on "
            "-div((1+u^2) grad u) = 1, P1, refine(4): the exact "
            "Jacobian gives residuals 5.9e-2, 1.2e-4, 1.4e-9, "
            "1.2e-16 — each roughly the square of the last — "
            "while a frozen Jacobian holds a constant ratio of "
            "0.006 for five iterations.)",
            "[API] basis.interpolate(u) returns a "
            "skfem.element.DiscreteField with .value (evaluated "
            "at quadrature points) and .grad attributes. Use "
            "the field inside @BilinearForm/@LinearForm bodies "
            "to access the previous-iterate solution and its "
            "gradient. Signal: type(basis.interpolate(u_dof_vec))."
            "__name__ == 'DiscreteField'; hasattr(it, 'grad') "
            "is True; for u = constant, the .grad array is "
            "uniformly zero. (Verified empirically 2026-06-01.)",
            "[API] basis.interpolate(u).grad has shape "
            "(spatial_dim, n_elements, n_quad_points) in 2D. "
            "Indexing patterns like .grad[0] (x-component over "
            "all elements and quad points) work directly inside "
            "@Form decorated functions. Signal: "
            "basis.interpolate(np.ones(N) * c).grad shape is "
            "(2, n_elements, n_quad_points_per_element) for a "
            "2D mesh with P1 elements (n_quad=3 per triangle "
            "by default); all values are 0 because the function "
            "is constant. (Verified empirically 2026-06-01 with "
            "MeshTri.refined(2).)",
            "[Numerical] Quadratic convergence diagnostic: "
            "scipy.sparse.linalg.norm(residual_vector) computed "
            "via skfem.LinearForm.assemble at each iteration "
            "should drop with successive ratios r_{k+1}/r_k that "
            "decrease by ~10x per step near the solution (slope "
            "of log-residual vs iteration index doubles). Signal: "
            "if the ratio stays roughly constant across iterations "
            "the assembled skfem.BilinearForm Jacobian is wrong "
            "or inexact. (Verified empirically 2026-08-30 — the "
            "same measurement as the Newton-Jacobian entry above: "
            "a constant 0.006 ratio is exactly what a deliberately "
            "frozen Jacobian produces.)",
            "[Numerical] For difficult problems (large initial "
            "residual, poor initial guess) add line search or "
            "damping: u += alpha * du with alpha < 1, chosen "
            "to ensure the residual norm decreases at each "
            "step. A simple backtracking line search halves "
            "alpha until scipy.sparse.linalg.norm of the "
            "skfem.LinearForm-assembled residual at u + "
            "alpha*du is below the previous value. Signal: "
            "with alpha=1, the residual norm grows across "
            "iterations; with alpha=0.5 backtracking, the "
            "residual decreases monotonically. (Claim "
            "inherited.)",
        ],
    },
}

GENERATORS = {
    "nonlinear_2d": _nonlinear_2d_skfem,
}
