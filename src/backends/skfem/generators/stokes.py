"""scikit-fem Stokes flow generators and knowledge."""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow with Taylor-Hood P2/P1.

    Fixed 2026-06-01 (was unrunnable):
      * Mesh: MeshTri.refined(N) (vs prior MeshQuad —
        ElementQuad2 has assembly issues with the
        divergence form in skfem 12).
      * intorder=4 on BOTH bases (trial/test mismatch
        on the divergence bilinear form was the immediate
        ValueError on import).
      * B-block sign: skfem.models.general.divergence
        returns +∫q·div(u); the catalog negates to form
        the standard saddle-point [[K, -B^T], [-B, 0]].
      * Pressure pin: a single pressure DOF is fixed at
        0 to remove the constant-pressure null space
        (without this, spsolve raises a singular-matrix
        error even with the correct BC).
      * Driven-cavity BC: u = (1, 0) on top edge, no-slip
        elsewhere — replaces the prior all-zero BC which
        gave the trivial u=0 solution.
    """
    nx_refine = int(params.get("refine", 4))
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — scikit-fem"""
from skfem import (
    MeshTri, Basis, BilinearForm,
    ElementVector, ElementTriP1, ElementTriP2,
    asm, condense, solve,
)
from skfem.helpers import grad, ddot
from skfem.models.general import divergence
from scipy.sparse import bmat
import numpy as np
import json

# Mesh + Taylor-Hood spaces.
# Trial / test bases must share quadrature for the mixed
# divergence(basis_u, basis_p) bilinear form. Default
# intorder differs between P2 and P1 → ValueError
# 'Quadrature mismatch'. Force intorder=4 on both.
m = MeshTri().refined({nx_refine})
basis_u = Basis(m, ElementVector(ElementTriP2()), intorder=4)
basis_p = Basis(m, ElementTriP1(), intorder=4)


@BilinearForm
def stiffness(u, v, w):
    return ddot(grad(u), grad(v))


K_block = asm(stiffness, basis_u)
# B[q, u] = ∫q·div(u) dx (skfem convention is +div, so
# we negate to form the standard saddle-point block
# [[K, -B^T], [-B, 0]]).
B_block = -asm(divergence, basis_u, basis_p)

A = bmat([[K_block, B_block.T], [B_block, None]], format='csr')
F = np.zeros(A.shape[0])

# Driven-cavity BC: u = (1, 0) on top edge, no-slip elsewhere.
# ElementVector interleaves x/y dofs at each node:
# dof[2i]=x-component, dof[2i+1]=y-component.
doflocs_u = basis_u.doflocs
by = doflocs_u[1]
top_x = np.isclose(by[0::2], 1.0)
u_bc = np.zeros(basis_u.N)
u_bc[0::2] = np.where(top_x, 1.0, 0.0)
u_bc[1::2] = 0.0

# Pin a single pressure DOF (closest to the origin)
# to fix the constant-pressure null space.
pdofs = basis_p.doflocs.T
pin_p_local = int(np.argmin(
    np.linalg.norm(pdofs[:, :2], axis=1)))
pin_p_global = basis_u.N + pin_p_local

D_u = basis_u.get_dofs().flatten()
D = np.concatenate([
    D_u,
    np.array([pin_p_global], dtype=np.int64),
])
x_full = np.zeros(A.shape[0])
x_full[:basis_u.N] = u_bc

sol = solve(*condense(A, F, D=D, x=x_full))
u_h = sol[:basis_u.N]
p_h = sol[basis_u.N:]
print(f"Stokes Taylor-Hood: total DOFs = {{A.shape[0]}}")
print(f"  ||u_x||_inf = {{np.abs(u_h[0::2]).max():.4f}}")
print(f"  ||u_y||_inf = {{np.abs(u_h[1::2]).max():.4f}}")
print(f"  ||p||_inf   = {{np.abs(p_h).max():.4f}}")

summary = {{
    "n_dofs": int(A.shape[0]),
    "max_u_x": float(np.abs(u_h[0::2]).max()),
    "max_u_y": float(np.abs(u_h[1::2]).max()),
    "max_p":   float(np.abs(p_h).max()),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Stokes solve complete.")
'''


KNOWLEDGE = {
    "stokes": {
        "description": "Stokes flow — Taylor-Hood P2/P1 or Mini element (examples 18, 24, 30, 32)",
        "solver": "Block system: [[A, B^T], [B, 0]], solved via direct or Krylov-Uzawa",
        "elements": "Taylor-Hood: ElementVector(ElementTriP2()) + ElementTriP1(). Mini: ElementTriMini()",
        "pitfalls": [
            "[Numerical] The Stokes block system [[A, B^T], "
            "[B, 0]] is INDEFINITE: with one pressure DOF pinned "
            "the condensed block has as many negative eigenvalues "
            "as there are free pressure unknowns. CG has no "
            "convergence guarantee on such a matrix. Use a direct "
            "solver (scipy.sparse.linalg.spsolve with UMFPACK / "
            "SuperLU), MinRes, or block Uzawa iteration. "
            "Signal: DO NOT guard this with cg's info flag. On "
            "the assembled Taylor-Hood block system scipy's cg "
            "returns info = 0 — converged — raising nothing and "
            "warning nothing, at a small relative residual. There "
            "is no info != 0 to catch and no O(1) stall to "
            "observe, so an agent told to check the flag writes a "
            "guard that never fires and reads its own silence as "
            "proof the solver was appropriate. The theoretical "
            "caution stands and the indefiniteness is real; the "
            "honest check is to compute the relative residual "
            "||A x - b|| / ||b|| yourself, require np.isfinite on "
            "the iterate, and compare against a direct solve, "
            "which lands orders of magnitude below what CG "
            "reports as converged. The nullity of the condensed "
            "block and the count of its negative eigenvalues are "
            "both computable without solving at all. (Verified "
            "empirically 2026-08-06 on skfem 12.0.1 / scipy "
            "1.15.3 — indefiniteness confirmed, the stated cg "
            "signal falsified.)",
            "[Numerical] Krylov-Uzawa (skfem example 30) splits "
            "the saddle-point: outer iteration on the pressure "
            "Schur complement, inner solve of the velocity block. "
            "The split works, but two numbers previously "
            "attached to it do not. The per-outer pressure "
            "residual does NOT drop by 0.1..0.3 — measured on a "
            "Taylor-Hood cavity the unpreconditioned Schur CG "
            "contracts far more slowly than that, and the factor "
            "WORSENS under refinement, as an unpreconditioned "
            "Schur solve should; a gate keyed on that band calls "
            "a healthy run broken. And monolithic Krylov on the "
            "full block does NOT stall: CG on the condensed "
            "system reaches the requested tolerance with "
            "info = 0 at every level tried, which the "
            "indefiniteness entry above found independently. "
            "Signal: compare the two by COST, not by "
            "converges-versus-not, and do not compare their "
            "iteration counts directly — an Uzawa outer step "
            "carries a whole velocity solve, so outer iterations "
            "and Krylov iterations are not the same unit. If you "
            "want the outer factor to behave, precondition the "
            "Schur complement (a pressure mass matrix is the "
            "standard choice); the bare version is not "
            "mesh-independent. (Verified empirically 2026-08-06 "
            "on skfem 12.0.1 / scipy 1.15.3 — both quoted "
            "signals corrected.)",
            "[Physics] Enclosed-flow Stokes admits the constant "
            "pressure null space — pin pressure at one DoF "
            "(boundary_values[p_dof] = 0) or add a Lagrange "
            "multiplier mean(p) = 0. Open flows with "
            "do-nothing (traction-free) outlet BCs determine "
            "pressure uniquely without pinning. Signal: "
            "spsolve(A_block, b_block) returns SUCCESSFULLY — "
            "and that is the whole difficulty. The unpinned "
            "system is singular but CONSISTENT, so SuperLU "
            "returns a particular solution: nothing is raised, "
            "the warning record is EMPTY (no MatrixRankWarning), "
            "the vector is finite, and the velocity matches the "
            "pinned solution to round-off. Only the pressure "
            "LEVEL is arbitrary, which is exactly why neither a "
            "catch-warnings guard nor np.isfinite can see it — "
            "see the same correction on hydraulic_resistance and "
            "navier_stokes. The observable that IS produced: "
            "take the nullity of the condensed block (exactly "
            "one, its null vector zero on the velocity block and "
            "constant on the pressure block), or solve twice "
            "pinning DIFFERENT DOFs at DIFFERENT values and "
            "check that the velocities agree to round-off while "
            "the two pressures differ by an exact constant. Same "
            "family as fenics poisson#3 (pure Neumann). "
            "(Verified empirically 2026-08-06 on skfem 12.0.1 / "
            "scipy 1.15.3 — previously carried as an inherited, "
            "unverified claim.)",
            "[Syntax] 3D Stokes (skfem example 32): use "
            "ElementTetP2 for velocity + ElementTetP1 for "
            "pressure (Taylor-Hood). Other tet combinations "
            "violate inf-sup. THERE IS NO GridFunction IN "
            "scikit-fem — hasattr(skfem, 'GridFunction') is "
            "False; that is NGSolve vocabulary. A skfem solution "
            "is a plain 1-D numpy.ndarray over the stacked "
            "[u; p] block, and you slice the pressure out of it "
            "at offset basis_u.N. "
            "Signal: do not look for a checkerboard by eye — "
            "count the spurious pressure modes instead, which "
            "needs no solve at all. Assemble the block, condense "
            "the velocity boundary and one pinned pressure DOF, "
            "and take the nullity of the condensed matrix: for "
            "the Taylor-Hood pair that count stays CONSTANT as "
            "the mesh is refined, while for an equal-order "
            "P1/P1 pair it GROWS with the mesh, which is the "
            "inf-sup failure made countable. A second, cheaper "
            "check is the pressure range from a lightly "
            "regularised solve: bounded and converging for "
            "Taylor-Hood, growing by an order of magnitude per "
            "refinement for equal order. (Verified empirically "
            "2026-08-06 on skfem 12.0.1 — the GridFunction "
            "wording was foreign vocabulary and is removed; the "
            "previous 'integral of div(u) is O(1) instead of "
            "O(eps)' wording did NOT reproduce cleanly under a "
            "regularised solve and has been replaced by the "
            "mode-count signature.)",
            "[API] CRITICAL: skfem.Basis.Nbfun returns the "
            "PER-ELEMENT DOF count (3 for ElementTriP1, 6 for "
            "ElementTriP2), NOT the global DOF count. Use "
            "basis.N or the assembled matrix shape (A.shape[0]) "
            "for the global count. Signal: basis.Nbfun on a "
            "refined MeshTri is 3 or 6 regardless of how many "
            "elements; basis.N grows with refinement. (Verified "
            "empirically 2026-06-01: P1 Nbfun=3, P2 Nbfun=6 on "
            "MeshTri.refined(2), while N=25 and N=81 "
            "respectively. Using Nbfun for DOF splitting "
            "silently slices wrong.)",
            "[API] ElementVector DOF ordering IS reliable and IS "
            "interleaved — the previous 'unreliable, depends on "
            "element type' wording is wrong and was never "
            "checked. Component 1 occupies the even global "
            "indices and component 2 the odd ones, with "
            "dof(u^2) = dof(u^1) + 1 exactly, and that layout is "
            "IDENTICAL for ElementTriP1, ElementTriP2 and "
            "ElementTriP3 and for the tetrahedral family: "
            "basis.doflocs[:, 0::dim] reproduces the scalar "
            "basis doflocs for every component. You may still "
            "prefer two scalar bases for a block layout you "
            "control, but it buys layout preference, not "
            "correctness. Signal: assert the interleaving rather "
            "than avoiding it — check basis.N == dim * "
            "scalar_basis.N and that get_dofs(...).nodal['u^1'] "
            "and ['u^2'] differ by exactly one at every entry. "
            "Do NOT slice a vector solution as [:N] and [N:] "
            "expecting components; that is the BLOCKED layout, "
            "which skfem does not use, and it silently mixes the "
            "two components. (Verified empirically 2026-08-06 on "
            "skfem 12.0.1 — claim falsified.)",
            "[API] skfem.asm() with two bases of mismatched "
            "intorder raises ValueError 'Quadrature mismatch: "
            "trial and test functions should have same number "
            "of integration points.'. Mixed Stokes (P2 velocity "
            "+ P1 pressure) needs intorder=4 (or higher) on "
            "BOTH bases. Signal: scipy ValueError text matches "
            "the exact wording above when asm(b_form, p2_basis, "
            "p1_basis) has different intorder. (Verified "
            "empirically 2026-06-01 with intorder=2 vs 6.)",
            "[Physics] The Stokes pressure SIGN is a property "
            "of the form YOU write, not of the library. "
            "scikit-fem ships no Stokes form at all, so the "
            "-p*div(v) convention is the CATALOG TEMPLATE's "
            "choice; writing +div(u)*q instead is a "
            "one-character change that skfem accepts without "
            "complaint. NGSolve's tutorial form uses the "
            "opposite sign. Cross-verified by running both "
            "libraries in one process: flipping the sign of the "
            "divergence coupling negates the pressure field "
            "EXACTLY and leaves the velocity field "
            "BIT-IDENTICAL, in skfem and in NGSolve alike. "
            "That is precisely why porting a weak form between "
            "the two unchanged is dangerous — the quantity you "
            "would sanity-check first is completely unaffected, "
            "and only the pressure sign moves, which a reader "
            "inspecting a pressure plot for SHAPE will not "
            "notice. Signal: do not try to compare the two "
            "codes DoF by DoF — they mesh and number "
            "differently, so there is no correspondence to "
            "compare. Compare the sign of the pressure at a "
            "named boundary against the sign of the driving "
            "pressure, in each code separately; and within one "
            "code, check that flipping the coupling leaves the "
            "velocity unchanged, which confirms you have "
            "isolated a sign convention rather than a bug. "
            "(Cross-verified empirically 2026-08-06 on skfem "
            "12.0.1 and NGSolve 6.2.2604.)",
        ],
    },
}

GENERATORS = {
    "stokes_2d": _stokes_2d,
}
