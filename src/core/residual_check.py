"""Discrete-residual check: does the delivered field actually solve the problem?

WHY THIS EXISTS
---------------
Attestation binds a reported number to a FILE. It never binds it to a PROBLEM.
An adversarial review demonstrated the consequence: the cheapest forgery is not
an elaborate one but a trivial one — four nodes and a constant hit any target
exactly, with no knowledge of the physics at all; and an eight-line script that
emits an analytic field plus an h-dependent perturbation is *more accurate*
than a genuine solve, 82x faster, and passes a mesh-independence study.

Nothing that inspects only the data can tell those apart, because as data they
are perfectly well-formed. The distinguishing property is not in the numbers'
appearance but in whether they SATISFY THE EQUATIONS they claim to solve.

WHAT THIS CHECKS
----------------
Given the delivered mesh and field, openPASO assembles the discrete operator of
the stated problem ON THAT MESH and measures

    rho = || A u_delivered - b ||  /  || b ||        (interior degrees of freedom)

For a genuine finite element solution rho is at the linear solver's tolerance,
because A u = b is precisely the system that was solved. For anything else it
is not:

  * the exact solution sampled at nodes does NOT satisfy the discrete system —
    it misses by the truncation error, which is orders of magnitude larger;
  * a perturbed analytic field misses by more;
  * a constant, or a field on a mesh that is not the stated domain, misses
    grossly or is refused outright by the domain check.

This is the check the attestation module's docstring names and defers. It
converts "the data looks plausible" into "the data satisfies the equations",
which is the only statement that separates solving from fabricating.

SCOPE
-----
Implemented for the operator families the evaluation uses: scalar diffusion
    -div(K grad u) = f
with constant or spatially varying scalar/tensor coefficient, on simplicial
meshes in 2D and 3D, with Dirichlet data on the outer boundary. It is not a
universal PDE checker: for an unsupported operator it returns UNSUPPORTED
rather than a false pass, so a caller can never mistake "not checked" for
"checked and fine".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualVerdict:
    supported: bool
    relative_residual: float | None
    # True = consistent with a real solve, False = positively not,
    # None = neither could be established (see status).
    solver_like: bool | None
    n_interior: int
    domain_measure: float | None
    detail: str
    status: str = ""                  # solves | does_not_solve | inconclusive


# WHY THERE ARE TWO THRESHOLDS AND A BAND BETWEEN THEM
#
# A single cut-off cannot work, and measuring showed exactly how it fails. The
# residual of an analytic field sampled at the nodes falls as h^4 relative to
# the system's own scale — 6.3e-4, 4.1e-5, 2.7e-6, 1.7e-7, 1.1e-8, 6.7e-10 over
# six uniform refinements — while a genuine direct solve stays at round-off,
# about 1e-16, whatever the mesh. So a fixed 1e-8 cut-off silently starts
# CERTIFYING the forgery once the mesh is fine enough, and at the other end it
# accuses an honest run that used an iterative solver with a 1e-8 tolerance.
# Both errors are bad; the second is worse, because it condemns correct work.
#
# What actually separates the two is not a number but a scaling: a real solve
# sits at the round-off floor of this system, which depends on its size and not
# on h. So the pass threshold is that floor, and a verdict is only returned when
# the evidence supports it:
#
#   at or below the round-off floor  -> SOLVES        (only a solve gets here)
#   at or above FORGERY_RESIDUAL_MIN -> DOES NOT SOLVE(no solver leaves this)
#   in between                       -> INCONCLUSIVE  (neither credited nor
#                                                      accused)
#
# The band is where a loosely-converged honest solve and a fine-mesh forgery
# genuinely overlap. Neither is given the benefit of the doubt: the run is not
# certified as solving its equations, and it is not accused of not solving them.
# Refusing to answer when the evidence is ambiguous is the whole difference
# between a check and a guess.

# No solver leaves a residual this large. Above it, the field was not obtained
# by solving this system.
FORGERY_RESIDUAL_MIN = 1e-6
# Never demand better than this, however small the system.
ROUNDOFF_FLOOR_MIN = 1e-14
# Round-off grows with system size; this multiplies sqrt(n) * machine epsilon.
ROUNDOFF_SAFETY = 10.0
# Domain measure must match the stated domain to this relative tolerance.
DOMAIN_TOL = 1e-6


def _require_skfem():
    try:
        import skfem  # noqa: F401
        return True
    except Exception:
        return False


def check_elasticity_residual(points, cells, values, *, dim: int,
                              source_fn, young: float, poisson: float,
                              domain_measure_expected: float | None = None,
                              ) -> ResidualVerdict:
    """Does this displacement field solve linear elastostatics on this mesh?

    Scalar diffusion is the easy case and it was the only one covered, which
    meant the anti-fabrication check silently did nothing on exactly the
    problems where fabrication matters — elasticity is the most common thing
    anyone actually runs. UNSUPPORTED is honest, but it is not protection.

    Same discriminator as the scalar case: assemble -div(sigma(u)) = f on the
    DELIVERED mesh and measure how far the delivered displacement is from
    satisfying it. A genuine solve sits at the round-off floor; an analytic
    field sampled at the nodes misses by the truncation error.

    values is the nodal displacement, shape (n_points, dim). source_fn(x)
    returns the body force, shape (dim, n).
    """
    if not _require_skfem():
        return ResidualVerdict(False, None, None, 0, None,
                               "scikit-fem unavailable; cannot assemble",
                               status="unsupported")
    from skfem import (Basis, BilinearForm, ElementTetP1, ElementTriP1,
                       ElementVector, LinearForm, asm)
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    pts = np.asarray(points, float)
    vals = np.asarray(values, float)
    if vals.ndim != 2 or vals.shape[1] < dim:
        return ResidualVerdict(False, None, None, 0, None,
                               f"expected a {dim}-component displacement field, "
                               f"got shape {vals.shape}", status="unsupported")
    vals = vals[:, :dim]
    if pts.shape[0] != vals.shape[0]:
        return ResidualVerdict(False, None, None, 0, None,
                               "field length does not match point count",
                               status="unsupported")

    mesh = _build_mesh(pts, cells, dim)
    if mesh is None:
        return ResidualVerdict(False, None, None, 0, None,
                               f"no cells of dimension {dim} that openPASO can "
                               f"assemble on", status="unsupported")

    scalar_elem = ElementTriP1() if dim == 2 else ElementTetP1()
    basis = Basis(mesh, ElementVector(scalar_elem))

    one = LinearForm(lambda v, w: v[0])
    measure = float(asm(one, basis).sum())
    if domain_measure_expected is not None:
        rel = abs(measure - domain_measure_expected) / max(
            abs(domain_measure_expected), 1e-30)
        if rel > DOMAIN_TOL:
            return ResidualVerdict(
                True, None, False, 0, measure,
                f"mesh covers measure {measure:.6g}, but the stated domain has "
                f"{domain_measure_expected:.6g}: this is not the problem's domain",
                status="does_not_solve")

    lam, mu = lame_parameters(young, poisson)
    A = asm(linear_elasticity(lam, mu), basis)

    @LinearForm
    def l_form(v, w):
        f = source_fn(w.x)
        return sum(f[i] * v[i] for i in range(dim))

    b = asm(l_form, basis)

    # Scatter the nodal field onto the vector basis using the basis's own dof
    # map rather than assuming an interleaving. Guessing the component order is
    # how a check like this silently measures the wrong thing.
    u = np.zeros(A.shape[0])
    nd = basis.nodal_dofs
    if nd.shape != (dim, pts.shape[0]):
        return ResidualVerdict(False, None, None, 0, measure,
                               f"basis has {nd.shape} nodal dofs, field has "
                               f"{vals.shape}", status="unsupported")
    for c in range(dim):
        u[nd[c]] = vals[:, c]

    boundary = basis.get_dofs().flatten()
    interior = np.setdiff1d(np.arange(A.shape[0]), boundary)
    if interior.size == 0:
        return ResidualVerdict(True, None, False, 0, measure,
                               "no interior degrees of freedom to test",
                               status="does_not_solve")

    r = A @ u - b
    num = float(np.linalg.norm(r[interior]))
    scale = float(np.linalg.norm((abs(A) @ np.abs(u))[interior]))
    den = max(float(np.linalg.norm(b[interior])), scale)
    rho = num / den if den > 0 else (0.0 if num == 0 else float("inf"))
    floor = max(ROUNDOFF_FLOOR_MIN,
                ROUNDOFF_SAFETY * math.sqrt(interior.size) * np.finfo(float).eps)

    if rho <= floor:
        return ResidualVerdict(
            True, rho, True, int(interior.size), measure,
            f"the displacement satisfies the discrete elasticity equations at "
            f"this system's round-off floor ({rho:.3e} <= {floor:.3e})",
            status="solves")
    if rho >= FORGERY_RESIDUAL_MIN:
        return ResidualVerdict(
            True, rho, False, int(interior.size), measure,
            f"the displacement does NOT satisfy the discrete elasticity "
            f"equations (relative residual {rho:.3e} >= "
            f"{FORGERY_RESIDUAL_MIN:.0e}); it was not obtained by solving this "
            f"problem on this mesh", status="does_not_solve")
    return ResidualVerdict(
        True, rho, None, int(interior.size), measure,
        f"INCONCLUSIVE: relative residual {rho:.3e} lies between this system's "
        f"round-off floor ({floor:.3e}) and the level no solver leaves "
        f"({FORGERY_RESIDUAL_MIN:.0e}); neither certified nor rejected",
        status="inconclusive")


def _build_mesh(points: np.ndarray, cells, dim: int):
    """Construct a scikit-fem mesh from delivered points/cells.

    Cell blocks are normalised through the fabrication gate's helper rather
    than indexed directly. Indexing is what broke here: meshio returns
    CellBlock objects, which are not subscriptable, so `c[0]` raised TypeError
    on every real artefact while the module's own tests — which build cells as
    plain tuples — kept passing. The check was sound and unreachable.
    """
    from skfem import MeshTri, MeshTet

    from .fabrication_gate import _cell_arrays
    blocks = _cell_arrays(cells)
    tri = [arr for name, arr in blocks if name.startswith("triangle")]
    tet = [arr for name, arr in blocks if name.startswith("tetra")]
    p = np.asarray(points, float)
    if dim == 2 and tri:
        t = np.vstack(tri)
        return MeshTri(p[:, :2].T.copy(), t.T.copy())
    if dim == 3 and tet:
        t = np.vstack(tet)
        return MeshTet(p[:, :3].T.copy(), t.T.copy())
    return None


# STOKES IS NOT COVERED, AND THE ATTEMPT IS INSTRUCTIVE.
#
# A `check_stokes_residual` was written here and then removed, because the
# approach cannot work and it is better to say so than to ship a check that
# cannot discriminate.
#
# The residual is only meaningful against the SAME discrete system that was
# solved. For diffusion and elasticity a nodal (P1) field and a P1 assembly are
# the same system, so the check is exact. Stokes is not: a genuine Stokes
# solution comes from an inf-sup-STABLE pair — Taylor-Hood P2/P1 — while a vtu
# carries nodal values, and a P1/P1 assembly is unstable. Measured: the P1/P1
# reference system is exactly singular, and a real Taylor-Hood solution sampled
# at the nodes has no reason to satisfy it. In the attempt a genuine solve came
# back INCONCLUSIVE and a divergent field u = (x, y) came back INCONCLUSIVE
# too — the check separated nothing.
#
# Reconstructing the stable system from nodal output is impossible: the P2
# edge degrees of freedom are not in the file. So Stokes needs either the
# solver's own matrix or a different family of evidence (a mesh-refinement
# study, a divergence norm, a benchmark quantity), not this one.
#
# `check_run_residual` therefore reports UNSUPPORTED for saddle-point problems,
# which is honest and is not protection. Anyone extending this should start from
# why the above fails rather than from the assembly.


def _is_tensor(k, dim: int) -> bool:
    """Is this coefficient a dim x dim tensor rather than a scalar field?

    Checked structurally, not by numpy shape: each entry is itself a field over
    the quadrature points, so a tensor is a nested sequence of arrays and a
    scalar field is one array whose leading dimension happens to be the element
    count. Those are indistinguishable by shape alone on some meshes.
    """
    if isinstance(k, (list, tuple)):
        return len(k) == dim and all(
            isinstance(row, (list, tuple)) and len(row) == dim for row in k)
    return False


def check_residual(points, cells, values, *, dim: int,
                   source_fn, coeff_fn=None,
                   advection_fn=None, reaction_fn=None,
                   domain_measure_expected: float | None = None,
                   ) -> ResidualVerdict:
    """Measure how well `values` satisfies the stated scalar problem on this mesh.

    The operator is the general second-order linear scalar one,

        -div(K grad u) + b . grad u + c u = f

    which is not generality for its own sake: it is five of the PDE classes the
    catalog actually carries, in one assembly. Poisson and steady heat are
    K only; convection-diffusion adds b; reaction-diffusion adds positive c;
    Helmholtz is c = -k^2, which is indefinite and must therefore be allowed to
    be negative. Before this, everything except pure diffusion came back
    UNSUPPORTED — honest, but no protection at all against a fabricated result
    on any of them.

    source_fn(x)    -> f at coordinates x (array shape (dim, n))
    coeff_fn(x)     -> K at x: either a scalar field, or a dim x dim nested
                       sequence of scalar fields for an anisotropic tensor.
                       None means K = 1.
    advection_fn(x) -> b at x, a dim-component field. None means no advection.
    reaction_fn(x)  -> c at x, a scalar field. None means no reaction term.

    The tensor form matters because anisotropic diffusion is where a scalar
    coefficient quietly gives the wrong answer rather than an error: assembling
    with the trace, or with K[0][0], produces a perfectly well-formed operator
    for a different problem, and every residual measured against it is
    meaningless in the direction that lets a forgery through.
    """
    if not _require_skfem():
        return ResidualVerdict(False, None, None, 0, None,
                               "scikit-fem unavailable; cannot assemble")

    from skfem import Basis, ElementTriP1, ElementTetP1, asm, BilinearForm, LinearForm
    from skfem.helpers import dot, grad

    pts = np.asarray(points, float)
    vals = np.asarray(values, float).reshape(-1)
    if pts.shape[0] != vals.shape[0]:
        return ResidualVerdict(False, None, None, 0, None,
                               "field length does not match point count")

    mesh = _build_mesh(pts, cells, dim)
    if mesh is None:
        return ResidualVerdict(False, None, None, 0, None,
                               f"no simplicial cells of dimension {dim} in the delivered mesh")

    elem = ElementTriP1() if dim == 2 else ElementTetP1()
    basis = Basis(mesh, elem)

    # Domain measure: a field on the wrong domain is not a solution of the
    # stated problem, however well-formed it looks.
    # integrand must carry the basis function: sum_i integral(phi_i) = |domain|.
    # Writing `1.0 + 0.0*v` drops it and yields (dofs per cell) x |domain|.
    one = LinearForm(lambda v, w: v)
    measure = float(asm(one, basis).sum())
    if domain_measure_expected is not None:
        rel = abs(measure - domain_measure_expected) / max(
            abs(domain_measure_expected), 1e-30)
        if rel > DOMAIN_TOL:
            return ResidualVerdict(
                True, None, False, 0, measure,
                f"mesh covers measure {measure:.6g}, but the stated domain has "
                f"{domain_measure_expected:.6g} (relative difference {rel:.3g}): "
                f"this is not the problem's domain")

    @BilinearForm
    def a_form(u, v, w):
        # Diffusion.
        if coeff_fn is None:
            out = dot(grad(u), grad(v))
        else:
            k = coeff_fn(w.x)
            if _is_tensor(k, dim):
                gu, gv = grad(u), grad(v)
                out = sum(k[i][j] * gu[j] * gv[i]
                          for i in range(dim) for j in range(dim))
            else:
                out = k * dot(grad(u), grad(v))
        # Advection: b . grad(u) v
        if advection_fn is not None:
            b_ = advection_fn(w.x)
            gu = grad(u)
            out = out + sum(b_[i] * gu[i] for i in range(dim)) * v
        # Reaction: c u v. Negative c is how Helmholtz (-lap u - k^2 u) is
        # expressed, and it is the indefinite case, so it must be allowed.
        if reaction_fn is not None:
            out = out + reaction_fn(w.x) * u * v
        return out

    @LinearForm
    def l_form(v, w):
        return source_fn(w.x) * v

    A = asm(a_form, basis)
    b = asm(l_form, basis)

    if vals.shape[0] != A.shape[0]:
        return ResidualVerdict(False, None, None, 0, measure,
                               "degrees of freedom do not match the delivered field")

    boundary = basis.get_dofs().flatten()
    interior = np.setdiff1d(np.arange(A.shape[0]), boundary)
    if interior.size == 0:
        return ResidualVerdict(True, None, False, 0, measure,
                               "no interior degrees of freedom to test")

    r = A @ vals - b
    num = float(np.linalg.norm(r[interior]))

    # Normalising by ||b|| alone is wrong, and wrong in the direction that
    # accuses honest work. A source-free problem driven purely by Dirichlet data
    # — Laplace with non-zero boundary values, one of the most ordinary things
    # in finite elements — has b = 0 on the interior rows, so the ratio was
    # inf and a solve accurate to 2e-15 was reported as "not obtained by
    # solving this problem".
    #
    # The scale that does not vanish is the size of the terms BEFORE they
    # cancel: |A| |u| is what the residual is small compared to. Using the
    # larger of that and ||b|| keeps the familiar meaning when a source is
    # present and stays finite when it is not.
    scale = float(np.linalg.norm((abs(A) @ np.abs(vals))[interior]))
    den = max(float(np.linalg.norm(b[interior])), scale)
    rho = num / den if den > 0 else (0.0 if num == 0 else float("inf"))

    floor = max(ROUNDOFF_FLOOR_MIN,
                ROUNDOFF_SAFETY * math.sqrt(interior.size) * np.finfo(float).eps)

    if rho <= floor:
        return ResidualVerdict(
            True, rho, True, int(interior.size), measure,
            f"the field satisfies the discrete equations at the round-off "
            f"floor of this system (relative residual {rho:.3e} <= "
            f"{floor:.3e}): it was obtained by solving them",
            status="solves")
    if rho >= FORGERY_RESIDUAL_MIN:
        return ResidualVerdict(
            True, rho, False, int(interior.size), measure,
            f"the field does NOT satisfy the discrete equations (relative "
            f"residual {rho:.3e} >= {FORGERY_RESIDUAL_MIN:.0e}, far above "
            f"anything a solver leaves); it was not obtained by solving this "
            f"problem on this mesh",
            status="does_not_solve")
    return ResidualVerdict(
        True, rho, None, int(interior.size), measure,
        f"INCONCLUSIVE: the relative residual is {rho:.3e}, between this "
        f"system's round-off floor ({floor:.3e}) and the level no solver "
        f"leaves ({FORGERY_RESIDUAL_MIN:.0e}). A loosely-converged solve and a "
        f"very fine-mesh analytic field both land here, so openPASO neither "
        f"certifies nor rejects this field. Solve to a tighter tolerance to "
        f"obtain a verdict",
        status="inconclusive")
