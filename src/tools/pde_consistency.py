"""Does the delivered field actually satisfy the equation the task stated?

WHY THIS EXISTS. Measured over 464 single-code runs, among
those with a complete level set the SELF-convergence order — computed
from the agent's own numbers, no reference — has a median of 1.96 (bare) and
1.99 (openPASO). The discretisations converge cleanly. So an order near zero
against an independent reference is almost never the finite element method
failing to converge; it is a field converging beautifully TO THE WRONG
FUNCTION (7% of all runs), or a field whose overall size is wrong by orders
of magnitude (11-30% of runs).

A refinement study cannot see either one, and neither can the agent's own
verdict: MESH_INDEPENDENCE = NOT_CONVERGED catches about three quarters of the
wrong runs but also fires on HALF the correct ones, so on its own it is close to
uninformative.

WHAT THIS CHECKS, AND WHY IT LEAKS NO REFERENCE SOLUTION. For any smooth test
function v that vanishes on the boundary, a field u solving

    L u = -div(K grad u) = f          with u prescribed on the boundary

satisfies the weak identity

    integral of u * (L* v)  ==  integral of f * v

where L* is the adjoint, and L* = L for symmetric constant K. Every ingredient
is PUBLIC: the operator and the source come from the task text, and the values
come from the agent's own run. No exact solution is used, none is
revealed, and nothing here can be run backwards to obtain one — a single
scalar identity per test function cannot reconstruct a field.

MEASURED SEPARATION, by execution on 26 anisotropic-Poisson runs, whose probe grid
is a 44x44 midpoint rule so the quadrature is exact to O(h^2):

    solves the stated PDE   2.32e-02 -> 5.27e-03 -> 1.23e-03 -> 2.56e-04
    does not                6.25e+00 -> 6.37e+00 -> 6.40e+00 -> 6.40e+00
    does not (other cause)  3.13e+00 -> 3.01e+00 -> 2.96e+00 -> 2.93e+00

Four orders of magnitude apart at the finest level, and the failing ones are
flat, which is the signature: a wrong operator, a wrong source, or a source
evaluated in the wrong coordinate frame leaves a residual that refinement
cannot remove.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LevelResult:
    level: int
    n_points: int
    residual: float
    detail: str = ""


@dataclass
class ConsistencyResult:
    levels: list = field(default_factory=list)
    rate: float | None = None
    verdict: str = "UNKNOWN"
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "observed_rate": self.rate,
            "levels": [{"level": r.level, "points": r.n_points,
                        "relative_weak_residual": r.residual,
                        "detail": r.detail} for r in self.levels],
        }


def _detect_midpoint_grid(coords: list) -> tuple:
    """Is this a tensor grid of cell midpoints, and what is the cell volume?

    The quadrature below is a midpoint rule, which is second-order accurate on
    exactly this arrangement and meaningless on a scatter. Returning the weight
    ONLY when the arrangement is right is what keeps a number from being
    reported that the method cannot support.
    """
    axes = []
    for d in range(len(coords[0])):
        vals = sorted({round(p[d], 12) for p in coords})
        if len(vals) < 2:
            return None, "an axis carries a single distinct coordinate"
        steps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        h = sum(steps) / len(steps)
        if h <= 0 or max(abs(s - h) for s in steps) > 1e-9 * max(h, 1e-30):
            return None, "the points are not uniformly spaced along every axis"
        axes.append((vals, h))
    if len(coords) != math.prod(len(v) for v, _ in axes):
        return None, "the points do not fill a full tensor grid"
    weight = 1.0
    for _, h in axes:
        weight *= h
    return weight, ""


def _adjoint_of_v_flat(coeff, pts, box):
    """L* v and v for a v whose VALUE AND SLOPE both vanish on the box.

    WHY THIS v AND NOT THE OBVIOUS ONE. Integrating the identity by parts twice
    leaves two boundary terms:

        boundary integral of  -K grad(u).n v   -> zero because v = 0
        boundary integral of   u K grad(v).n   -> survives unless u = 0 there

    The natural choice, a product of half-period sines, is zero on every face
    but its slope is not, so the second term only vanishes when the FIELD is
    zero on the boundary -- which one side of a coupled problem never is,
    because the interface carries the partner's data. That version refused
    nearly every coupled side: measured across 94 graded coupled cells, side B
    came back NOT_APPLICABLE 92 times.

    v = prod sin^2 has value and normal derivative both zero on every face, so
    BOTH terms vanish for any u at all, and it is used for every case.

    Measured on a manufactured case deliberately not zero on the boundary
    (u = sin(pi x) sin(pi y) + x, so -lap u = 2 pi^2 sin sin), levels 16 to 128:

        v                 correct field          wrong field (0.5x amplitude)
        sines             8.13e-01 flat          3.13e-01 flat
        sin^2 (here)      4.87e-03 -> 7.53e-05   5.02e-01 flat
                          falls 64.7x

    The sines cannot tell them apart at all, and rank the correct field WORSE.
    sin^2 separates them, and the fall is the documented factor of four per
    refinement. It also holds at k = 200 on an off-unit box, catches a 3 %
    amplitude error (3.47e-02 flat), and catches the near-zero field shape that
    C9 keeps producing. Where u IS zero on the boundary the sines are exact
    (residual 1.8e-16) and this one merely converges -- that is the whole cost,
    and the verdict reads the fall rather than an absolute floor.

    Replayed over the 94 graded coupled cells of the accepted operator
    (benchmarks/pde_check_calibration):

        v                          false alarms   wrong caught   no verdict
        sines                          1 of 14        6 of 17           78
        sin^2 behind a size test       1 of 14       16 of 17           35
        sin^2 always (this)            0 of 14       14 of 17           35
    """
    import numpy as np
    dim = len(box)
    xs = [np.asarray([p[d] for p in pts], dtype=float) for d in range(dim)]
    Ls = [float(hi - lo) for lo, hi in box]
    args = [math.pi * (xs[d] - box[d][0]) / Ls[d] for d in range(dim)]
    sin2 = [np.sin(a) ** 2 for a in args]

    v = np.ones_like(xs[0])
    for s in sin2:
        v = v * s

    # d2/dx_i^2 of sin^2(a_i) = 2 (pi/L_i)^2 cos(2 a_i); the cross terms
    # d2/dx_i dx_j carry sin(2a_i) sin(2a_j) (pi/L_i)(pi/L_j).
    def d2(i, j):
        out = np.ones_like(xs[0])
        for d in range(dim):
            if i == j:
                out = out * (2.0 * (math.pi / Ls[d]) ** 2 * np.cos(2 * args[d])
                             if d == i else sin2[d])
            elif d == i or d == j:
                out = out * (math.pi / Ls[d]) * np.sin(2 * args[d])
            else:
                out = out * sin2[d]
        return out

    K = _as_tensor(coeff, dim)
    Lv = np.zeros_like(xs[0])
    for i in range(dim):
        for j in range(dim):
            if K[i][j] != 0.0:
                Lv = Lv - K[i][j] * d2(i, j)
    return v, Lv


def _as_tensor(coeff, dim: int) -> list:
    if isinstance(coeff, (int, float)):
        return [[float(coeff) if i == j else 0.0 for j in range(dim)]
                for i in range(dim)]
    rows = [[float(c) for c in row] for row in coeff]
    if len(rows) != dim or any(len(r) != dim for r in rows):
        raise ValueError(f"coefficient tensor is not {dim}x{dim}")
    for i in range(dim):
        for j in range(i + 1, dim):
            if abs(rows[i][j] - rows[j][i]) > 1e-12 * max(1.0, abs(rows[i][j])):
                raise ValueError(
                    "the coefficient tensor is not symmetric; this check "
                    "assumes a self-adjoint operator, so a non-symmetric K "
                    "needs the true adjoint and is refused rather than "
                    "answered wrongly")
    return rows


def _eval_source(expr: str, pts, dim: int):
    """Evaluate the task's source expression at the points, in PYTHON syntax."""
    import numpy as np
    names = ["x", "y", "z"][:dim]
    env = {n: np.asarray([p[d] for p in pts], dtype=float)
           for d, n in enumerate(names)}
    env.update({"np": np, "pi": math.pi, "sin": np.sin, "cos": np.cos,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "tan": np.tan, "tanh": np.tanh, "sinh": np.sinh,
                "cosh": np.cosh, "abs": np.abs, "Abs": np.abs})
    out = eval(expr, {"__builtins__": {}}, env)              # noqa: S307
    return np.asarray(out, dtype=float) * np.ones_like(env[names[0]])


def check_levels(levels: dict, source_expr: str, coefficient,
                 box: list) -> ConsistencyResult:
    """levels maps a level number to a list of (coords..., u) rows."""
    import numpy as np
    res = ConsistencyResult()
    for lvl in sorted(levels):
        rows = levels[lvl]
        if not rows:
            res.levels.append(LevelResult(lvl, 0, float("nan"),
                                          "no rows"))
            continue
        dim = len(rows[0]) - 1
        pts = [r[:dim] for r in rows]
        u = np.asarray([r[dim] for r in rows], dtype=float)
        if not np.all(np.isfinite(u)):
            res.levels.append(LevelResult(
                lvl, len(rows), float("nan"),
                "the delivered field carries a non-finite value"))
            continue
        weight, why = _detect_midpoint_grid(pts)
        if weight is None:
            res.levels.append(LevelResult(lvl, len(rows), float("nan"), why))
            continue
        # THE IDENTITY NEEDS u = 0 ON THE BOUNDARY, NOT ONLY v = 0.
        #
        # Integrating by parts twice leaves
        #     int u (L* v) - int (L u) v = - closed_int u k dv/dn + closed_int v k du/dn
        # and v vanishing on the box kills only the SECOND term. The first
        # survives unless u vanishes there too. So on a subdomain with a
        # non-zero trace on any face -- which is EVERY side of a partitioned
        # coupling, since the interface carries data -- the identity fails for
        # a perfectly correct field.
        #
        # Measured on one coupled problem, whose side B has an interface trace
        # of about -3.7e-03: this check called side B INCONSISTENT for all four
        # runs examined, INCLUDING one verified correct against an independent
        # reference with order 1.95. A false accusation against the one right
        # answer, and the coupling payload was telling agents to run it on
        # each side.
        #
        # Detected from the delivered field alone: on a grid of cell midpoints
        # a field vanishing on the boundary has an outermost layer of size
        # O(h) relative to its own scale, while a face carrying data does not.
        # A FIELD CARRYING DATA ON ITS BOUNDARY IS THE NORMAL COUPLED CASE, AND
        # IT USED TO BE REFUSED. The sines below vanish on every face but their
        # slope does not, so the identity needs u = 0 there too. Measured across
        # 94 graded coupled cells, that refused side B 92 times -- i.e. this
        # check could not speak about one side of almost any coupled problem,
        # which is exactly where it was about to be run by default.
        #
        # `_adjoint_of_v_flat` uses a v whose value AND slope vanish, so both
        # boundary terms go for any u, and it is used UNCONDITIONALLY.
        #
        # IT WAS A FALLBACK BEHIND A SIZE TEST ON THAT OUTERMOST LAYER, AND THE
        # SIZE TEST CANNOT BE MADE TO WORK. On a midpoint grid the outermost
        # probe sits h/2 inside the face, so a field that IS zero on the
        # boundary still reads |grad u| h/2 there -- percent-level, the same
        # order as a face genuinely carrying a partner's data. The two cases
        # are not separable by magnitude. Measured: C2 seed 6751, graded
        # CORRECT, has a layer of 0.067 of its own scale, went to the sines,
        # and was called INCONSISTENT -- 1.32e-02 -> 5.79e-03 -> 1.06e-02, flat
        # and non-monotone. The same three files under the v below fall
        # 1.25e-02 -> 3.76e-03 -> 1.59e-03, monotone, and read CONSISTENT. The
        # field was right the whole time; the routing was wrong.
        #
        # The sines are exact (1.8e-16) where u really does vanish and this v
        # merely converges there, which is the only thing given up, and the
        # verdict reads the FALL not an absolute floor. Convergence is enough.
        v, Lv = _adjoint_of_v_flat(coefficient, pts, box)
        f = _eval_source(source_expr, pts, dim)
        lhs = float(np.sum(u * Lv) * weight)
        rhs = float(np.sum(f * v) * weight)
        denom = max(abs(rhs), 1e-300)
        res.levels.append(LevelResult(lvl, len(rows), abs(lhs - rhs) / denom,
                                      f"lhs={lhs:.6e} rhs={rhs:.6e}"))
    return _decide(res)


def _decide(res: ConsistencyResult) -> ConsistencyResult:
    """Turn per-level residuals into a verdict.

    Shared by the scalar and the elasticity path so one operator cannot drift
    into a different rule than the other: the verdict reads the FALL, and the
    round-off, meaningless-magnitude and too-few-levels branches below are the
    ones both operators need.
    """
    import math
    # A RESIDUAL OF 1e+297 IS NOT A VERDICT. Measured on real runs
    # outside this check's operator, the relative residual came back as
    # 1.197e+297 and 5.190e+293 — the ratio of two quantities that have nothing
    # to do with each other. The magnitudes were reported with a confident
    # INCONSISTENT and CONSISTENT respectively. A number that large means the
    # comparison is meaningless, not that the field is very wrong.
    for r in res.levels:
        if math.isfinite(r.residual) and abs(r.residual) > 1e6:
            r.detail = (r.detail + " | REFUSED: relative residual "
                        f"{r.residual:.3e} is far outside anything a "
                        "discretisation produces, so the two sides of the "
                        "identity are not comparable — check that the source "
                        "term, the coefficient and the field are the ones the "
                        "task states, and that this is a scalar "
                        "-div(K grad u) = f problem at all")
            r.residual = float("nan")
    good = [r for r in res.levels if math.isfinite(r.residual)]
    if len(good) < 2:
        res.verdict = "NOT_APPLICABLE"
        res.explanation = (
            "fewer than two levels could be checked. This test needs the "
            "prescribed probe points, which form a uniform tensor grid of cell "
            "midpoints; on a scatter the midpoint quadrature has no accuracy "
            "and no number is reported rather than a misleading one."
            + (" " + good[0].detail if good else "")
            + " " + "; ".join(r.detail for r in res.levels if r.detail)[:300])
        return res
    first, last = good[0].residual, good[-1].residual
    if last > 0 and first > 0 and len(good) >= 2:
        span = len(good) - 1
        res.rate = math.log(first / last) / (span * math.log(2.0)) if last else None
    # A RESIDUAL AT ROUND-OFF IS THE IDENTITY HOLDING, NOT A FLAT FAILURE.
    #
    # The decay test asks whether the residual FALLS, which is the right
    # question for a discretisation whose error shrinks with h. It has no
    # answer when the residual is already zero: a field that satisfies the weak
    # identity exactly gives 0.000e+00 at every level, `last < first/3` is
    # false, and the verdict came out INCONSISTENT with the explanation "the
    # weak residual is FLAT: 0.000e+00 -> 0.000e+00" — condemning the one
    # field that could not be more right. Found by writing the test that asks
    # whether a genuine scalar-diffusion case still passes.
    if max(first, last) <= 1e-12:
        res.verdict = "CONSISTENT"
        res.explanation = (
            f"the weak residual is at round-off ({first:.3e} -> {last:.3e}): "
            f"the identity holds as exactly as double precision allows, so "
            f"your field satisfies the equation you were given. Note that this "
            f"is what an ANALYTIC or interpolated exact field also gives — it "
            f"says the operator and source match, not that a solver ran.")
    elif last < first / 3.0:
        res.verdict = "CONSISTENT"
        res.explanation = (
            f"the weak residual falls {first:.3e} -> {last:.3e} across "
            f"{len(good)} levels (rate {res.rate:.2f} per refinement). Your "
            f"field satisfies the equation you were given, so a remaining "
            f"error is discretisation, not a wrong model.")
    else:
        res.verdict = "INCONSISTENT"
        res.explanation = (
            f"the weak residual is FLAT: {first:.3e} -> {last:.3e} over "
            f"{len(good)} levels. Refinement cannot remove it, so the field is "
            f"converging to something that is not the solution of the stated "
            f"problem. Look at the source term first — a sign, a missing term, "
            f"or an expression evaluated in element-local instead of global "
            f"coordinates — then the coefficient, then which boundary carries "
            f"which condition. A run that reports this honestly scores above "
            f"one that submits it as converged.")
    return res


def _adjoint_elastic_flat(lam: float, mu: float, pts, box, direction: int):
    """v and L*v for LINEAR ELASTICITY, v with value and slope zero on the box.

    For constant lambda and mu the elasticity operator is self-adjoint, exactly
    as constant-K diffusion is, so the same weak identity holds componentwise:

        integral u . (L* v)  =  integral f . v ,
        L* v = -div(sigma(v)) = -[ mu lap(v) + (mu + lambda) grad(div v) ]

    and with a v whose value AND normal slope vanish on every face, both
    boundary terms go for any u -- including a coupled side whose interface
    carries the partner's displacement.

    `direction` picks v = (W, 0) or (0, W) with W = prod sin^2, so the two
    calls together test both momentum equations rather than a single mixture
    that a sign error in one component could hide.

    WHY THIS EXISTS. The scalar check REFUSES elasticity, deliberately: handed
    an elasticity result set it once reported the field converging to the wrong
    solution, which was meaningless. That refusal covers the campaign's C9
    family, whose equation is exactly this one -- so the one coupled family
    with a recent CORRECT was the one nothing could check.

    Calibrated on a manufactured case deliberately not zero on the boundary
    (u = (sin(pi x) sin(pi y) + x, (sin(pi x) sin(pi y))/2 + y/2), f computed
    from it symbolically, lambda = 480, mu = 1200), levels 16 to 128, worst of
    the two directions:

        field                      residual                     fall
        exact                      5.578e-03 -> 8.611e-05       64.8x
        3 % low on ux              3.501e-02 -> 3.008e-02        1.2x
        3 % low on uy              3.541e-02 -> 3.008e-02        1.2x
        half amplitude             5.028e-01 -> 5.000e-01        1.0x
        uy dropped entirely        1.000e+00 -> 1.000e+00        1.0x

    The exact field falls at the documented factor of four per refinement and
    every wrong one stays flat, including a 3 % amplitude error in a single
    component.
    """
    import numpy as np

    if len(box) != 2:
        raise ValueError("the elasticity identity here is written for 2D")
    xs = [np.asarray([p[d] for p in pts], dtype=float) for d in range(2)]
    Ls = [float(hi - lo) for lo, hi in box]
    a, b = math.pi / Ls[0], math.pi / Ls[1]
    ax = a * (xs[0] - box[0][0])
    by = b * (xs[1] - box[1][0])
    S, T = np.sin(ax) ** 2, np.sin(by) ** 2
    Sp, Tp = a * np.sin(2 * ax), b * np.sin(2 * by)
    Spp, Tpp = 2 * a * a * np.cos(2 * ax), 2 * b * b * np.cos(2 * by)

    W = S * T
    lap = Spp * T + S * Tpp
    Wxx, Wxy, Wyy = Spp * T, Sp * Tp, S * Tpp

    if direction == 0:                       # v = (W, 0)
        Lvx = -(mu * lap + (mu + lam) * Wxx)
        Lvy = -((mu + lam) * Wxy)
        return (W, np.zeros_like(W)), (Lvx, Lvy)
    Lvx = -((mu + lam) * Wxy)                # v = (0, W)
    Lvy = -(mu * lap + (mu + lam) * Wyy)
    return (np.zeros_like(W), W), (Lvx, Lvy)


def check_levels_elastic(levels: dict, source_x: str, source_y: str,
                         lam: float, mu: float, box: list) -> ConsistencyResult:
    """Does a DISPLACEMENT field satisfy -div(sigma(u)) = f on this box?

    `levels` maps a level number to rows of (x, y, ux, uy). Both momentum
    equations are tested and the WORST of the two decides the level, so a
    component that is right cannot cover for one that is not.
    """
    import numpy as np

    res = ConsistencyResult()
    for lvl in sorted(levels):
        rows = levels[lvl]
        if not rows:
            res.levels.append(LevelResult(lvl, 0, float("nan"), "no rows"))
            continue
        pts = [r[:2] for r in rows]
        ux = np.asarray([r[2] for r in rows], dtype=float)
        uy = np.asarray([r[3] for r in rows], dtype=float)
        if not (np.all(np.isfinite(ux)) and np.all(np.isfinite(uy))):
            res.levels.append(LevelResult(
                lvl, len(rows), float("nan"),
                "the delivered field carries a non-finite value"))
            continue
        weight, why = _detect_midpoint_grid(pts)
        if weight is None:
            res.levels.append(LevelResult(lvl, len(rows), float("nan"), why))
            continue
        fx = _eval_source(source_x, pts, 2)
        fy = _eval_source(source_y, pts, 2)
        worst, detail = -1.0, ""
        for direction, f_here in ((0, fx), (1, fy)):
            (vx, vy), (Lvx, Lvy) = _adjoint_elastic_flat(lam, mu, pts, box,
                                                         direction)
            lhs = float(np.sum(ux * Lvx + uy * Lvy) * weight)
            rhs = float(np.sum(f_here * (vx if direction == 0 else vy))
                        * weight)
            rel = abs(lhs - rhs) / max(abs(rhs), 1e-300)
            if rel > worst:
                worst, detail = rel, (f"worst component is the "
                                      f"{'x' if direction == 0 else 'y'} "
                                      f"momentum equation: "
                                      f"lhs={lhs:.6e} rhs={rhs:.6e}")
        res.levels.append(LevelResult(lvl, len(rows), worst, detail))
    return _decide(res)
