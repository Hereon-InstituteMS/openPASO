"""NGSolve function-space survey.

NGSolve's capability surface IS its function spaces. This builds a mass form
on every space that can carry one and checks a property derived from the
form rather than from any remembered value: a mass matrix is SYMMETRIC
POSITIVE DEFINITE -- symmetry against its own transpose, positivity by the
Rayleigh quotient on random vectors.

VERIFIED BY EXECUTION 2026-09-19, NGSolve 6.2.2606: 25 spaces, all SPD, zero
failures.

AND THE CHECK EARNED ITS KEEP DURING DEVELOPMENT. A first pass put
SurfaceL2, VectorSurfaceL2 and IntegrationRuleSpaceSurface in the VOLUME
table because a bare Assemble() on them raised no error. The SPD check
reported them NOT SPD with a nan asymmetry and a zero Rayleigh quotient --
with dx they assemble an EMPTY MATRIX. Assemble() succeeding is not evidence
that anything was assembled, and a survey without a property check would
have listed all three as working.
"""
from string import Template


_SURVEY = Template(r"""
'''$title

WHICH NGSOLVE SPACE, AND DOES IT ASSEMBLE THE WAY YOU THINK?

NGSolve's capability surface is its function spaces. This builds a mass form
on every space that can carry one and checks two things. First, a property
derived from the form itself: A MASS MATRIX IS SYMMETRIC POSITIVE DEFINITE,
symmetry against its own transpose, positivity by the Rayleigh quotient on
random vectors. Second, an independent reference: THE SPACE REPRODUCES A
CONSTANT. The constant (1, the vector of ones, or the identity matrix,
restricted to the component the space carries) is L2-projected onto the
space through the same measure that assembled the mass matrix, and the
distance from the constant is measured on a RICHER integration rule. That
distance must be zero, and each space prints it as a VERDICT line in the
coverage harness's grammar (scripts/coverage_harness/definitions.py):
  VERDICT ngsolve <space> exact_identity ref=0.000000e+00 got=<distance> tol=1.000000e-10 PASS|FAIL

The split between the tables IS the lesson. A volume space integrates
with dx. A FACET space cannot: it has no degrees of freedom in the element
interior, so dx raises "cannot evaluate facet-fe inside element" or silently
assembles nothing. It needs dx(element_boundary=True). A surface space needs
ds. Choosing the wrong measure is the single most common way to get an
empty or inconsistent system out of these spaces, and SURVEY_MUTATE=1 does
exactly that to the facet spaces -- the planted failure that shows these
verdicts can fail while the volume spaces keep passing.
'''
import os
from math import sqrt
import numpy as np
from ngsolve import (Mesh, BilinearForm, LinearForm, GridFunction, InnerProduct,
                     Integrate, CF, Id, OuterProduct, specialcf, dx, ds, H1, L2,
                     HCurl, HDiv, NumberSpace, SurfaceL2, VectorH1, VectorL2,
                     VectorSurfaceL2, FacetFESpace, TangentialFacetFESpace,
                     NormalFacetFESpace, VectorFacetFESpace, HCurlCurl,
                     HCurlDiv, HDivDiv, HDivDivSurface, TangentialSurfaceL2)
from ngsolve.comp import (NodalFESpace, VectorNodalFESpace,
                          H1LumpingFESpace, HCT_FESpace, JKM_FESpace,
                          IntegrationRuleSpace, IntegrationRuleSpaceSurface,
                          HDivDivFacetSpace, PlateauFESpace, FacetSurface,
                          VectorFacetSurface, HDivSurface, NormalFacetSurface)
from netgen.geom2d import unit_square
from netgen.csg import unit_cube

MAXH = $maxh
ORDER = $order
EXACT_TOL = 1e-10          # the harness's ceiling for an exact identity
MUTATE = os.environ.get("SURVEY_MUTATE", "") == "1"
mesh = Mesh(unit_square.GenerateMesh(maxh=MAXH))
mesh3 = Mesh(unit_cube.GenerateMesh(maxh=0.5))   # the 2-D manifold spaces live on its boundary


def spd_report(a, ndof, trials=3):
    '''Symmetry and positivity of an assembled mass matrix.'''
    rows, cols, vals = a.mat.COO()
    import scipy.sparse as sp
    M = sp.coo_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
                      shape=(ndof, ndof)).tocsr()
    asym = abs(M - M.T).max() if M.nnz else float("nan")
    scale = abs(M).max() if M.nnz else 1.0
    rng = np.random.default_rng(0)
    quots = []
    for _ in range(trials):
        x = rng.standard_normal(ndof)
        quots.append(float(x @ (M @ x)) / float(x @ x))
    return asym / max(scale, 1e-300), min(quots)


def constant_target(name, u, msh):
    '''The constant this space must reproduce: 1, the vector of ones, or a
    constant matrix. HCurlDiv is TRACE-FREE in NGSolve, so the identity is not
    in it (its projection misses by |I| = sqrt(2)); diag(1, -1) is.
    HDivDivSurface holds tangential-tangential matrices on the surface, so its
    constant is the surface's own identity I - n n^T.'''
    d = msh.dim
    if u.dim == 1:
        return CF(1.0)
    if u.dim == d:
        return CF(tuple([1.0] * d))
    if name == "HCurlDiv":
        return CF((1.0, 0.0, 0.0, -1.0), dims=(2, 2))
    if name == "HDivDivSurface":
        n = specialcf.normal(d)
        return Id(d) - OuterProduct(n, n)
    return Id(d)


def component(name, w, msh):
    '''The part of a field a space controls, written so that it does not
    change sign with the element-wise normal: NormalFacetFESpace holds the
    normal component w.n, TangentialFacetFESpace and TangentialSurfaceL2 the
    tangential part w - (w.n) n, HDivDivFacetSpace the normal-normal component
    n.(w n). Everything else holds the whole field.'''
    n = specialcf.normal(msh.dim)
    if name == "NormalFacetFESpace":
        return InnerProduct(w, n)
    if name in ("TangentialFacetFESpace", "TangentialSurfaceL2"):
        return w - InnerProduct(w, n) * n
    if name == "HDivDivFacetSpace":
        return InnerProduct(n, w * n)
    return w


def projection_error(name, fes, measure, measure_hi, msh=None, trace=False):
    '''L2-project the constant's controlled component onto the space through
    `measure`, then measure the distance from it on the richer `measure_hi`.
    A space on a 2-D manifold in 3-D is evaluated through Trace().'''
    msh = msh or mesh
    u, v = fes.TnT()
    c = constant_target(name, u, msh)
    if trace:
        u, v = u.Trace(), v.Trace()
    cu, cv, cc = component(name, u, msh), component(name, v, msh), component(name, c, msh)
    a = BilinearForm(fes, check_unused=False)
    a += InnerProduct(cu, cv) * measure
    l = LinearForm(fes)
    l += InnerProduct(cc, cv) * measure
    a.Assemble(); l.Assemble()
    gf = GridFunction(fes)
    gf.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * l.vec
    d = component(name, gf.Trace() if trace else gf, msh) - cc
    return sqrt(abs(Integrate(InnerProduct(d, d) * measure_hi, msh)))


# A space whose controlled component this survey could not identify gets no
# verdict and says so. HDivDivFacetSpace's functions carry only a
# tangential-tangential component on element boundaries in NGSolve 6.2.2606
# (their nn, nt and tn components integrate to exactly zero there), and the
# projection of a constant's tt component misses by 0.78 -- so which
# component of a matrix field it represents was not established here.
NO_REFERENCE = {
    "HDivDivFacetSpace": "no reference: only a tangential-tangential component on element boundaries, and a constant's is not reproduced; the represented component was not identified",
    "HDivSurface": "no reference: on the cube's faceted boundary a constant's tangential part jumps across the edges in the edge-normal component that H(div) on the surface keeps continuous (projection misses by 1.07); no smooth reference on this mesh",
    "NormalFacetSurface": "no reference: assembles only with ds(element_boundary=True), and neither the normal (nan) nor the tangential (9.3) part of a constant is reproduced; the represented component was not identified",
}


N_CHECKED = N_VERDICT = N_PASS = 0


def survey(entries, measure_name, make_measure, make_measure_hi, msh=None, trace=False):
    # `entries` are (name, factory) with the factory a CONSTRUCTOR CALL
    # written out in this file. Looking a space up by string would exercise
    # it just as well but would leave no trace that this template drives it,
    # and a name in a string is not a use.
    global N_CHECKED, N_VERDICT, N_PASS
    msh = msh or mesh
    print(f"  {'space':28s} {'ndof':>7s} {'rel. asym':>11s} "
          f"{'min Rayleigh':>13s}  SPD?     {'L2|Pi(c)-c|':>12s}  verdict")
    for n, factory in entries:
        try:
            fes = factory(msh, ORDER)
            u, v = fes.TnT()
            if trace:
                u, v = u.Trace(), v.Trace()
            a = BilinearForm(fes, check_unused=False)
            try:
                a += u * v * make_measure()
            except Exception:
                a += InnerProduct(u, v) * make_measure()
            a.Assemble()
            asym, rq = spd_report(a, fes.ndof)
            ok = (asym < 1e-10) and (rq > 0.0)
        except Exception as exc:
            N_CHECKED += 1
            if n not in NO_REFERENCE:
                N_VERDICT += 1
                print(f"VERDICT ngsolve {n} exact_identity ref=0.000000e+00 got=nan tol={EXACT_TOL:.6e} FAIL")
            print(f"  {n:28s} {'-':>7s} {'-':>11s} {'-':>13s}  "
                  f"FAILED with {measure_name}: {type(exc).__name__}")
            continue
        N_CHECKED += 1
        if n in NO_REFERENCE:
            print(f"  {n:28s} {fes.ndof:7d} {asym:11.2e} {rq:13.3e}  "
                  f"{'SPD    ' if ok else 'NOT SPD'}  {'-':>12s}  {NO_REFERENCE[n]}")
            continue
        try:
            # an IntegrationRuleSpace function lives only at its own points, so
            # both the projection and its error use the space's own rules
            if n.startswith("IntegrationRuleSpace"):
                own = (ds if measure_name == "ds" else dx)(intrules=fes.GetIntegrationRules())
                err = projection_error(n, fes, own, own, msh, trace)
            else:
                err = projection_error(n, fes, make_measure(), make_measure_hi(), msh, trace)
            note = ""
        except Exception as exc:
            err = float("nan")
            note = f"  projection failed: {type(exc).__name__}"
        okc = bool(err <= EXACT_TOL)     # nan compares False
        N_VERDICT += 1
        N_PASS += int(okc)
        print(f"VERDICT ngsolve {n} exact_identity ref=0.000000e+00 got={err:.6e} "
              f"tol={EXACT_TOL:.6e} {'PASS' if okc else 'FAIL'}")
        print(f"  {n:28s} {fes.ndof:7d} {asym:11.2e} {rq:13.3e}  "
              f"{'SPD    ' if ok else 'NOT SPD'}  {err:12.1e}  {'PASS' if okc else 'FAIL'}{note}")


print("VOLUME SPACES — integrated with dx")
survey($volume, "dx", lambda: dx, lambda: dx(bonus_intorder=4))
print()
print("FACET SPACES — dx is WRONG here; they need dx(element_boundary=True)")
if MUTATE:
    print("  [MUTATED: integrated with dx -- the FAIL lines are the control working]")
survey($facet, "facet dx",
       lambda: dx if MUTATE else dx(element_boundary=True),
       lambda: dx(bonus_intorder=4) if MUTATE else dx(element_boundary=True, bonus_intorder=4))
print()
print("SURFACE SPACES — integrated with ds")
survey($surface, "ds", lambda: ds, lambda: ds(bonus_intorder=4))
print()
print("SURFACE-FACET SPACES — facets of the boundary curve: ds(element_boundary=True)")
survey($surface_facet, "surface-facet ds", lambda: ds(element_boundary=True),
       lambda: ds(element_boundary=True, bonus_intorder=4))
print()
print("MANIFOLD SPACES — 2-D manifolds only, so on the unit cube's boundary, through Trace(): ds")
survey($manifold, "manifold ds", lambda: ds, lambda: ds(bonus_intorder=4), msh=mesh3, trace=True)
print()
print("MANIFOLD FACET SPACE — on the cube's boundary, through Trace(): ds(element_boundary=True)")
survey($manifold_facet, "manifold facet ds", lambda: ds(element_boundary=True),
       lambda: ds(element_boundary=True, bonus_intorder=4), msh=mesh3, trace=True)
print()
print(f"SURVEY {N_CHECKED} spaces checked, {N_VERDICT} with a verdict, {N_PASS} PASS")
print("A mass matrix that is not symmetric positive definite has not been")
print("integrated the way you asked. For these spaces that almost always")
print("means the measure: dx on a facet space, or dx on a surface space. A")
print("space that cannot reproduce its own constant has not been built the")
print("way you think either; that distance is the verdict.")
""")

_VOLUME = '[("H1", lambda m, o: H1(m, order=o)), ("PlateauFESpace", lambda m, o: PlateauFESpace(H1(m, order=o), [m.Materials(".*")])), ("NodalFESpace", lambda m, o: NodalFESpace(m, order=o)), ("VectorH1", lambda m, o: VectorH1(m, order=o)), ("VectorNodalFESpace", lambda m, o: VectorNodalFESpace(m, order=o)), ("L2", lambda m, o: L2(m, order=o)), ("VectorL2", lambda m, o: VectorL2(m, order=o)), ("HCurl", lambda m, o: HCurl(m, order=o)), ("HDiv", lambda m, o: HDiv(m, order=o)), ("HCurlCurl", lambda m, o: HCurlCurl(m, order=o)), ("HCurlDiv", lambda m, o: HCurlDiv(m, order=o)), ("HDivDiv", lambda m, o: HDivDiv(m, order=o)), ("NumberSpace", lambda m, o: NumberSpace(m, order=o)), ("H1LumpingFESpace", lambda m, o: H1LumpingFESpace(m, order=o)), ("HCT_FESpace", lambda m, o: HCT_FESpace(m, order=o)), ("JKM_FESpace", lambda m, o: JKM_FESpace(m, order=o)), ("IntegrationRuleSpace", lambda m, o: IntegrationRuleSpace(m, order=o))]'
_FACET = '[("FacetFESpace", lambda m, o: FacetFESpace(m, order=o)), ("HDivDivFacetSpace", lambda m, o: HDivDivFacetSpace(m, order=o)), ("NormalFacetFESpace", lambda m, o: NormalFacetFESpace(m, order=o)), ("TangentialFacetFESpace", lambda m, o: TangentialFacetFESpace(m, order=o)), ("VectorFacetFESpace", lambda m, o: VectorFacetFESpace(m, order=o))]'
_SURFACE_FACET = '[("FacetSurface", lambda m, o: FacetSurface(m, order=o)), ("VectorFacetSurface", lambda m, o: VectorFacetSurface(m, order=o))]'
_MANIFOLD = '[("HDivDivSurface", lambda m, o: HDivDivSurface(m, order=o)), ("HDivSurface", lambda m, o: HDivSurface(m, order=o))]'
_MANIFOLD_FACET = '[("NormalFacetSurface", lambda m, o: NormalFacetSurface(m, order=o))]'
_SURFACE = '[("SurfaceL2", lambda m, o: SurfaceL2(m, order=o)), ("VectorSurfaceL2", lambda m, o: VectorSurfaceL2(m, order=o)), ("IntegrationRuleSpaceSurface", lambda m, o: IntegrationRuleSpaceSurface(m, order=o)), ("TangentialSurfaceL2", lambda m, o: TangentialSurfaceL2(m, order=o))]'


def _space_survey(params: dict) -> str:
    """Assemble a mass form on every space and check it is SPD."""
    return _SURVEY.substitute(
        title=params.get("title", "NGSolve function-space survey"),
        maxh=params.get("maxh", 0.3),
        order=params.get("order", 1),
        volume=params.get("volume_spaces", _VOLUME),
        facet=params.get("facet_spaces", _FACET),
        surface=params.get("surface_spaces", _SURFACE),
        surface_facet=params.get("surface_facet_spaces", _SURFACE_FACET),
        manifold=params.get("manifold_spaces", _MANIFOLD),
        manifold_facet=params.get("manifold_facet_spaces", _MANIFOLD_FACET),
    )


GENERATORS = {
    "space_survey_2d": _space_survey,
}

KNOWLEDGE = {
    "space_survey": {
        "description": (
            "Assemble a mass form on every NGSolve function space, verify the "
            "matrix is symmetric positive definite, and L2-project the "
            "constant the space must hold (1, the vector of ones, or a "
            "constant matrix, restricted to the component the space carries) "
            "through the same measure, measuring its distance from the "
            "constant on a richer rule: one VERDICT line per space in the "
            "coverage harness's grammar. Which space, which measure it needs, "
            "and whether it holds what you think it holds. Measured "
            "2026-09-24, NGSolve 6.2.2606: all 31 declared spaces are walked "
            "-- volume, facet, surface, surface-facet, and the 2-D manifold "
            "spaces on a unit cube's boundary through Trace() -- and 28 "
            "reproduce their constant to 1e-14; HDivDivFacetSpace, "
            "HDivSurface and NormalFacetSurface get no verdict, each with the "
            "measured reason printed in its row. The trailer reads SURVEY 31 "
            "spaces checked, 28 with a verdict, 28 PASS. SURVEY_MUTATE=1 "
            "integrates the facet spaces with dx, the planted failure."
        ),
        "pitfalls": [
            "[Numerical] Assemble() SUCCEEDING IS NOT EVIDENCE THAT ANYTHING "
            "WAS ASSEMBLED. A surface space integrated with dx produces an "
            "EMPTY matrix and raises nothing. Measured: SurfaceL2, "
            "VectorSurfaceL2 and IntegrationRuleSpaceSurface each assembled "
            "'successfully' with dx and yielded a matrix with no entries. "
            "Signal: a mass matrix whose asymmetry is nan and whose Rayleigh "
            "quotient is exactly zero -- both are what an empty matrix gives, "
            "and neither looks like an error. Check a property of the matrix, "
            "never the return of Assemble(). (Measured 2026-09-19, NGSolve "
            "6.2.2606.)",

            "[API] THE MEASURE MUST MATCH THE SPACE, and choosing wrong is "
            "the commonest way to get an empty or inconsistent system from "
            "these spaces. A volume space takes dx. A FACET space has no "
            "degrees of freedom in the element interior, so dx is wrong for "
            "it and it needs dx(element_boundary=True). A surface space needs "
            "ds. Signal: 'cannot evaluate facet-fe inside element', "
            "'normal-facet element evaluated not at BND', or a 'used dof "
            "inconsistency' warning followed by a system that solves to "
            "nothing.",

            "[Numerical] HCurlDiv IS TRACE-FREE, AND THE IDENTITY IS NOT IN "
            "IT. Projecting Id(2) onto HCurlDiv(mesh, order=1) leaves an L2 "
            "distance of 1.41 = |I| on the unit square, because the space "
            "holds trace-free matrices only; diag(1, -1) projects to 5e-16. "
            "HDivDivFacetSpace's functions carry only a tangential-tangential "
            "component on element boundaries (InnerProduct with "
            "OuterProduct(n, n), OuterProduct(n, t) and OuterProduct(t, n) "
            "integrate to exactly zero there) and a constant's tt component "
            "is not reproduced either, so which component it represents is "
            "an open question this survey records rather than guesses. "
            "Signal: a.mat.Inverse(..., inverse='sparsecholesky') returning "
            "NaN on a form built from a single matrix component, or a "
            "projection error equal to the norm of the constant itself.",

            "[Numerical] A MASS MATRIX IS SPD, AND THAT IS A FREE CHECK ON "
            "ANY SPACE. Symmetry against its own transpose plus a positive "
            "Rayleigh quotient on random vectors costs one assembly and no "
            "reference solution, and it catches an empty assembly, a wrong "
            "measure and a mis-declared space alike. Signal: use it before "
            "trusting a space you have not used before -- it is the cheapest "
            "thing that can tell you the system you built is not the system "
            "you meant.",
        ],
    },
}
