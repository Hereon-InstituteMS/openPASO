"""Make the discrete-residual check reachable from a run.

`core/residual_check.py` can tell a solve from a forgery — it assembles the
stated operator on the delivered mesh and measures how far the delivered field
is from satisfying it, which separates the two by many orders of magnitude. It
was unreachable: no tool took a problem statement, so nothing could ever call
it, and the verification gate's strongest check sat in the repository doing
nothing.

This module is the bridge. A run may declare the PDE it claims to solve, in the
same terms the problem statement is already given in — an operator family, a
source term, a coefficient, the domain's measure. None of that is an answer:
the source term is what the engineer is handed, not what they compute. That
distinction is the whole reason this check is admissible in a gate that must
never see an exact solution. A residual needs f, not u.

WHAT A DECLARATION LOOKS LIKE

    {"operator": "diffusion",
     "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
     "coefficient": "1.0",
     "dim": 2,
     "domain_measure": 1.0}

`source` and `coefficient` are expressions in x, y, z evaluated with numpy and
nothing else — no builtins, no imports, no attribute access. They are numeric
expressions, not a scripting surface.

WHAT IT CANNOT DO
Only the operator families residual_check implements are supported (scalar
diffusion on simplices, P1, Dirichlet data on the outer boundary). Anything else
returns UNSUPPORTED, never a pass — "not checked" must never be reachable as
"checked and fine", because a gate that silently approves what it cannot examine
is worse than one that admits the gap.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from .residual_check import check_residual

# Everything an expression may name. Deliberately small: these are the functions
# a source term is written with, and nothing here can reach the filesystem, the
# interpreter, or any object's attributes.
_SAFE_NAMES = {
    name: getattr(np, name) for name in
    ("sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
     "exp", "log", "log10", "sqrt", "abs", "sign", "minimum", "maximum",
     "power", "floor", "ceil")
}
_SAFE_NAMES.update(pi=np.pi, e=np.e)

# One assembly, five PDE classes. `diffusion` remains the name for the whole
# scalar family because `advection` and `reaction` are optional terms of it:
#   poisson / steady heat        K only
#   convection_diffusion         K, b
#   reaction_diffusion           K, c > 0
#   helmholtz                    K, c = -k^2  (indefinite; c may be negative)
# The aliases exist so an agent can declare the problem by the name it knows it
# by, rather than having to recognise that its problem is "diffusion".
SUPPORTED_OPERATORS = ("diffusion", "elasticity", "convection_diffusion",
                       "reaction_diffusion", "helmholtz", "poisson", "heat")
_SCALAR_ALIASES = {"convection_diffusion", "reaction_diffusion", "helmholtz",
                   "poisson", "heat"}

# Nodes only appear in a legitimate numeric expression. Attribute access,
# subscripting, comprehensions, lambdas, calls to anything not in _SAFE_NAMES,
# and every statement form are absent by construction.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.FloorDiv, ast.Tuple,
)


class ResidualSpecError(ValueError):
    """The declaration itself is malformed — refused before anything runs."""


def _compile_expression(text: str, label: str):
    """Compile a numeric expression, refusing anything that is not one."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ResidualSpecError(f"{label}: not a valid expression ({exc.msg})")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ResidualSpecError(
                f"{label}: {type(node).__name__} is not allowed here; a source "
                f"term is a numeric expression in x, y, z, not a program")
        if isinstance(node, ast.Call):
            fn = node.func
            if not isinstance(fn, ast.Name) or fn.id not in _SAFE_NAMES:
                got = getattr(fn, "id", type(fn).__name__)
                raise ResidualSpecError(
                    f"{label}: '{got}' is not an available function; use one of "
                    + ", ".join(sorted(_SAFE_NAMES)))
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES \
                and node.id not in ("x", "y", "z"):
            raise ResidualSpecError(
                f"{label}: unknown name '{node.id}'; expressions may use the "
                f"coordinates x, y, z and the standard functions")
    return compile(tree, f"<{label}>", "eval")


def _tensor_evaluator(codes, dim: int):
    """Evaluate a dim x dim grid of expressions into nested field arrays."""
    cell = [[_evaluator(codes[i][j], dim) for j in range(dim)]
            for i in range(dim)]
    def evaluate(coords):
        return [[cell[i][j](coords) for j in range(dim)] for i in range(dim)]
    return evaluate


def _evaluator(code, dim: int):
    """Turn a compiled expression into f(X) over coordinates of shape (dim, n)."""
    def evaluate(coords):
        env = dict(_SAFE_NAMES)
        env["x"] = coords[0]
        env["y"] = coords[1] if len(coords) > 1 else np.zeros_like(coords[0])
        env["z"] = coords[2] if len(coords) > 2 else np.zeros_like(coords[0])
        value = eval(code, {"__builtins__": {}}, env)   # noqa: S307
        # A constant expression evaluates to a scalar; the forms need an array.
        return value * np.ones_like(coords[0]) if np.isscalar(value) else value
    return evaluate


def _parse_coefficient(coefficient, dim: int):
    """A coefficient is a scalar expression, or a dim x dim nested list of them.

    Anisotropic diffusion has to be expressible: collapsing a tensor to a scalar
    assembles a well-formed operator for a DIFFERENT problem, and a residual
    measured against the wrong operator is wrong in the permissive direction.
    """
    if coefficient in (None, "", "1", "1.0"):
        return None
    if isinstance(coefficient, (list, tuple)):
        rows = list(coefficient)
        if len(rows) != dim or any(
                not isinstance(r, (list, tuple)) or len(r) != dim for r in rows):
            raise ResidualSpecError(
                f"a tensor coefficient must be {dim}x{dim}; got "
                f"{len(rows)} row(s) of "
                f"{[len(r) if isinstance(r, (list, tuple)) else '?' for r in rows]}")
        return [[_compile_expression(str(rows[i][j]), f"coefficient[{i}][{j}]")
                 for j in range(dim)] for i in range(dim)]
    return _compile_expression(str(coefficient), "coefficient")


def parse_spec(spec: str | dict) -> dict:
    """Validate a residual declaration, raising ResidualSpecError if malformed."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise ResidualSpecError(f"not valid JSON: {exc}")
    if not isinstance(spec, dict):
        raise ResidualSpecError("expected a JSON object describing the problem")

    operator = str(spec.get("operator", "diffusion")).lower()
    if operator not in SUPPORTED_OPERATORS:
        raise ResidualSpecError(
            f"operator '{operator}' is not one openPASO can assemble; supported: "
            + ", ".join(SUPPORTED_OPERATORS))
    if "source" not in spec:
        raise ResidualSpecError(
            "a residual needs the problem's source term; give `source` as an "
            "expression in x, y, z (use \"0\" for a source-free problem)")

    dim = int(spec.get("dim", 2))
    if dim not in (2, 3):
        raise ResidualSpecError(f"dim must be 2 or 3, got {dim}")

    if operator == "elasticity":
        src = spec["source"]
        if not isinstance(src, (list, tuple)) or len(src) != dim:
            raise ResidualSpecError(
                f"elasticity needs a {dim}-component body force; give `source` "
                f"as a list of {dim} expressions, e.g. [\"0\", \"-9.81*rho\"]")
        for key in ("young", "poisson"):
            if key not in spec:
                raise ResidualSpecError(
                    f"elasticity needs `{key}`; openPASO cannot assemble the "
                    f"operator without the material constants")
        return {
            "operator": operator,
            "dim": dim,
            "source_code": [_compile_expression(str(s), f"source[{i}]")
                            for i, s in enumerate(src)],
            "coefficient_code": None,
            "young": float(spec["young"]),
            "poisson": float(spec["poisson"]),
            "domain_measure": (float(spec["domain_measure"])
                               if spec.get("domain_measure") is not None else None),
            "field": str(spec.get("field", "")),
            "result_file": spec.get("result_file"),
        }

    measure = spec.get("domain_measure")
    adv = spec.get("advection")
    if adv is not None:
        if not isinstance(adv, (list, tuple)) or len(adv) != dim:
            raise ResidualSpecError(
                f"`advection` must be a {dim}-component velocity, e.g. "
                f"[\"1\", \"0\"]; got {adv!r}")
        adv = [_compile_expression(str(a), f"advection[{i}]")
               for i, a in enumerate(adv)]
    react = spec.get("reaction")
    if react is not None:
        # Helmholtz is c = -k^2, so a negative reaction coefficient is correct
        # and must not be treated as a malformed declaration.
        react = _compile_expression(str(react), "reaction")
    if operator == "helmholtz" and react is None:
        raise ResidualSpecError(
            "helmholtz needs `reaction`: the operator is -div(K grad u) + c u "
            "with c = -k^2, so give reaction as the negative squared wave "
            "number, e.g. \"-16*pi**2\" for k = 4*pi")
    return {
        "operator": operator,
        "dim": dim,
        "source_code": _compile_expression(str(spec["source"]), "source"),
        "advection_code": adv,
        "reaction_code": react,
        "coefficient_code": _parse_coefficient(spec.get("coefficient"), dim),
        "domain_measure": float(measure) if measure is not None else None,
        "field": str(spec.get("field", "")),
        "result_file": spec.get("result_file"),
    }


def _vector_evaluator(codes, dim: int):
    """Evaluate a dim-component body force into an array of shape (dim, n)."""
    comps = [_evaluator(c, dim) for c in codes]
    def evaluate(coords):
        return np.array([c(coords) for c in comps])
    return evaluate


def _as_result(verdict, field_name: str, artefact) -> dict:
    """One place that turns a ResidualVerdict into the gate's dict, so the
    scalar and vector paths cannot drift apart in what they report."""
    if not verdict.supported:
        return {"verdict": "UNSUPPORTED", "detail": verdict.detail,
                "field": field_name, "artefact": artefact.name}
    return {
        "verdict": {"solves": "SOLVES", "does_not_solve": "DOES_NOT_SOLVE",
                    "inconclusive": "INCONCLUSIVE"}.get(
                        verdict.status,
                        "SOLVES" if verdict.solver_like else "DOES_NOT_SOLVE"),
        "relative_residual": verdict.relative_residual,
        "n_interior_dofs": verdict.n_interior,
        "domain_measure": verdict.domain_measure,
        "field": field_name,
        "artefact": artefact.name,
        "detail": verdict.detail,
    }


def _coefficient_evaluator(code, dim: int):
    if code is None:
        return None
    return (_tensor_evaluator(code, dim) if isinstance(code, list)
            else _evaluator(code, dim))


def check_run_residual(spec: str | dict, result_files) -> dict:
    """Measure whether the run's field satisfies the problem it declared.

    Returns a dict the verification gate can act on. `verdict` is one of:
      SOLVES        — the field satisfies the discrete system to solver tolerance
      DOES_NOT_SOLVE— it does not; this field was not obtained by solving this
                      problem on this mesh
      UNSUPPORTED   — openPASO cannot assemble this problem, so nothing was checked
      REFUSED       — the declaration or the artefacts were unusable

    Never raises: a gate that dies on a malformed declaration is a gate an agent
    can switch off by malforming one.
    """
    from .fabrication_gate import select_artefact
    from .mesh_independence import read_nodal_mesh

    try:
        parsed = parse_spec(spec)
    except ResidualSpecError as exc:
        return {"verdict": "REFUSED", "detail": str(exc)}

    candidates = [Path(p) for p in (result_files or [])]
    explicit = Path(parsed["result_file"]) if parsed["result_file"] else None
    chosen, why = select_artefact(candidates, explicit)
    if chosen is None:
        return {"verdict": "REFUSED",
                "detail": f"no single result artefact to check: {why}"}

    try:
        points, cells, fields = read_nodal_mesh(chosen)
    except Exception as exc:
        return {"verdict": "REFUSED",
                "detail": f"{chosen.name} could not be read: {exc}"}
    if not fields:
        return {"verdict": "REFUSED",
                "detail": f"{chosen.name} carries no nodal field"}

    wanted = parsed["field"]
    if wanted and wanted not in fields:
        return {"verdict": "REFUSED",
                "detail": (f"field '{wanted}' is not in {chosen.name}; openPASO "
                           f"does not substitute another")}
    name = wanted or next(iter(fields))
    values = np.asarray(fields[name], float)
    dim = parsed["dim"]

    if parsed["operator"] == "elasticity":
        from .residual_check import check_elasticity_residual
        if values.ndim != 2:
            return {"verdict": "REFUSED",
                    "detail": (f"'{name}' is not a vector field; elasticity "
                               f"needs a {dim}-component displacement")}
        ev = check_elasticity_residual(
            points, cells, values, dim=dim,
            source_fn=_vector_evaluator(parsed["source_code"], dim),
            young=parsed["young"], poisson=parsed["poisson"],
            domain_measure_expected=parsed["domain_measure"])
        return _as_result(ev, name, chosen)

    if values.ndim > 1:
        return {"verdict": "UNSUPPORTED",
                "detail": (f"'{name}' is a vector field but the declared "
                           f"operator is scalar diffusion; declare "
                           f"operator='elasticity' for a displacement field")}

    verdict = check_residual(
        points, cells, values, dim=dim,
        source_fn=_evaluator(parsed["source_code"], dim),
        coeff_fn=_coefficient_evaluator(parsed["coefficient_code"], dim),
        advection_fn=(_vector_evaluator(parsed["advection_code"], dim)
                      if parsed.get("advection_code") else None),
        reaction_fn=(_evaluator(parsed["reaction_code"], dim)
                     if parsed.get("reaction_code") is not None else None),
        domain_measure_expected=parsed["domain_measure"])

    if not verdict.supported:
        return {"verdict": "UNSUPPORTED", "detail": verdict.detail,
                "field": name, "artefact": chosen.name}
    return {
        "verdict": {"solves": "SOLVES",
                    "does_not_solve": "DOES_NOT_SOLVE",
                    "inconclusive": "INCONCLUSIVE"}.get(
                        verdict.status,
                        "SOLVES" if verdict.solver_like else "DOES_NOT_SOLVE"),
        "relative_residual": verdict.relative_residual,
        "n_interior_dofs": verdict.n_interior,
        "domain_measure": verdict.domain_measure,
        "field": name,
        "artefact": chosen.name,
        "detail": verdict.detail,
    }
