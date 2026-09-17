"""Kratos shape optimization generators and knowledge."""


from ._structural_real import real_structural_script

_OPT_DRIVER = '''\
"""Shape optimisation by finite-difference steepest descent, WITH EVERY
ANALYSIS SOLVED BY KRATOS.

The previous version of this template assembled and solved the elasticity
problem itself with numpy/scipy while calling itself "Shape optimization -
compliance minimization - Kratos (standalone)". It never imported
KratosMultiphysics, so no result from it could be attributed to Kratos.

Each design evaluation here writes a Kratos script for the current geometry and
runs it as a SUBPROCESS, so the analysis is Kratos's and its console output is
captured per evaluation. That is deliberately the expensive way round: a
finite-difference shape gradient costs one Kratos solve per perturbed
parameter per step. If that is too expensive for your problem, reduce
N_OPT_STEPS or the number of design variables -- do NOT replace the analysis
with an assembly written here, because then the answer is not Kratos's.

For a real adjoint-based shape optimisation Kratos ships
OptimizationApplication / ShapeOptimizationApplication; this template is
gradient-free-by-finite-difference on purpose, so it depends on nothing beyond
StructuralMechanicsApplication.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

N_OPT_STEPS = {n_steps}
STEP = {step}
LY0 = {ly}
FD_EPS = {fd_eps}


def analysis(height, tag):
    """One Kratos solve for a design with this height. Returns the compliance
    proxy max|u| -- replace with your own objective."""
    src = SCRIPT_TEMPLATE.replace("__LY__", repr(float(height)))
    Path(f"design_{{tag}}.py").write_text(src)
    r = subprocess.run([sys.executable, f"design_{{tag}}.py"],
                       capture_output=True, text=True, timeout=1800)
    Path(f"design_{{tag}}.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"Kratos analysis for design {{tag}} failed:\\n"
                         + r.stdout + r.stderr)
    return float(json.load(open("results_summary.json"))["max_abs_displacement"])


history = []
h = LY0
for step in range(1, N_OPT_STEPS + 1):
    j0 = analysis(h, f"{{step}}_base")
    jp = analysis(h + FD_EPS, f"{{step}}_pert")
    grad = (jp - j0) / FD_EPS
    h = max(1e-3, h - STEP * grad)
    history.append({{"step": step, "height": float(h),
                    "objective": j0, "gradient": float(grad)}})
    print(f"opt step {{step}}: J={{j0:.6e}} dJ/dh={{grad:.6e}} -> h={{h:.6f}}")

json.dump({{"history": history, "n_kratos_solves": 2 * N_OPT_STEPS,
           "analysis_solver": "Kratos StructuralMechanicsApplication",
           "gradient": "forward finite difference, one Kratos solve per "
                       "perturbation"}},
          open("optimization_summary.json", "w"), indent=2)
print("Kratos shape optimisation complete.")
'''


def _shape_optimization_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE - values are defaults, determine appropriate values for your specific problem.

    Shape optimisation whose every design evaluation is a real Kratos
    StructuralMechanics solve, run as a subprocess so it is attributable.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 10)
    script = real_structural_script(
        title="Design evaluation for the shape optimisation, Kratos",
        nx=nx, ny=ny, lx=params.get("lx", 2.0), ly=params.get("ly", 1.0),
        young=params.get("E", 1000.0), nu=params.get("nu", 0.3),
        traction=params.get("traction", (0.0, -1.0)),
        plane=params.get("plane", "strain"))
    # the design variable is the height, so make LY substitutable
    script = script.replace(f"LX, LY = {params.get('lx', 2.0)}, "
                            f"{params.get('ly', 1.0)}",
                            f"LX, LY = {params.get('lx', 2.0)}, __LY__")
    body = _OPT_DRIVER.format(n_steps=params.get("n_opt_steps", 5),
                              step=params.get("step_size", 0.01),
                              ly=params.get("ly", 1.0),
                              fd_eps=params.get("fd_eps", 1e-3))
    # repr() literal FIRST, for the same two reasons as the co-simulation
    # template: module-level code runs top to bottom, and the child script
    # carries its own docstring whose triple quote would end an embedded block.
    return "SCRIPT_TEMPLATE = " + repr(script) + "\n\n" + body


KNOWLEDGE = {
    "shape_optimization": {
        "description": "Shape optimization via Kratos ShapeOptimizationApplication",
        "application": "ShapeOptimizationApplication (pip install KratosShapeOptimizationApplication)",
        "algorithms": [
            "steepest_descent", "penalized_projection", "trust_region",
            "gradient_projection", "bead_optimization",
        ],
        "objective_types": ["compliance (strain energy)", "stress (max von Mises)",
                           "mass/volume", "eigenfrequency", "custom (user-defined)"],
        "filtering": {
            "vertex_morphing": "Henze-Hinterberger filter for mesh-independent shape update",
            "radius": "Filter radius controls smoothness of shape changes",
            "damping": "Near-boundary damping to prevent mesh distortion",
        },
        "pitfalls": [
            "[API] ShapeOptimizationApplication and OptimizationApplication are two SEPARATE applications with disjoint Python surfaces, and the shape-optimisation vocabulary lives entirely in the former. The mappers (MapperVertexMorphing and the sliding / in-plane variants), the utility classes (GeometryUtilities, DampingUtilities, MeshControllerUtilities, OptimizationUtilities) and the design variables (SHAPE_UPDATE, SHAPE_CHANGE, CONTROL_POINT_UPDATE, NORMALIZED_SURFACE_NORMAL, DF1DX) are attributes of ShapeOptimizationApplication only. Mappers are built through its own ShapeOptimizationApplication.mapper_factory.CreateMapper, not through a core factory. Signal: dotting any of these names off KratosMultiphysics or off OptimizationApplication raises AttributeError at attribute access, before a mapper is constructed or a gradient is asked for; the same name resolves on ShapeOptimizationApplication.",
        ],
        "guidance": [
            "[Numerical] Shape gradients require adjoint solve or finite differences",
            "[Numerical] Mesh quality degrades with large shape changes: use mesh smoothing",
            "[Numerical] Filter radius should be > 2-3x element edge length",
            "[Numerical] Constrained optimization: use penalized projection or augmented Lagrangian",
            "[Numerical] For manufacturing constraints: use bead optimization or geometric filtering",
        ]
    },
}

GENERATORS = {
    "shape_optimization_2d": _shape_optimization_2d_kratos,
}
