"""Kratos topology optimization generators and knowledge.

Application: TopologyOptimizationApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _topology_opt_2d generator
# emitted an availability-probe stub that only import-checked
# TopologyOptimizationApplication and wrote {"note": "not installed"} with
# no solver run. TopologyOptimizationApplication is NOT importable in the
# installed Kratos stack. The stub generator and its
# 'topology_optimization_2d' registry entry have been removed;
# 'topology_optimization' is no longer advertised in
# KratosBackend.supported_physics(). The KNOWLEDGE block below is retained
# as a reference-only entry.


KNOWLEDGE = {
    "topology_optimization": {
        "description": "Topology optimization: SIMP, level-set, compliance/stress objectives",
        "application": "TopologyOptimizationApplication",
        "methods": {
            "SIMP": "Solid Isotropic Material with Penalization (density-based)",
            "level_set": "Level-set topology optimization",
        },
        "objectives": ["compliance_minimization", "stress_minimization",
                       "multi_objective", "frequency_maximization"],
        "constraints": ["volume_fraction", "stress_limit", "displacement_limit"],
        "pitfalls": [
            "[Integration] Requires StructuralMechanicsApplication as dependency Signal: StructuralMechanicsApplication imports while TopologyOptimizationApplication itself is not published as a wheel at any version \u2014 the prerequisite is satisfiable and the application it is for is not reachable on a pip stack. OptimizationApplication is the maintained replacement and does import.",
        ],
        "guidance": [
            "[Numerical] SIMP penalization factor p=3 is standard",
            "[Numerical] Filter radius needed to avoid checkerboard patterns",
            "[Numerical] Mesh-dependent results without proper filtering",
        ]
    },
}

# Empty: no runnable topology-optimization generator —
# TopologyOptimizationApplication is not installable in this Kratos stack
# (see honesty-audit note above).
GENERATORS = {}
