"""Kratos compressible flow generators and knowledge.

Covers compressible potential flow and compressible Euler/Navier-Stokes.
Applications: CompressiblePotentialFlowApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _compressible_potential_2d
# generator was an availability-probe stub (import-check +
# {"note": "not installed"}, no solver run). CompressiblePotentialFlow-
# Application is NOT importable in the installed Kratos stack, so
# 'compressible_potential' has been removed from the generator registry and
# from KratosBackend.supported_physics(). KNOWLEDGE retained for reference.


KNOWLEDGE = {
    "compressible_potential": {
        "description": "Compressible potential flow (subsonic/transonic) around aerodynamic bodies",
        "application": "CompressiblePotentialFlowApplication",
        "elements": {
            "2D": ["IncompressiblePotentialFlowElement2D3N", "CompressiblePotentialFlowElement2D3N",
                   "TransonicPerturbationPotentialFlowElement2D3N"],
            "3D": ["IncompressiblePotentialFlowElement3D4N", "CompressiblePotentialFlowElement3D4N"],
        },
        "solver_types": ["potential_flow_solver (linear/nonlinear)"],
        "pitfalls": [
            "[Numerical] Far-field BC: use PotentialWallCondition for solid walls Signal: CreateNewCondition with the bare name raises RuntimeError 'The Condition \"PotentialWallCondition\" is not registered!'; the registered spellings carry a dimension-and-node-count suffix (PotentialWallCondition2D2N / 3D3N).",
            "[Physics] Freestream conditions are set through FREE_STREAM_VELOCITY, FREE_STREAM_MACH and FREE_STREAM_DENSITY, which are attributes of CompressiblePotentialFlowApplication. There is no FREESTREAM_VELOCITY (one word) and no MACH_INFINITY ('infinity' spelled out): neither name exists anywhere in Kratos, including its 48-application C++ source, and an earlier version of this entry prescribed both. Signal: dotting either non-existent name off the application or off core raises AttributeError at attribute access, before any value is assigned; the FREE_STREAM_* spellings resolve.",
        ],
        "guidance": [
            "[Numerical] Transonic: requires shock-capturing stabilization",
            "[Physics] Lift/drag computed from pressure integration on body surface",
        ]
    },
}

# Empty: CompressiblePotentialFlowApplication not installable in this Kratos
# stack; the prior generator was a no-solve probe stub (removed).
GENERATORS = {}
