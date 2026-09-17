"""Kratos Reduced Order Modeling (ROM) generators and knowledge.

Covers POD, HROM, neural network surrogates.
Application: RomApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _rom_2d generator emitted
# an availability-probe stub that only import-checked RomApplication and
# wrote {"note": "not installed"} with no solver run. RomApplication is NOT
# importable in the installed Kratos stack (the working python3 stack ships
# only StructuralMechanics, ConvectionDiffusion, ContactStructuralMechanics
# and LinearSolvers). The stub generator and its 'rom_2d' registry entry
# have been removed; 'rom' is no longer advertised in
# KratosBackend.supported_physics(). The KNOWLEDGE block below is retained
# as a reference-only entry (knowledge(kratos, rom) still works) — it makes
# no claim that the app is runnable here.


KNOWLEDGE = {
    "rom": {
        "description": "Reduced Order Modeling: POD, HROM, neural network surrogates",
        "application": "RomApplication (pip install KratosRomApplication)",
        "methods": {
            "POD": "Proper Orthogonal Decomposition — project FOM onto reduced basis",
            "HROM": "Hyper-Reduced Order Model — also reduce integration cost",
            "LSPG": "Least-Squares Petrov-Galerkin projection",
            "ANN": "Artificial Neural Network surrogate from training snapshots",
        },
        "workflow": ["1. Run full-order model (FOM) for training parameters",
                     "2. Collect snapshots", "3. Build reduced basis (SVD/POD)",
                     "4. Train ROM/HROM", "5. Evaluate at new parameters in real-time"],
        "pitfalls": [
            "[Numerical] Works best with StructuralMechanics and FluidDynamics applications Signal: RomApplication, StructuralMechanicsApplication and FluidDynamicsApplication must all load into one interpreter; a missing one raises ModuleNotFoundError at import, before any snapshot is taken.",
        ],
        "guidance": [
            "[Numerical] Training snapshots must cover the parameter space adequately",
            "[Numerical] HROM requires empirical cubature method (ECM) for element selection",
            "[Numerical] ROM accuracy degrades for strongly nonlinear problems",
        ]
    },
}

# Empty: no runnable ROM generator — RomApplication is not installable in
# this Kratos stack (see honesty-audit note above).
GENERATORS = {}
