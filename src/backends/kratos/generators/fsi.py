"""Kratos FSI generators and knowledge."""


# NOTE (2026-06-26 honesty audit): the previous _fsi_2d_kratos generator was
# an availability-probe stub (it only import-checked CoSimulation/FSI apps and
# wrote {"note": ...} with no solver run). FSIApplication and
# CoSimulationApplication are NOT importable in the installed Kratos stack, so
# 'fsi' has been removed from the generator registry and from
# KratosBackend.supported_physics(). The KNOWLEDGE block below is retained as a
# reference-only entry (accelerator/mapper names, pitfalls).


KNOWLEDGE = {
    "fsi": {
        "description": "Fluid-Structure Interaction (FSI/CoSimulation)",
        "applications": ["FSIApplication (partitioned DN)", "CoSimulationApplication (general coupling)"],
        "cosimulation": {
            "coupling_schemes": ["Gauss-Seidel (weak/strong)", "Jacobi (weak/strong)"],
            # See cosimulation physics catalog for the cross-checked
            # set of CoSim file-stem accelerator names.
            "convergence_accelerators_via_cosim": [
                "constant_relaxation", "aitken", "mvqn",
                "block_mvqn", "block_ibqnls", "iqnils", "anderson",
            ],
            "data_transfer_via_cosim": [
                "nearest_neighbor", "nearest_element", "barycentric",
                "coupling_geometry", "radial_basis_function",
            ],
            "solver_wrappers": ["kratos (internal)", "external (CoSimIO for coupling with other codes)"],
        },
        # FSIApplication ships its OWN convergence-accelerator
        # classes (Python attributes on the FSI module, distinct
        # from the CoSim file stems above). Verified by walking
        # dir(KratosMultiphysics.FSIApplication) 2026-06-01.
        "fsi_application_accelerators": [
            "AitkenConvergenceAccelerator",
            "ConstantRelaxationConvergenceAccelerator",
            "IBQNMVQNConvergenceAccelerator",
            "IBQNMVQNRandomizedSVDConvergenceAccelerator",
            "MVQNFullJacobianConvergenceAccelerator",
            "MVQNRandomizedSVDConvergenceAccelerator",
            "MVQNRecursiveJacobianConvergenceAccelerator",
        ],
        "fsi_application_partitioned_utils": [
            "FSIUtils", "SharedPointsMapper",
            "PartitionedFSIUtilitiesArray2D",
            "PartitionedFSIUtilitiesArray3D",
            "PartitionedFSIUtilitiesDouble2D",
            "PartitionedFSIUtilitiesDouble3D",
        ],
        "pitfalls": [
            "[Integration] KratosFSIApplication is a SEPARATE pip package (pip install KratosFSIApplication) \u2014 it is NOT pulled in by KratosMultiphysics core. The FSI catalog previously inherited the cosim accelerator list verbatim, conflating TWO distinct surfaces: (a) CoSimulationApplication accelerator FILE stems (block_ibqnls, iqnils, mvqn, block_mvqn, aitken, anderson) under convergence_accelerators/, vs (b) FSIApplication CLASS names (IBQNMVQNConvergenceAccelerator, MVQNFullJacobianConvergenceAccelerator, AitkenConvergenceAccelerator, ...). \"ibqn\" (bare) is neither \u2014 neither a CoSim file nor an FSI class. Signal: \"import KratosMultiphysics.FSIApplication\" on a fresh .venv raises ModuleNotFoundError; using \"ibqn\" as a CoSim accelerator \"type\" raises a factory ImportError finding \"ibqn.py\". (Verified 2026-06-01 \u2014 see also cosimulation physics pitfall #0 + Tier-2 fixture cosimulation_accelerator_mapper_names.)",
            "[Numerical] FSI needs: FluidDynamics + StructuralMechanics + MeshMoving + Mapping applications Signal: two of the four \u2014 MeshMovingApplication and MappingApplication \u2014 never appear in a user-facing FSI JSON, so their absence surfaces as a ModuleNotFoundError raised inside partitioned_fsi_base_solver at run time rather than as a validation error at plan time.",
        ],
        "guidance": [
            "[Numerical] CoSimIO enables coupling Kratos with any external solver (including our MCP agents)",
            "[Numerical] Aitken relaxation recommended for initial runs; MVQN for production",
        ]
    },
}

# Empty: FSIApplication / CoSimulationApplication not installable in this
# Kratos stack; the prior generator was a no-solve probe stub (removed).
GENERATORS = {}
