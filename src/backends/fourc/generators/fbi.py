"""Fluid-Beam Interaction (FBI) generator for 4C.

Covers coupling of a 3-D fluid field (incompressible Navier-Stokes) with
embedded 1-D beam elements.  The beams are immersed in the fluid domain
and interact via penalty or mortar coupling.  Applications include flow
around fibers, stent deployment in blood vessels, and fiber-reinforced
composites with fluid flow.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FBIGenerator(BaseGenerator):
    """Generator for Fluid-Beam Interaction problems in 4C."""

    module_key = "fbi"
    display_name = "Fluid-Beam Interaction (FBI)"
    problem_type = "Fluid_Beam_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Fluid-Beam Interaction (FBI) couples a 3-D incompressible "
                "Navier-Stokes fluid field with embedded 1-D beam elements.  "
                "The beams are immersed in the fluid volume and do not "
                "require a body-fitted mesh.  Coupling is achieved via "
                "penalty or mortar methods that transfer fluid drag forces "
                "to the beams and impose the beam velocity as a constraint "
                "on the fluid.  The PROBLEM TYPE is "
                "'Fluid_Beam_Interaction'.  There is no 'FBI DYNAMIC' "
                "section: the partitioned time loop is driven by FSI "
                "DYNAMIC (TIMESTEP, NUMSTEP, MAXTIME, RESULTSEVERY) with "
                "FSI DYNAMIC/PARTITIONED SOLVER for the coupling "
                "iteration, while the coupling itself is configured in "
                "FLUID BEAM INTERACTION (COUPLING, PRESORT_STRATEGY, "
                "STARTSTEP) and FLUID BEAM INTERACTION/BEAM TO FLUID "
                "MESHTYING (MESHTYING_DISCRETIZATION, "
                "CONSTRAINT_STRATEGY, PENALTY_PARAMETER, SEARCH_RADIUS, "
                "...).  The remaining dynamics sections are STRUCTURAL "
                "DYNAMIC (for the beams) and FLUID DYNAMIC.  The fluid "
                "mesh uses standard FLUID elements while beams use BEAM3R "
                "or BEAM3EB elements.  No ALE mesh motion is needed since "
                "the coupling is immersed (non-body-fitted)."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "FSI DYNAMIC",
                "FLUID BEAM INTERACTION",
                "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
            ],
            "optional_sections": [
                "FSI DYNAMIC/PARTITIONED SOLVER",
                "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/RUNTIME VTK OUTPUT",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "STRUCTURAL DYNAMIC/GENALPHA",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/BEAMS",
                "IO/RUNTIME VTK OUTPUT/FLUID",
                "BINNING STRATEGY",
            ],
            "materials": {
                "MAT_BeamReissnerElastHyper": {
                    "description": (
                        "Geometrically exact Reissner beam material.  "
                        "Defines axial, shear, bending, and torsional "
                        "stiffness for 1-D beam elements embedded in "
                        "the fluid."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus of beam material",
                            "range": "> 0",
                        },
                        "POISSONRATIO": {
                            "description": "Poisson's ratio of beam material",
                            "range": "(0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density of beam material",
                            "range": "> 0",
                        },
                        "CROSSAREA": {
                            "description": "Cross-sectional area of beam",
                            "range": "> 0",
                        },
                        "SHEARCORR": {
                            "description": "Shear correction factor",
                            "range": "> 0 (typically 1.0)",
                        },
                        "MOMINPOL": {
                            "description": "Polar moment of inertia",
                            "range": "> 0",
                        },
                        "MOMIN2": {
                            "description": "Second moment of area (axis 2)",
                            "range": "> 0",
                        },
                        "MOMIN3": {
                            "description": "Second moment of area (axis 3)",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid material for the background "
                        "fluid field."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity [Pa s]",
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Fluid density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "structure_solver": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "Beam problems are typically small and well-suited "
                        "to direct solvers."
                    ),
                },
                "fluid_solver": {
                    "type": "Belos or UMFPACK",
                    "notes": (
                        "Fluid solver for the background Navier-Stokes "
                        "equations.  Iterative for large 3-D problems."
                    ),
                },
            },
            "coupling_parameters": {
                "FLUID BEAM INTERACTION/COUPLING": (
                    "Direction of the FBI coupling: 'fluid' (fluid is "
                    "driven by the beam), 'solid' (beam is driven by the "
                    "fluid) or 'two-way' (default, fully coupled)."
                ),
                "FLUID BEAM INTERACTION/PRESORT_STRATEGY": (
                    "'bruteforce' (default) or 'binning'.  This is the "
                    "switch that turns on binning for the beam-fluid "
                    "pair search; a BINNING STRATEGY section alone is a "
                    "silent no-op."
                ),
                "FLUID BEAM INTERACTION/STARTSTEP": (
                    "Time step at which the fluid-beam coupling starts "
                    "(default 0, i.e. from the first step)."
                ),
                "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/MESHTYING_DISCRETIZATION": (
                    "'none' (default, i.e. inactive), "
                    "'gauss_point_to_segment' or 'mortar'.  This is what "
                    "used to be miscalled a COUPLING_TYPE -- there is no "
                    "COUPLING_TYPE key anywhere in the FBI grammar."
                ),
                "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/CONSTRAINT_STRATEGY": (
                    "'none' (default) or 'penalty'.  Penalty is the only "
                    "enforcement implemented for beam-to-fluid meshtying."
                ),
                "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/PENALTY_PARAMETER": (
                    "Penalty stiffness for the immersed coupling.  "
                    "Controls how strongly the beam velocity constraint "
                    "is enforced on the fluid.  Too small leads to "
                    "fluid penetrating the beam; too large causes "
                    "ill-conditioning."
                ),
                "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/SEARCH_RADIUS": (
                    "Absolute search radius for beam-to-fluid pairs "
                    "(default 1000).  This is the only SEARCH_RADIUS in "
                    "4C's whole grammar."
                ),
                "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/MORTAR_SHAPE_FUNCTION": (
                    "'none' (default), 'line2', 'line3' or 'line4'.  "
                    "Required when MESHTYING_DISCRETIZATION: mortar."
                ),
                "FSI DYNAMIC": (
                    "Drives the FBI time loop: TIMESTEP, NUMSTEP, "
                    "MAXTIME, RESULTSEVERY, RESTARTEVERY.  FBI is a "
                    "partitioned scheme, so FSI DYNAMIC/PARTITIONED "
                    "SOLVER holds CONVTOL, COUPVARIABLE and ITEMAX."
                ),
            },
            "pitfalls": [
                (
                    "[Input] FBI does NOT use ALE mesh motion. "
                    "The fluid mesh is FIXED (Eulerian); the "
                    "beams move through the fluid via "
                    "immersed coupling. Signal: none — and "
                    "that is the hazard. Adding an ALE "
                    "DYNAMIC section to an FBI input does not "
                    "abort: 4C swallows it silently, builds "
                    "no ALE discretisation, and returns "
                    "exactly the same answer, so an author "
                    "who added it believing it enables mesh "
                    "motion gets a clean converged run in "
                    "which the fluid mesh never moved. There "
                    "is no 4C_fbi_factory.cpp and no 'FBI is "
                    "incompatible with ALE' message. "
                    "(Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] The FBI penalty parameter "
                    "must be tuned, and it lives in FLUID "
                    "BEAM INTERACTION/BEAM TO FLUID "
                    "MESHTYING/PENALTY_PARAMETER — NOT in a "
                    "'BEAM INTERACTION/BEAM TO FLUID "
                    "MESHTYING' section, which 4C rejects "
                    "outright with \"Section '...' is not a "
                    "valid section name.\" from "
                    "core/io/src/4C_io_input_file.cpp. "
                    "Signal: too LARGE aborts with 'The "
                    "nonlinear solver did not converge!' from "
                    "solver_nonlin_nox/"
                    "4C_solver_nonlin_nox_problem.cpp — a NOX "
                    "message, with nothing about cond(K). Too "
                    "SMALL is the dangerous one: it converges "
                    "silently and simply under-couples the "
                    "beam, with no warning and no 'slip "
                    "through' anywhere in the log. Compare "
                    "against a tuned reference. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Mesh] Beam elements (BEAM3R, BEAM3EB) "
                    "are 1D line elements. The fluid mesh "
                    "must cover the entire region occupied by "
                    "beams. Signal: none — a beam lying "
                    "outside the fluid mesh raises no error. "
                    "Both fields are built, no search "
                    "diagnostic is printed, NOX reports the "
                    "solution 'is already converged' because "
                    "the beam carries no fluid load, and the "
                    "beam displacement comes back exactly "
                    "zero. A partially overlapping beam is "
                    "worse: it moves, wrongly, and still says "
                    "nothing. The quoted 'beam "
                    "outside fluid domain' does not exist in "
                    "4C, and neither does the file it was "
                    "attributed to, 4C_fbi_partitioner.cpp. "
                    "(Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Binning for the beam-fluid "
                    "pair search is switched on by FLUID BEAM "
                    "INTERACTION/PRESORT_STRATEGY (choices "
                    "bruteforce and binning; the DEFAULT is "
                    "bruteforce). A BINNING STRATEGY section "
                    "with BIN_SIZE_LOWER_BOUND only "
                    "parameterises the binning coupler once "
                    "that switch is thrown. Signal: none — "
                    "adding BINNING STRATEGY on its own is a "
                    "silent no-op, accepted without any "
                    "'ignored' notice and giving an identical "
                    "run. Confirm which coupler ran from 4C's "
                    "TimeMonitor table at the end of the log: "
                    "FBI::FBIBinningCoupler::Search appears "
                    "only under PRESORT_STRATEGY: binning. "
                    "(Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Output] Structural output uses the "
                    "BEAM discretisation, NOT the standard "
                    "STRUCTURE output. Use IO/RUNTIME VTK "
                    "OUTPUT/BEAMS for beam visualisation. "
                    "Signal: none — configuring IO/RUNTIME "
                    "VTK OUTPUT/STRUCTURE instead does not "
                    "produce empty output, it produces NO "
                    "structure output file at all: the "
                    "structure-beams .vtu files and their "
                    ".pvd collection simply never appear. The "
                    "run still exits 0 and passes its result "
                    "tests, and the beam-to-fluid coupling "
                    ".vtu files are still written, which is "
                    "what makes it look as though output "
                    "worked. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] FBI beam material uses "
                    "DEDICATED beam material types (e.g. "
                    "MAT_BeamReissnerElastHyper) — NOT "
                    "standard solid materials. Cross-"
                    "sectional properties (CROSSAREA, "
                    "MOMINPOL, etc.) must be specified. "
                    "Signal: a solid material on a BEAM3R "
                    "aborts at element-read time in "
                    "beam3/src/4C_beam3_reissner_input.cpp "
                    "with \"The material parameter definition "
                    "'m_elasthyper' is not supported by "
                    "Beam3r element! Choose "
                    "MAT_BeamReissnerElastHyper, "
                    "MAT_BeamReissnerElastHyper_ByModes or "
                    "MAT_BeamReissnerElastPlastic!\" — it "
                    "names the offending material and lists "
                    "the legal ones. The quoted 'beam "
                    "element requires beam material' does not "
                    "exist in 4C, and neither does the file it "
                    "was attributed to, 4C_mat_beam_base.cpp. "
                    "(Corrected by execution 2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "fiber_in_channel_3d",
                    "description": (
                        "A flexible fiber immersed in a 3-D channel flow.  "
                        "The fluid drag causes the fiber to deform and the "
                        "fiber in turn disturbs the flow field.  Tests "
                        "penalty coupling, beam large deformation, and "
                        "fluid stabilization."
                    ),
                    "template_variant": "penalty_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "penalty_3d",
                "description": (
                    "3-D FBI with penalty coupling: flexible beam in "
                    "channel flow.  BEAM3R elements immersed in FLUID "
                    "HEX8 elements, penalty-based velocity coupling, "
                    "UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "penalty_3d") -> str:
        templates = {
            "penalty_3d": self._template_penalty_3d,
        }
        if variant == "default":
            variant = "penalty_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_penalty_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Fluid-Beam Interaction (FBI) with Penalty Coupling
            #
            # A flexible beam is immersed in a 3-D fluid channel.  The
            # fluid exerts drag on the beam, causing it to deform, and
            # the beam displaces the fluid via the coupling penalty term.
            #
            # Mesh: requires TWO exodus files:
            #   Fluid mesh: "fluid.e" with
            #     element_block 1 = fluid domain (HEX8)
            #     node_set 1 = inlet
            #     node_set 2 = outlet
            #     node_set 3 = walls (no-slip)
            #   Beam mesh: "beams.e" with
            #     element_block 1 = beam elements (LINE2)
            #     node_set 1 = beam clamped end
            # ---------------------------------------------------------------
            TITLE:
              - "3-D fluid-beam interaction -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Beam_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              USE_ABSOLUTE_POSITIONS: true
              TRIAD_VISUALIZATIONPOINT: true
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true

            # == FBI time loop =================================================
            # FBI is a partitioned scheme driven by FSI DYNAMIC.  There is no
            # "FBI DYNAMIC" section -- 4C rejects it with
            # "Section 'FBI DYNAMIC' is not a valid section name."
            FSI DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              RESULTSEVERY: <results_output_interval>
            FSI DYNAMIC/PARTITIONED SOLVER:
              CONVTOL: <coupling_convergence_tolerance>
              COUPVARIABLE: "Force"
              ITEMAX: <max_coupling_iterations>

            # == Structure (beams) =============================================
            # BEAM3R carries large rotations, so use GenAlphaLieGroup with
            # MASSLIN: rotations.  Classical "GenAlpha" segfaults in
            # Beam3r::calc_internal_and_inertia_forces_and_stiff, and
            # GenAlpha + MASSLIN: rotations aborts with "MASSLIN=ml_rotations
            # is not supported by classical GenAlpha!".
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "GenAlphaLieGroup"
              MASSLIN: "rotations"
              LOADLIN: true
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>
            STRUCTURAL DYNAMIC/GENALPHA:
              RHO_INF: <genalpha_rho_inf>

            # == Fluid =========================================================
            FLUID DYNAMIC:
              TIMEINTEGR: "Np_Gen_Alpha"
              TIMESTEP: <fluid_timestep>
              NUMSTEP: <fluid_num_steps>
              LINEAR_SOLVER: 2
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == FBI coupling ==================================================
            FLUID BEAM INTERACTION:
              COUPLING: "<coupling_direction>"          # fluid | solid | two-way
              PRESORT_STRATEGY: "<presort_strategy>"    # bruteforce | binning
            FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING:
              MESHTYING_DISCRETIZATION: "<meshtying_discretization>"
              CONSTRAINT_STRATEGY: "penalty"
              PENALTY_PARAMETER: <penalty_parameter>
              SEARCH_RADIUS: <search_radius>
            FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/RUNTIME VTK OUTPUT:
              WRITE_OUTPUT: true
              NODAL_FORCES: true
              CONSTRAINT_VIOLATION: true

            # == Binning (only used when PRESORT_STRATEGY: binning) ============
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "beam_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "fluid_solver"

            # == Materials =====================================================
            MATERIALS:
              # Beam material (Reissner elastohyper)
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam_Young_modulus>
                  POISSONRATIO: <beam_Poisson_ratio>
                  DENS: <beam_density>
                  CROSSAREA: <beam_cross_section_area>
                  SHEARCORR: <beam_shear_correction_factor>
                  MOMINPOL: <beam_polar_moment_of_inertia>
                  MOMIN2: <beam_moment_of_inertia_2>
                  MOMIN3: <beam_moment_of_inertia_3>
              # Fluid material
              - MAT: 2
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>

            # == Boundary Conditions ===========================================

            # Beam: clamped end
            DESIGN POINT DIRICH CONDITIONS:
              - E: <beam_clamped_node_set_id>
                NUMDOF: 6
                ONOFF: [1, 1, 1, 1, 1, 1]
                VAL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0, 0, 0]

            # Fluid: inlet
            DESIGN SURF DIRICH CONDITIONS:
              - E: <inlet_face_id>
                NUMDOF: 4
                ONOFF: [1, 1, 1, 0]
                VAL: [<inlet_velocity_x>, <inlet_velocity_y>, <inlet_velocity_z>, 0.0]
                FUNCT: [<inlet_ramp_function>, 0, 0, 0]
              # Fluid: no-slip walls
              - E: <wall_face_id>
                NUMDOF: 4
                ONOFF: [1, 1, 1, 0]
                VAL: [0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0]

            # Inlet ramp-up function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry ======================================================
            # Beam-to-fluid meshtying needs a Hermite-centerline beam:
            # BEAM3R/LINE3 with HERMITE_CENTERLINE true, or BEAM3EB.  A plain
            # BEAM3R/LINE2 aborts with "Beam3tosolidmeshtying: beam::n_val_=2
            # detected for beam3r element w/o Hermite centerline".  BEAM3R
            # also needs nodal triads: either TRIADS (9 doubles for LINE3) or
            # NODAL_ROTATION_VECTORS naming a cell-data field in the mesh.
            STRUCTURE GEOMETRY:
              FILE: "<beam_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  BEAM3R:
                    LINE3:
                      MAT: 1
                      HERMITE_CENTERLINE: true
                      NODAL_ROTATION_VECTORS: "<triad_cell_field_name>"

            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    HEX8:
                      MAT: 2
                      NA: Euler

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_beam_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_beam_displacement>
                  TOLERANCE: <result_tolerance>
              - FLUID:
                  DIS: "fluid"
                  NODE: <result_fluid_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_fluid_velocity>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check beam Young's modulus
        young = params.get("YOUNG") or params.get("beam_YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"Beam YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"Beam YOUNG must be a positive number, got {young!r}."
                )

        # Check penalty parameter
        penalty = params.get("PENALTY_PARAMETER")
        if penalty is not None:
            try:
                p = float(penalty)
                if p <= 0:
                    issues.append(
                        f"PENALTY_PARAMETER must be > 0, got {p}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"PENALTY_PARAMETER must be a positive number, "
                    f"got {penalty!r}."
                )

        # Check fluid viscosity
        viscosity = params.get("DYNVISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"Fluid DYNVISCOSITY must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DYNVISCOSITY must be a positive number, "
                    f"got {viscosity!r}."
                )

        # Check fluid density
        density = params.get("DENSITY")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(
                        f"Fluid DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check cross-sectional area
        crossarea = params.get("CROSSAREA")
        if crossarea is not None:
            try:
                a = float(crossarea)
                if a <= 0:
                    issues.append(
                        f"CROSSAREA must be > 0, got {a}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CROSSAREA must be a positive number, "
                    f"got {crossarea!r}."
                )

        # COUPLING_TYPE does not exist in the FBI grammar
        if "COUPLING_TYPE" in params:
            issues.append(
                "COUPLING_TYPE is not an FBI parameter.  Use FLUID BEAM "
                "INTERACTION/COUPLING (fluid|solid|two-way) for the "
                "coupling direction and FLUID BEAM INTERACTION/BEAM TO "
                "FLUID MESHTYING/MESHTYING_DISCRETIZATION "
                "(gauss_point_to_segment|mortar) for the discretisation."
            )

        # Check coupling direction
        coupling = params.get("COUPLING")
        if coupling is not None and coupling not in (
            "fluid", "solid", "two-way",
        ):
            issues.append(
                f"COUPLING must be 'fluid', 'solid' or 'two-way', "
                f"got {coupling!r}."
            )

        # Check meshtying discretisation
        discretization = params.get("MESHTYING_DISCRETIZATION")
        if discretization is not None and discretization not in (
            "none", "gauss_point_to_segment", "mortar",
        ):
            issues.append(
                "MESHTYING_DISCRETIZATION must be 'none', "
                "'gauss_point_to_segment' or 'mortar', "
                f"got {discretization!r}."
            )

        # Check constraint enforcement
        strategy = params.get("CONSTRAINT_STRATEGY")
        if strategy is not None and strategy not in ("none", "penalty"):
            issues.append(
                "CONSTRAINT_STRATEGY must be 'none' or 'penalty' for "
                f"beam-to-fluid meshtying, got {strategy!r}."
            )

        # Check presort strategy
        presort = params.get("PRESORT_STRATEGY")
        if presort is not None and presort not in ("bruteforce", "binning"):
            issues.append(
                f"PRESORT_STRATEGY must be 'bruteforce' or 'binning', "
                f"got {presort!r}."
            )

        # Check search radius
        search_r = params.get("SEARCH_RADIUS")
        if search_r is not None:
            try:
                r = float(search_r)
                if r <= 0:
                    issues.append(f"SEARCH_RADIUS must be > 0, got {r}.")
            except (TypeError, ValueError):
                issues.append(
                    f"SEARCH_RADIUS must be a positive number, "
                    f"got {search_r!r}."
                )

        return issues
