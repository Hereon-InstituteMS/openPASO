"""FS3I (Fluid-Structure-Scalar-Scalar Interaction) generator for 4C.

Covers the 5-field coupling of fluid, structure, and two scalar transport
fields (one in the fluid domain, one in the structure domain).  The
fluid-structure interaction is handled by standard FSI (ALE-based), and
each domain carries an additional scalar transport field.  Applications
include drug delivery through arterial walls, mass transfer across
deformable membranes, and nutrient transport in biological tissues.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FS3IGenerator(BaseGenerator):
    """Generator for FS3I (5-field coupling) problems in 4C."""

    module_key = "fs3i"
    display_name = "FS3I (Fluid-Porous-Structure-Scalar-Scalar Interaction)"
    problem_type = "Fluid_Porous_Structure_Scalar_Scalar_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "FS3I is a 5-field coupling framework combining "
                "Fluid-Structure Interaction (FSI) with two scalar "
                "transport fields: one in the fluid domain and one in "
                "the structural (porous) domain.  The fluid field solves "
                "Navier-Stokes, the structure field solves momentum "
                "balance, and each domain has its own advection-diffusion "
                "scalar transport field.  At the FSI interface, both "
                "velocity continuity and scalar concentration continuity "
                "(or flux balance) are enforced.  The PROBLEM TYPE is "
                "'Fluid_Porous_Structure_Scalar_Scalar_Interaction'.  "
                "Required dynamics sections include FSI DYNAMIC, "
                "SCALAR TRANSPORT DYNAMIC (for the fluid-side scalar) "
                "and FS3I DYNAMIC for overall coupling parameters.  "
                "There is NO second scalar-transport section: 4C has "
                "no 'SCALAR TRANSPORT DYNAMIC 2'.  The structure-side "
                "scalar reuses SCALAR TRANSPORT DYNAMIC and is "
                "configured through the STRUCTSCAL_* keys of FS3I "
                "DYNAMIC (STRUCTSCAL_CONVFORM, STRUCTSCAL_INITIALFIELD, "
                "STRUCTSCAL_INITFUNCNO, STRUCTSCAL_FIELDCOUPLING) plus "
                "the FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION "
                "subsection.  Typical applications include drug elution "
                "from stents, oxygen transport through vessel walls, and "
                "mass transfer in filtration membranes."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "ALE DYNAMIC",
                "FSI DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC",
                "FS3I DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "SOLVER 3",
                "MATERIALS",
                "CLONING MATERIAL MAP",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
            ],
            "optional_sections": [
                "FS3I DYNAMIC/PARTITIONED",
                "FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION",
                "FSI DYNAMIC/MONOLITHIC SOLVER",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid for the free fluid domain."
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
                "MAT_scatra": {
                    "description": (
                        "Scalar transport material.  Used for BOTH the "
                        "fluid-side and the structure-side concentration "
                        "field -- the two differ only in the value of "
                        "DIFFUSIVITY, not in the material name."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Molecular diffusion coefficient [m^2/s].  "
                                "In the structure/porous domain this is "
                                "the effective diffusivity."
                            ),
                            "range": "> 0",
                        },
                        "REACOEFF": {
                            "description": (
                                "First-order reaction rate coefficient "
                                "[1/s].  Note the spelling: MAT_scatra "
                                "uses REACOEFF (one C), MAT_scatra_"
                                "reaction uses REACCOEFF (two C)."
                            ),
                            "range": ">= 0",
                        },
                    },
                },
                "MAT_scatra_reaction": {
                    "description": (
                        "Reaction term for drug metabolism or nutrient "
                        "consumption.  It is NOT a stand-alone scalar "
                        "material and carries no DIFFUSIVITY: it is "
                        "referenced through the REACIDS list of a "
                        "MAT_matlist_reactions entry, which also lists "
                        "the MAT_scatra species in MATIDS."
                    ),
                    "parameters": {
                        "NUMSCAL": {
                            "description": "Number of participating scalars",
                            "range": ">= 1",
                        },
                        "STOICH": {
                            "description": (
                                "Stoichiometric coefficients, one per scalar"
                            ),
                            "range": "list of int",
                        },
                        "REACCOEFF": {
                            "description": "Reaction rate coefficient [1/s]",
                            "range": ">= 0",
                        },
                        "COUPLING": {
                            "description": (
                                "Reaction coupling model, e.g. "
                                "'simple_multiplicative' or "
                                "'michaelis_menten'"
                            ),
                            "range": "string",
                        },
                        "ROLE": {
                            "description": (
                                "Role of each scalar in the reaction"
                            ),
                            "range": "list",
                        },
                    },
                },
                "MAT_ElastHyper": {
                    "description": (
                        "Hyperelastic structural material (same as FSI)."
                    ),
                    "parameters": {
                        "NUMMAT": {
                            "description": "Number of sub-materials",
                            "range": "1",
                        },
                        "MATIDS": {
                            "description": "Sub-material IDs",
                            "range": "",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                        "POLYCONVEX": {
                            "description": "Polyconvexity check flag (wrapper)",
                            "range": "0 | 1",
                        },
                    },
                },
            },
            "solver": {
                "fsi_solver": {
                    "type": "UMFPACK or block solver",
                    "notes": (
                        "Solver for the FSI sub-problem (fluid + structure "
                        "+ ALE)."
                    ),
                },
                "scatra_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "Solver for the scalar transport fields.  Scalar "
                        "transport systems are typically well-conditioned."
                    ),
                },
            },
            "coupling_parameters": {
                "COUPALGO (in section 'FS3I DYNAMIC/PARTITIONED')": (
                    "Overall FS3I coupling approach.  Accepts "
                    "'fs3i_SequStagg' (one pass per step) or "
                    "'fs3i_IterStagg' (iterate the outer loop to "
                    "convergence; the default).  It lives in the "
                    "FS3I DYNAMIC/PARTITIONED subsection together with "
                    "CONVTOL and ITEMAX -- FS3I DYNAMIC itself has "
                    "none of these three.  There is no FS3I_APPROACH "
                    "key anywhere in 4C."
                ),
                "INF_PERM (in section 'FS3I DYNAMIC')": (
                    "Boolean flag for infinite interface permeability.  "
                    "true means the scalar is simply continuous across "
                    "the FSI interface; false means the interface "
                    "transfer is governed by the finite permeability "
                    "given per surface in DESIGN SCATRA COUPLING SURF "
                    "CONDITIONS."
                ),
                "DESIGN SCATRA COUPLING SURF CONDITIONS": (
                    "The scalar interface condition is a CONDITION, not "
                    "a key: one entry per side of the FSI interface, "
                    "sharing a COUPID, with NUMSCAL / ONOFF / PERMCOEF "
                    "/ CONDUCT / FILTR / WSSON / WSSCOEFFS.  There is "
                    "no SCATRA_COUPLING key anywhere in 4C."
                ),
                "STRUCTSCAL_FIELDCOUPLING / FLUIDSCAL_FIELDCOUPLING "
                "(in section 'FS3I DYNAMIC')": (
                    "Volume coupling between each carrier field and its "
                    "scalar field: 'volume_matching' (identical meshes, "
                    "the usual case for a cloned scalar field) or "
                    "'volume_nonmatching'."
                ),
            },
            "pitfalls": [
                (
                    "[Input] FS3I is a five-field problem: "
                    "fluid + structure + ALE + fluid-side "
                    "scalar + structure-side scalar. All "
                    "five must be configured. Signal: 4C does "
                    "police them, but never by naming the "
                    "missing section. Deleting SCALAR "
                    "TRANSPORT DYNAMIC trips FS3I's own "
                    "cross-field check in fs3i/4C_fs3i.cpp — "
                    "'Parameter(s) theta for one-step-theta "
                    "time-integration scheme defined in one "
                    "or more of the individual fields do(es) "
                    "not match for partitioned FS3I "
                    "computation.' — which names THETA, the "
                    "parameter that silently reverted to its "
                    "default. Deleting ALE DYNAMIC gives the "
                    "generic adapter/4C_adapter_ale.cpp "
                    "linear-solver message instead. The quoted "
                    "'FS3I field N not found' does not exist "
                    "in 4C, and neither does the file it was "
                    "attributed to, 4C_fs3i_factory.cpp. "
                    "Start from an FS3I tutorial: "
                    "greenfield is too error-prone for "
                    "5-field problems. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] CLONING MATERIAL MAP needs "
                    "THREE entries for a working FS3I deck, "
                    "not two: fluid -> scatra1, "
                    "structure -> scatra2, and fluid -> ale. "
                    "Signal: none — a missing entry gives no "
                    "4C diagnostic whatsoever. An uncaught "
                    "std::out_of_range terminates the "
                    "process, with zero 'PROC 0 ERROR in' "
                    "lines, no field name, no material id and "
                    "no occurrence of the word 'cloning'. "
                    "There is nothing to grep for, so count "
                    "the entries against the fields by hand. "
                    "The message 'cannot clone material for "
                    "<field>' does not exist. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Scalar transport in the "
                    "fluid domain is ADVECTION-DOMINATED, and "
                    "SCALAR TRANSPORT DYNAMIC/STABILIZATION/"
                    "STABTYPE is live: switching it between "
                    "no_stabilization and SUPG (with "
                    "DEFINITION_TAU: Codina) moves both "
                    "scalar fields while leaving the fluid "
                    "velocity untouched. Signal: none — "
                    "neither setting reports a Peclet number, "
                    "warns that an advection-dominated field "
                    "is running unstabilised, or fails to "
                    "converge. The visible oscillations are "
                    "not a diagnostic you can wait for; the "
                    "difference has to be looked for on "
                    "purpose against a stabilised reference. "
                    "(Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Structure-side scalar "
                    "typically has MUCH LOWER diffusivity "
                    "than fluid-side (~1e-9 vs 1e-3 m^2/s "
                    "for drug transport). Signal: none — a "
                    "nine-decade contrast does NOT make the "
                    "coupling iterate between two non-"
                    "converging states. It converges "
                    "normally, reaches the result-test "
                    "manager, and simply returns a different "
                    "answer on both scalar fields. And the "
                    "usual remedy is unavailable: FS3I "
                    "DYNAMIC/PARTITIONED accepts exactly "
                    "COUPALGO, CONVTOL and ITEMAX, so there "
                    "is no relaxation parameter to set — "
                    "adding STARTOMEGA is rejected at parse "
                    "time. Validate against a reference "
                    "solution rather than against "
                    "convergence. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] FS3I fluid elements need the "
                    "ALE kinematic flag, written as an "
                    "element-line token (`... MAT 1 NA ALE`), "
                    "not as a YAML key. Scalar transport on "
                    "the FLUID mesh also uses the ALE "
                    "velocity. Signal: Eulerian fluid "
                    "elements abort in "
                    "coupling/src/adapter/"
                    "4C_coupling_adapter.cpp with 'got N "
                    "master nodes but 0 slave nodes for "
                    "coupling' — the ALE discretisation is "
                    "never built, so the FSI interface has "
                    "nothing to couple to. The log contains "
                    "no occurrence of 'kinematic' or 'moving "
                    "mesh' at all, and no time step is "
                    "reached. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] The outer scalar-coupling loop "
                    "is configured in the FS3I "
                    "DYNAMIC/PARTITIONED subsection — "
                    "COUPALGO, CONVTOL and ITEMAX — not in "
                    "FS3I DYNAMIC itself, which has no "
                    "ITEMAX. Signal: writing ITEMAX in FS3I "
                    "DYNAMIC aborts at parse time in "
                    "core/io/src/4C_io_input_spec_builders.cpp "
                    "with 'Could not match this input' and "
                    "the key listed under '[!] The following "
                    "data remains unused:'. The outer loop is "
                    "in any case bounded by the ITEMAX "
                    "default whether or not you set it, so it "
                    "cannot iterate indefinitely. "
                    "(Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Do NOT assume every field "
                    "wants the same time step. The FS3I "
                    "DYNAMIC TIMESTEP is the OUTER coupling "
                    "step and 4C's own passing FS3I "
                    "regression decks deliberately set it "
                    "smaller than the FSI DYNAMIC and SCALAR "
                    "TRANSPORT DYNAMIC step. Signal: making "
                    "the coupling step equal to the field "
                    "step does not merely drift — it can fail "
                    "outright with "
                    "'Core::LinearSolver::BelosSolver: "
                    "Iterative solver did not converge.' from "
                    "core/linear_solver/src/method/"
                    "4C_linear_solver_method_iterative.cpp, "
                    "producing no result at all, so there is "
                    "nothing to compare a percentage against. "
                    "Take the step ratio from a working "
                    "reference deck. (Corrected by execution "
                    "2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "drug_eluting_stent_3d",
                    "description": (
                        "A drug-eluting stent in a pulsatile flow vessel.  "
                        "The FSI captures wall motion, the fluid-side "
                        "scalar tracks drug concentration in blood, and "
                        "the structure-side scalar models drug diffusion "
                        "through the vessel wall.  Tests the full 5-field "
                        "FS3I coupling."
                    ),
                    "template_variant": "fs3i_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "fs3i_3d",
                "description": (
                    "3-D FS3I: fluid-structure interaction with scalar "
                    "transport in both domains.  Neo-Hookean structure, "
                    "Newtonian fluid, ALE mesh motion, advection-"
                    "diffusion scalar transport.  UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "fs3i_3d") -> str:
        templates = {
            "fs3i_3d": self._template_fs3i_3d,
        }
        if variant == "default":
            variant = "fs3i_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_fs3i_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D FS3I (Fluid-Porous-Structure-Scalar-Scalar Interaction)
            #
            # 5-field coupling: fluid + structure + ALE + fluid-side scalar
            # + structure-side scalar.  The FSI sub-problem handles flow and
            # wall motion; the scalar transport fields model mass transfer
            # (e.g. drug concentration) in both domains.
            #
            # Mesh: requires exodus files with:
            #   "fsi.e" or separate fluid/structure meshes
            #   element_block 1 = structure (HEX8)
            #   element_block 2 = fluid (HEX8)
            #   node_set 1 = structure fixed end
            #   node_set 2 = FSI interface (structure side)
            #   node_set 3 = fluid inlet
            #   node_set 4 = fluid walls
            #   node_set 5 = FSI interface (fluid side)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D FS3I -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Porous_Structure_Scalar_Scalar_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "GenAlpha"
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>
            STRUCTURAL DYNAMIC/GENALPHA:
              RHO_INF: <genalpha_rho_inf>

            # == Fluid =========================================================
            FLUID DYNAMIC:
              TIMEINTEGR: "Np_Gen_Alpha"
              LINEAR_SOLVER: 2
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == ALE mesh motion ===============================================
            ALE DYNAMIC:
              ALE_TYPE: "springs_spatial"
              LINEAR_SOLVER: 1
              MAXITER: <ale_max_iterations>

            # == FSI coupling ==================================================
            FSI DYNAMIC:
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              SECONDORDER: true
            FSI DYNAMIC/MONOLITHIC SOLVER:
              SHAPEDERIVATIVES: true

            # == Scalar transport ==============================================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              # SCALAR TRANSPORT DYNAMIC/TIMEINTEGR spells the scheme with
              # underscores: BDF2 | Gen_Alpha | One_Step_Theta | Stationary.
              # ("OneStepTheta", the STRUCTURAL DYNAMIC spelling, is rejected.)
              TIMEINTEGR: "One_Step_Theta"
              THETA: <scatra_theta>
              TIMESTEP: <scatra_timestep>
              NUMSTEP: <scatra_num_steps>
              LINEAR_SOLVER: 3
              VELOCITYFIELD: "Navier_Stokes"

            # == FS3I coupling =================================================
            # FS3I DYNAMIC holds only the outer time loop and the
            # structure-scalar configuration.  The outer ITERATION loop
            # (COUPALGO / CONVTOL / ITEMAX) lives in the separate
            # top-level key "FS3I DYNAMIC/PARTITIONED" -- writing any of
            # those three in FS3I DYNAMIC aborts at parse time.
            FS3I DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              RESULTSEVERY: <results_output_interval>
              SCATRA_SOLVERTYPE: "nonlinear"
              INF_PERM: <infinite_interface_permeability>
              COUPLED_LINEAR_SOLVER: 3
              LINEAR_SOLVER1: 3
              LINEAR_SOLVER2: 3
            FS3I DYNAMIC/PARTITIONED:
              COUPALGO: "fs3i_IterStagg"
              CONVTOL: <fs3i_convergence_tolerance>
              ITEMAX: <fs3i_max_coupling_iterations>
            FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION:
              STABTYPE: "<structure_scalar_stabilization_type>"
              EVALUATION_TAU: "integration_point"
              EVALUATION_MAT: "integration_point"

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_ale_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "fluid_solver"
            SOLVER 3:
              SOLVER: "UMFPACK"
              NAME: "scatra_solver"

            # == Materials =====================================================
            MATERIALS:
              # Fluid material
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>
              # Structure material (Neo-Hookean)
              - MAT: 2
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [3]
                  DENS: <structure_density>
              - MAT: 3
                ELAST_CoupNeoHooke:
                  YOUNG: <structure_Young_modulus>
              # ALE pseudo-material
              - MAT: 4
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <ale_Young_modulus>
                  NUE: <ale_Poisson_ratio>
                  DENS: <ale_density>
              # Fluid-side scalar transport material
              - MAT: 5
                MAT_scatra:
                  DIFFUSIVITY: <fluid_scalar_diffusivity>
              # Structure-side scalar transport material
              - MAT: 6
                MAT_scatra:
                  DIFFUSIVITY: <structure_scalar_diffusivity>

            # Three entries are required: each scalar field is cloned from
            # the field that carries it (fluid -> scatra1, structure ->
            # scatra2) and the ALE mesh is cloned from the fluid.  There is
            # no scatra1 -> scatra2 clone; writing one kills the process
            # with an uncaught std::out_of_range and no 4C diagnostic.
            CLONING MATERIAL MAP:
              - SRC_FIELD: "fluid"
                SRC_MAT: 1
                TAR_FIELD: "scatra1"
                TAR_MAT: 5
              - SRC_FIELD: "structure"
                SRC_MAT: 2
                TAR_FIELD: "scatra2"
                TAR_MAT: 6
              - SRC_FIELD: "fluid"
                SRC_MAT: 1
                TAR_FIELD: "ale"
                TAR_MAT: 4

            # == Boundary Conditions ===========================================

            # 4C has no fluid-specific Dirichlet section: DESIGN SURF DIRICH
            # CONDITIONS carries structure (NUMDOF 3) and fluid
            # (NUMDOF 4 = vx vy vz p) entries alike.  Only ALE, TRANSPORT,
            # PORO and THERMO have their own DESIGN SURF ... DIRICH sections.
            DESIGN SURF DIRICH CONDITIONS:
              # Structure: fixed support
              - E: <structure_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]
              # Fluid: inlet velocity
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

            # ALE: fix outer boundaries
            DESIGN SURF ALE DIRICH CONDITIONS:
              - E: <ale_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Scalar transport: inlet concentration
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <scatra_inlet_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<inlet_scalar_concentration>]
                FUNCT: [0]

            # FSI coupling interface
            DESIGN FSI COUPLING SURF CONDITIONS:
              - E: <fsi_interface_structure_id>
                coupling_id: 1
              - E: <fsi_interface_fluid_id>
                coupling_id: 1

            # Scalar interface coupling.  This CONDITION -- not a key in
            # FS3I DYNAMIC -- is what couples the two scalar fields across
            # the FSI interface.  One entry per side, sharing a COUPID.
            DESIGN SCATRA COUPLING SURF CONDITIONS:
              - E: <fsi_interface_structure_id>
                NUMSCAL: 1
                ONOFF: [1]
                COUPID: 1
                PERMCOEF: <interface_permeability_coefficient>
                CONDUCT: <interface_conductivity>
                FILTR: <interface_filtration_coefficient>
                WSSON: false
                WSSCOEFFS: [0, 0]
              - E: <fsi_interface_fluid_id>
                NUMSCAL: 1
                ONOFF: [1]
                COUPID: 1
                PERMCOEF: <interface_permeability_coefficient>
                CONDUCT: <interface_conductivity>
                FILTR: <interface_filtration_coefficient>
                WSSON: false
                WSSCOEFFS: [0, 0]

            # Inlet ramp function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<structure_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>

            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 2
                  FLUID:
                    HEX8:
                      MAT: 1
                      NA: ALE

            RESULT DESCRIPTION:
              - FLUID:
                  DIS: "fluid"
                  NODE: <result_fluid_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_fluid_velocity>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_structure_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
              - SCATRA:
                  DIS: "scatra1"
                  NODE: <result_scatra_node_id>
                  QUANTITY: "phi1"
                  VALUE: <expected_concentration>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check fluid viscosity
        viscosity = params.get("DYNVISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"DYNVISCOSITY must be > 0, got {mu}."
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
                        f"DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check diffusivity
        for key in ("DIFFUSIVITY", "fluid_scalar_diffusivity",
                     "structure_scalar_diffusivity"):
            diff = params.get(key)
            if diff is not None:
                try:
                    d = float(diff)
                    if d <= 0:
                        issues.append(
                            f"{key} must be > 0, got {d}."
                        )
                except (TypeError, ValueError):
                    issues.append(
                        f"{key} must be a positive number, got {diff!r}."
                    )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check CLONING MATERIAL MAP
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for FS3I.  It needs "
                "THREE entries: fluid -> scatra1, structure -> scatra2 "
                "and fluid -> ale.  There is no scatra1 -> scatra2 "
                "entry."
            )

        # FS3I_APPROACH / SCATRA_COUPLING do not exist in 4C, and
        # CONVTOL / ITEMAX belong to FS3I DYNAMIC/PARTITIONED.
        for bogus in ("FS3I_APPROACH", "SCATRA_COUPLING"):
            if params.get(bogus) is not None:
                issues.append(
                    f"{bogus} is not a 4C input parameter.  Use "
                    f"FS3I DYNAMIC/PARTITIONED COUPALGO for the "
                    f"coupling scheme and DESIGN SCATRA COUPLING SURF "
                    f"CONDITIONS (plus FS3I DYNAMIC INF_PERM) for the "
                    f"scalar interface condition."
                )

        # Check fluid NA mode
        fluid_na = params.get("fluid_NA") or params.get("NA")
        if fluid_na is not None:
            if str(fluid_na).upper() != "ALE":
                issues.append(
                    f"Fluid elements MUST use NA: ALE for FS3I, "
                    f"got {fluid_na!r}."
                )

        return issues
