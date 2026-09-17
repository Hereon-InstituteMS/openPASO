"""XFEM Fluid-Structure Interaction (FSI XFEM) generator for 4C.

Covers FSI problems where the fluid-structure interface is captured via
XFEM instead of body-fitted (ALE) mesh motion.  The structural mesh moves
through a fixed background fluid mesh, which is enriched with XFEM
discontinuities at the interface.  This approach avoids ALE mesh
distortion issues for large structural deformations.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FSIXFEMGenerator(BaseGenerator):
    """Generator for XFEM-based FSI problems in 4C."""

    module_key = "fsi_xfem"
    display_name = "FSI XFEM (Fluid-Structure Interaction with XFEM)"
    problem_type = "Fluid_Structure_Interaction_XFEM"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "XFEM-based Fluid-Structure Interaction couples an "
                "incompressible Navier-Stokes fluid with a deformable "
                "structure using the eXtended Finite Element Method for "
                "the fluid field.  Unlike classical ALE-based FSI, the "
                "fluid mesh is FIXED (Eulerian) and the structural mesh "
                "moves through it.  The fluid approximation space is "
                "enriched with XFEM basis functions at the interface to "
                "capture the velocity and pressure discontinuities.  "
                "Nitsche's method enforces the interface kinematic and "
                "traction conditions weakly.  The PROBLEM TYPE is "
                "'Fluid_Structure_Interaction_XFEM'.  Required dynamics "
                "sections are STRUCTURAL DYNAMIC, FLUID DYNAMIC and FSI "
                "DYNAMIC (with COUPALGO: iter_xfem_monolithic), plus the "
                "XFEM settings.  There is NO section called 'XFLUID "
                "DYNAMIC': the XFEM settings are split over 'XFEM GENERAL' "
                "(cut and integration scheme) and 'XFLUID "
                "DYNAMIC/STABILIZATION' (coupling method, Nitsche penalty, "
                "ghost penalty).  Those slash-joined names are single "
                "literal top-level YAML keys, not nested maps.  The "
                "interface itself is declared by DESIGN XFEM FSI MONOLITHIC "
                "SURF CONDITIONS (or DESIGN XFEM FSI PARTITIONED SURF "
                "CONDITIONS) on the structure surface, keyed by COUPLINGID; "
                "without it there is no fluid-structure interface.  No ALE "
                "DYNAMIC section is needed (this is a key advantage over "
                "standard FSI).  Ghost-penalty stabilisation prevents "
                "ill-conditioning from small cut elements.  Result tests "
                "on the fluid field are named XFLUID, not FLUID."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "XFEM GENERAL",
                "XFLUID DYNAMIC/STABILIZATION",
                "FSI DYNAMIC",
                "FSI DYNAMIC/MONOLITHIC SOLVER",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
                "DESIGN XFEM FSI MONOLITHIC SURF CONDITIONS",
            ],
            "optional_sections": [
                "XFLUID DYNAMIC/GENERAL",
                "CUT GENERAL",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/EDGE-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "FSI DYNAMIC/PARTITIONED SOLVER",
                "STRUCTURAL DYNAMIC/GENALPHA",
                "DESIGN FSI COUPLING SURF CONDITIONS",
                # NOTE: IO/RUNTIME VTK OUTPUT is NOT usable here -- it aborts
                # this problem type before the first step.  See pitfalls.
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid for the background fluid domain."
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
                "MAT_ElastHyper (Neo-Hooke)": {
                    "description": (
                        "Hyperelastic material for the structure.  "
                        "Suitable for large-deformation FSI where ALE "
                        "would fail."
                    ),
                    "parameters": {
                        "NUMMAT": {
                            "description": "Number of sub-materials",
                            "range": "1",
                        },
                        "MATIDS": {
                            "description": "List of sub-material IDs",
                            "range": "",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "St. Venant-Kirchhoff material for small-strain "
                        "structural response."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus [Pa]",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "fluid_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Direct solver recommended for XFEM due to "
                        "variable system size from cut elements."
                    ),
                },
                "structure_solver": {
                    "type": "UMFPACK",
                    "notes": "Standard structural solver.",
                },
            },
            "xfem_fsi_parameters": {
                "COUPLING_METHOD": (
                    "In XFLUID DYNAMIC/STABILIZATION.  'Nitsche' (default, "
                    "recommended) for weak enforcement of velocity "
                    "continuity and traction equilibrium at the FSI "
                    "interface.  The only other accepted values are "
                    "'Hybrid_LM_Cauchy_stress' and "
                    "'Hybrid_LM_viscous_stress'; there is no 'penalty'."
                ),
                "NIT_STAB_FAC": (
                    "In XFLUID DYNAMIC/STABILIZATION.  The Nitsche penalty "
                    "factor (default 35), with NIT_STAB_FAC_TANG for the "
                    "tangential term.  There is no NITSCHE_PENALTY_PARAMETER "
                    "key in 4C."
                ),
                "GHOST_PENALTY": (
                    "Stabilisation for small cut elements.  Essential "
                    "for robustness when the structural mesh passes "
                    "close to fluid element boundaries.  All of its keys "
                    "live in XFLUID DYNAMIC/STABILIZATION -- there is no "
                    "'XFLUID DYNAMIC/GHOST PENALTY' section.  Switch it on "
                    "with GHOST_PENALTY_STAB (default false) and scale it "
                    "with GHOST_PENALTY_FAC; the transient counterparts are "
                    "GHOST_PENALTY_TRANSIENT_STAB (note the _STAB suffix) "
                    "and GHOST_PENALTY_TRANSIENT_FAC."
                ),
                "COUPALGO": (
                    "In FSI DYNAMIC.  Use 'iter_xfem_monolithic' -- that is "
                    "the only XFEM entry in 4C's COUPALGO enum.  There is no "
                    "XFEM_FSI_COUPALGO key, and no 'xfem_monolithic' or "
                    "'xfem_partitioned' value."
                ),
            },
            "pitfalls": [
                (
                    '[Input] FSI-XFEM does not use ALE mesh motion: the fluid mesh '
                    'is fixed and the structure interface cuts through it via XFEM '
                    'enrichment. But 4C does NOT object to leftover ALE plumbing. '
                    'Signal: a deck carrying BOTH an ALE DYNAMIC section and a '
                    "CLONING MATERIAL MAP runs to the exit banner, whose literal is "
                    "'finished normally' with the rank printed into it "
                    "(processor 0 finished normally), "
                    "and reproduces the reference results; there is no 'XFEM and "
                    "ALE are mutually exclusive' message and no "
                    '4C_xfem_fluid_setup.cpp in the source. Note also that this '
                    "problem type creates a discretisation named 'ale' regardless, "
                    "so grepping the log for 'ale' proves nothing either way. "
                    'Remove the sections because they are meaningless, not because '
                    '4C will tell you. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Numerical] Ghost-penalty stabilisation controls the '
                    'conditioning of cut elements with small volume fractions, but '
                    'on a monolithic XFSI problem its absence shows up as a wrong '
                    'answer, not a solver error. Signal: with GHOST_PENALTY_STAB: '
                    'false, or GHOST_PENALTY_FAC: 0.0, the Newton loop converges, '
                    'the run reaches its result test and the pinned values move; no '
                    'condition number, singular-matrix or factorisation message is '
                    'printed, and a direct solver such as UMFPACK factorises '
                    'without complaint. Compare against a reference run rather than '
                    'waiting for the solver to object. (Audit 2026-06-02; corrected '
                    'by execution 2026-08-06.)'
                ),
                (
                    '[Mesh] The structural mesh acts as the CUTTER MESH for the '
                    'fluid XFEM enrichment and its coupled surface must be closed. '
                    'An open surface is caught, but by the quadrature, not by a '
                    'classification check. Signal: dropping one side of the '
                    "coupling surface aborts with 'negative volume predicted by the "
                    "DirectDivergence integration rule;' from "
                    '4C_cut_direct_divergence.cpp, usually alongside a '
                    "Cut::Mesh::DebugDump line. There is no 'inside/outside "
                    "classification inconsistent' message and no silent permeation: "
                    'the run stops. Check that every face of the cutter body '
                    'carries the coupling condition. (Audit 2026-06-02; corrected '
                    'by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] The Nitsche penalty is NIT_STAB_FAC in XFLUID '
                    'DYNAMIC/STABILIZATION (default 35); 4C applies the viscosity '
                    'and cut-size scaling itself via VISC_STAB_TRACE_ESTIMATE and '
                    'VISC_STAB_HK, so the value is a bare dimensionless factor. '
                    'Signal: setting it far too low or far too high does NOT stall '
                    'Newton and does not print a condition number. The run '
                    'converges, reaches the result test, and the coupled '
                    'displacements and velocities are simply wrong. Leave it at the '
                    'default unless a reference solution tells you otherwise. '
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Choose the time step so the structure does not '
                    'traverse more than one fluid element per step; cut-topology '
                    'changes are reconstructed by a semi-Lagrangean search for '
                    'nodes the interface has just uncovered. Signal: too large a '
                    "step aborts with '<<< WARNING! Initial point for node {} for "
                    "finding the Lagrangean origin not in' plus 'domain! >>>', "
                    'two adjacent literals at '
                    '4C_xfem_xfluid_timeInt_std_SemiLagrange.cpp:167-168 with the '
                    'node id filled in between them. Despite being '
                    'printed with a WARNING prefix it is thrown, not logged: the '
                    'run stops and never reaches its result test. Reduce TIMESTEP '
                    'until the message disappears. (Audit 2026-06-02; corrected by '
                    'execution 2026-08-06.)'
                ),
                (
                    '[Output] You cannot inspect an XFSI fluid field in ParaView '
                    'through runtime VTK output, because 4C refuses to produce it. '
                    'Signal: adding IO/RUNTIME VTK OUTPUT with the FLUID and '
                    'STRUCTURE sub-sections aborts before the first step with '
                    "'Runtime output is not available in the old structure time "
                    'integration! You need to take the new one, i.e. set '
                    "INT_STRATEGY: Standard!' from 4C_structure_timint.cpp, and "
                    'setting INT_STRATEGY: Standard does not help, because the XFSI '
                    'adapter builds the legacy integrator regardless and the same '
                    'throw reappears. No .vtu is written either way. Use the '
                    'Ensight .result files or the Gmsh .pos output instead. (Audit '
                    '2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] Fluid element blocks must use NA: Euler; there is no '
                    'mesh motion in the fluid domain under XFEM. Signal: NA: ALE '
                    'parses cleanly, builds the discretisations, and then aborts '
                    "with 'Cannot find state {} in discretization {}' from "
                    '4C_fem_discretization.hpp:1849, rendered at run time as '
                    'Cannot find state dispnp in discretization fluid. '
                    'The message names neither XFEM nor '
                    'ALE nor the NA keyword that was mis-set, so it is easy to '
                    'misread as an output or restart problem. The quoted '
                    "'kinematic type incompatible' does not exist in 4C, and "
                    'neither does the file it was attributed to, '
                    '4C_fluid_xfem_factory.cpp. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
            ],
            "typical_experiments": [
                {
                    "name": "falling_sphere_xfem_3d",
                    "description": (
                        "A rigid or elastic sphere falling through a "
                        "viscous fluid.  The sphere surface cuts through "
                        "the fixed fluid mesh via XFEM.  Tests Nitsche "
                        "coupling, ghost penalty, and structural motion "
                        "through the fluid."
                    ),
                    "template_variant": "xfem_fsi_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "xfem_fsi_3d",
                "description": (
                    "3-D XFEM FSI: structure moving through fixed "
                    "fluid mesh.  Nitsche interface coupling, "
                    "ghost-penalty stabilisation, UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "xfem_fsi_3d") -> str:
        templates = {
            "xfem_fsi_3d": self._template_xfem_fsi_3d,
        }
        if variant == "default":
            variant = "xfem_fsi_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_xfem_fsi_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D XFEM-Based Fluid-Structure Interaction
            #
            # A deformable structure is immersed in a fixed background fluid
            # mesh.  The fluid-structure interface is captured via XFEM
            # enrichment (no ALE mesh motion needed).  Nitsche's method
            # enforces kinematic and traction coupling at the interface.
            #
            # Mesh: requires TWO exodus files:
            #   Fluid mesh: "fluid_bg.e" with
            #     element_block 1 = background fluid (HEX8)
            #     node_set 1 = inlet
            #     node_set 2 = outlet
            #     node_set 3 = walls (no-slip)
            #   Structure mesh: "structure.e" with
            #     element_block 1 = structure (HEX8)
            #     node_set 1 = structure Dirichlet (if any)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D XFEM FSI -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Structure_Interaction_XFEM"
            IO:
              STDOUTEVERY: <stdout_interval>
            # Do NOT add IO/RUNTIME VTK OUTPUT here.  On this problem type it
            # aborts before the first step with "Runtime output is not
            # available in the old structure time integration! You need to
            # take the new one, i.e. set `INT_STRATEGY: Standard`!" -- and
            # setting INT_STRATEGY: Standard does not help, because the XFSI
            # adapter builds the legacy integrator regardless.  Read results
            # from the Ensight .result files, or set OUTPUT_GMSH in IO plus
            # GMSH_SOL_OUT in XFEM GENERAL and read the Gmsh .pos files.

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "GenAlpha"
              TIMESTEP: <structure_timestep>
              NUMSTEP: <structure_num_steps>
              LINEAR_SOLVER: 2
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
              LINEAR_SOLVER: 1
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == XFEM settings =================================================
            # 'XFLUID DYNAMIC' is NOT a section on its own -- 4C aborts with
            # "Section 'XFLUID DYNAMIC' is not a valid section name."  The real
            # top-level keys are the slash-joined literals below, written as
            # ONE key each (not nested maps).  Cut/integration settings live in
            # the separate 'XFEM GENERAL' section.  There is no
            # 'XFLUID DYNAMIC/GHOST PENALTY' section either: the ghost-penalty
            # keys are part of XFLUID DYNAMIC/STABILIZATION.
            XFEM GENERAL:
              # Tessellation | DirectDivergence  (MomentFitting segfaults)
              VOLUME_GAUSS_POINTS_BY: "<volume_integration_scheme>"
              BOUNDARY_GAUSS_POINTS_BY: "<boundary_integration_scheme>"
            XFLUID DYNAMIC/STABILIZATION:
              # Nitsche | Hybrid_LM_Cauchy_stress | Hybrid_LM_viscous_stress
              COUPLING_METHOD: "<coupling_method>"
              # Nitsche penalty factor (default 35).  There is no
              # NITSCHE_PENALTY_PARAMETER key in 4C.
              NIT_STAB_FAC: <nitsche_penalty_factor>
              NIT_STAB_FAC_TANG: <nitsche_penalty_factor_tangential>
              GHOST_PENALTY_STAB: true
              GHOST_PENALTY_FAC: <ghost_penalty_factor>
              GHOST_PENALTY_TRANSIENT_STAB: true
              GHOST_PENALTY_TRANSIENT_FAC: <ghost_penalty_transient_factor>

            # == FSI coupling ==================================================
            FSI DYNAMIC:
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              # iter_xfem_monolithic is the ONLY XFEM value of COUPALGO
              COUPALGO: "iter_xfem_monolithic"
              RESULTSEVERY: <results_output_interval>
            FSI DYNAMIC/MONOLITHIC SOLVER:
              ITEMAX: <fsi_max_newton_iterations>
              INFNORMSCALING: false
              TOL_DIS_RES_L2: <fsi_displacement_residual_tolerance>
              TOL_VEL_RES_L2: <fsi_velocity_residual_tolerance>

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "fluid_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
            MATERIALS:
              # Fluid material
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>
              # Structure material (Neo-Hookean hyperelastic)
              - MAT: 2
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [3]
                  DENS: <structure_density>
              - MAT: 3
                ELAST_CoupNeoHooke:
                  YOUNG: <structure_Young_modulus>

            # == Boundary Conditions ===========================================

            # There is no 'DESIGN SURF STRUCT DIRICH CONDITIONS' section -- 4C
            # aborts with "Section 'DESIGN SURF STRUCT DIRICH CONDITIONS' is
            # not a valid section name."  Structure and fluid surface
            # Dirichlets share the ONE section 'DESIGN SURF DIRICH CONDITIONS';
            # each entry is routed by the node set it names, and NUMDOF tells
            # 4C which field it belongs to (3 for the structure, dim+1 = 4 for
            # the fluid).  The field-specific variants that do exist are
            # DESIGN SURF ALE / PORO / THERMO / TRANSPORT DIRICH CONDITIONS --
            # there is no STRUCT variant, because plain DIRICH is the
            # structural one.
            #
            # ENTITY_TYPE is REQUIRED whenever the geometry comes from a mesh
            # FILE: without it 4C aborts with "legacy_id condition N uses
            # legacy_id entity type but no legacy entities were defined".
            DESIGN SURF DIRICH CONDITIONS:
              # Fluid: inlet
              - E: <inlet_node_set_id>
                ENTITY_TYPE: "node_set_id"
                NUMDOF: 4
                ONOFF: [1, 1, 1, 0]
                VAL: [<inlet_velocity_x>, <inlet_velocity_y>, <inlet_velocity_z>, 0.0]
                FUNCT: [<inlet_ramp_function>, 0, 0, 0]
              # Fluid: no-slip walls
              - E: <wall_node_set_id>
                ENTITY_TYPE: "node_set_id"
                NUMDOF: 4
                ONOFF: [1, 1, 1, 0]
                VAL: [0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0]
              # Structure Dirichlet (optional: constrain motion)
              - E: <structure_dirichlet_node_set_id>
                ENTITY_TYPE: "node_set_id"
                NUMDOF: 3
                ONOFF: [<dof1_fix>, <dof2_fix>, <dof3_fix>]
                VAL: [<dof1_val>, <dof2_val>, <dof3_val>]
                FUNCT: [<dof1_funct>, <dof2_funct>, <dof3_funct>]

            # The fluid-structure interface: the structure surface that cuts
            # the background fluid mesh.  Without this condition there is no
            # XFEM interface at all.
            DESIGN XFEM FSI MONOLITHIC SURF CONDITIONS:
              - E: <fsi_interface_node_set_id>
                ENTITY_TYPE: "node_set_id"
                COUPLINGID: 1

            # Inlet ramp function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry ======================================================
            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    HEX8:
                      MAT: 1
                      NA: Euler

            STRUCTURE GEOMETRY:
              FILE: "<structure_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>

            # The fluid result field is named XFLUID, not FLUID.  A 'FLUID'
            # entry parses but is never run: 4C then aborts with
            # "expected N tests but performed 0".
            RESULT DESCRIPTION:
              - XFLUID:
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

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(
                        f"YOUNG must be > 0, got {e}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check structural density
        dens = params.get("DENS") or params.get("structure_density")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(
                        f"Structural DENS must be > 0, got {d}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENS must be a positive number, got {dens!r}."
                )

        # Reject the fabricated Nitsche key outright: 4C has no such
        # parameter and would abort with "Could not match this input".
        if params.get("NITSCHE_PENALTY_PARAMETER") is not None:
            issues.append(
                "NITSCHE_PENALTY_PARAMETER does not exist in 4C.  The "
                "Nitsche penalty factor is NIT_STAB_FAC (default 35) in "
                "XFLUID DYNAMIC/STABILIZATION, with NIT_STAB_FAC_TANG for "
                "the tangential term."
            )

        # Check the real Nitsche penalty keys
        for key in ("NIT_STAB_FAC", "NIT_STAB_FAC_TANG"):
            nitsche = params.get(key)
            if nitsche is None:
                continue
            try:
                n = float(nitsche)
                if n <= 0:
                    issues.append(f"{key} must be > 0, got {n}.")
            except (TypeError, ValueError):
                issues.append(
                    f"{key} must be a positive number, got {nitsche!r}."
                )

        # Check coupling method against 4C's actual enum
        coupling = params.get("COUPLING_METHOD")
        if coupling is not None and coupling not in (
            "Nitsche", "Hybrid_LM_Cauchy_stress", "Hybrid_LM_viscous_stress",
        ):
            issues.append(
                f"COUPLING_METHOD must be one of 'Nitsche', "
                f"'Hybrid_LM_Cauchy_stress', 'Hybrid_LM_viscous_stress', "
                f"got {coupling!r}.  ('penalty' is not a 4C value.)"
            )

        # Check the FSI coupling algorithm
        coupalgo = params.get("COUPALGO")
        if coupalgo is not None and coupalgo != "iter_xfem_monolithic":
            issues.append(
                f"COUPALGO must be 'iter_xfem_monolithic' for XFEM FSI -- "
                f"that is the only XFEM entry in 4C's COUPALGO enum -- "
                f"got {coupalgo!r}."
            )
        if params.get("XFEM_FSI_COUPALGO") is not None:
            issues.append(
                "XFEM_FSI_COUPALGO does not exist in 4C.  Set COUPALGO: "
                "iter_xfem_monolithic in FSI DYNAMIC instead."
            )

        # Warn about ALE sections
        has_ale = params.get("has_ale_dynamic")
        if has_ale:
            issues.append(
                "FSI XFEM does NOT use ALE mesh motion.  Remove the "
                "ALE DYNAMIC section and CLONING MATERIAL MAP."
            )

        # Check fluid NA mode
        fluid_na = params.get("fluid_NA") or params.get("NA")
        if fluid_na is not None:
            if str(fluid_na).upper() != "EULER":
                issues.append(
                    f"Fluid elements MUST use NA: Euler for XFEM FSI "
                    f"(no ALE), got {fluid_na!r}."
                )

        return issues
