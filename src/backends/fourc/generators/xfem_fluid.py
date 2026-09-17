"""XFEM Fluid generator for 4C.

Covers fluid problems with XFEM (eXtended Finite Element Method) interfaces.
The XFEM approach enriches the fluid approximation space to capture
discontinuities (e.g. two-phase interfaces, embedded boundaries, or void
regions) without requiring the mesh to conform to the interface.  The
interface geometry is typically described by a level-set field or a boundary
mesh.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class XFEMFluidGenerator(BaseGenerator):
    """Generator for XFEM fluid problems in 4C."""

    module_key = "xfem_fluid"
    display_name = "XFEM Fluid (Fluid with XFEM Interfaces)"
    problem_type = "Fluid_XFEM"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The XFEM Fluid module solves incompressible Navier-Stokes "
                "problems with discontinuities captured via the eXtended "
                "Finite Element Method.  The fluid mesh does NOT need to "
                "conform to embedded interfaces, void boundaries, or "
                "two-phase fronts.  Instead, the approximation space is "
                "enriched with discontinuous shape functions along the "
                "interface.  The interface geometry is defined either by a "
                "level-set function (a FUNCT referenced through "
                "LEVELSETFIELDNO from a DESIGN XFEM LEVELSET ... VOL "
                "CONDITIONS block) or by an embedded boundary mesh (cutter "
                "mesh: a separate structure discretisation carrying DESIGN "
                "XFEM WEAK DIRICHLET SURF CONDITIONS and DESIGN XFEM "
                "DISPLACEMENT SURF CONDITIONS with a common COUPLINGID).  "
                "The PROBLEM TYPE is 'Fluid_XFEM'.  There is NO section "
                "called 'XFLUID DYNAMIC'.  The XFEM settings are split over "
                "two real top-level sections: 'XFEM GENERAL' (cut and "
                "integration scheme, Gmsh debug output) and 'XFLUID "
                "DYNAMIC/STABILIZATION' (interface coupling method, Nitsche "
                "penalty, ghost penalty).  'XFLUID DYNAMIC/GENERAL' holds "
                "the XFEM time-integration and fluid-fluid options.  These "
                "slash-joined names are single literal top-level YAML keys, "
                "not nested maps.  Standard FLUID DYNAMIC settings "
                "(time integration, stabilisation) are also required.  "
                "Elements use FLUID HEX8 or FLUID TET4 with NA: Euler.  "
                "Result tests use RESULT DESCRIPTION entries named XFLUID, "
                "not FLUID."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "FLUID DYNAMIC",
                "XFEM GENERAL",
                "XFLUID DYNAMIC/STABILIZATION",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "XFLUID DYNAMIC/GENERAL",
                "CUT GENERAL",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/EDGE-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/FLUID",
                # Interface declaration -- one of these, NOT a geometry section:
                "DESIGN XFEM LEVELSET WEAK DIRICHLET VOL CONDITIONS",
                "DESIGN XFEM LEVELSET NEUMANN VOL CONDITIONS",
                "DESIGN XFEM LEVELSET NAVIER SLIP VOL CONDITIONS",
                "DESIGN XFEM WEAK DIRICHLET SURF CONDITIONS",
                "DESIGN XFEM DISPLACEMENT SURF CONDITIONS",
                "DESIGN XFEM NEUMANN SURF CONDITIONS",
                "DESIGN XFEM NAVIER SLIP SURF CONDITIONS",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid material for the background "
                        "fluid domain."
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
                "MAT_fluid (second phase)": {
                    "description": (
                        "Second fluid material for two-phase XFEM "
                        "problems.  Used on the other side of the "
                        "interface."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity of second phase [Pa s]",
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Density of second phase [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "fluid_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Direct solver (UMFPACK) is robust for XFEM since "
                        "the enriched system may have variable size.  "
                        "For large problems use Belos with ILU "
                        "preconditioner."
                    ),
                },
            },
            "xfem_parameters": {
                "COUPLING_METHOD": (
                    "In XFLUID DYNAMIC/STABILIZATION.  Interface coupling "
                    "method.  The ONLY accepted values are 'Nitsche' "
                    "(default, recommended), 'Hybrid_LM_Cauchy_stress' and "
                    "'Hybrid_LM_viscous_stress'.  There is no 'penalty' "
                    "option: 4C rejects it with 'Could not match this "
                    "input' and lists the three legal values."
                ),
                "NIT_STAB_FAC": (
                    "In XFLUID DYNAMIC/STABILIZATION.  The Nitsche penalty "
                    "factor (default 35), with NIT_STAB_FAC_TANG for the "
                    "tangential term.  Dimensionless: 4C applies the "
                    "viscous and element-size scaling itself through "
                    "VISC_STAB_TRACE_ESTIMATE and VISC_STAB_HK.  There is "
                    "no NITSCHE_PENALTY_PARAMETER key anywhere in 4C."
                ),
                "GHOST_PENALTY": (
                    "Ghost-penalty stabilisation for small cut elements.  "
                    "Controls ill-conditioning caused by elements with "
                    "very small cut volumes.  All of its keys live in "
                    "XFLUID DYNAMIC/STABILIZATION -- there is no "
                    "'XFLUID DYNAMIC/GHOST PENALTY' section.  The keys are "
                    "GHOST_PENALTY_STAB, GHOST_PENALTY_FAC, "
                    "GHOST_PENALTY_TRANSIENT_STAB (note the _STAB suffix), "
                    "GHOST_PENALTY_TRANSIENT_FAC, GHOST_PENALTY_2nd_STAB, "
                    "GHOST_PENALTY_2nd_FAC, GHOST_PENALTY_2nd_STAB_NORMAL, "
                    "GHOST_PENALTY_PRESSURE_2nd_FAC and "
                    "GHOST_PENALTY_ADD_INNER_FACES.  All default to false "
                    "or to a small factor, so ghost penalty is OFF unless "
                    "you switch it on."
                ),
                "VOLUME_GAUSS_POINTS_BY": (
                    "In XFEM GENERAL (NOT in any XFLUID DYNAMIC section).  "
                    "Integration scheme for cut volume cells: "
                    "'Tessellation' (default, subdivide into sub-cells), "
                    "'DirectDivergence' (divergence-theorem quadrature) or "
                    "'MomentFitting'.  Use Tessellation or "
                    "DirectDivergence -- see the MomentFitting pitfall "
                    "below.  BOUNDARY_GAUSS_POINTS_BY, same section and "
                    "same three values, controls the boundary cells."
                ),
            },
            "pitfalls": [
                (
                    '[Input] XFEM fluid does NOT use ALE mesh motion. The mesh is '
                    'fixed (Eulerian) and the interface cuts through elements. An '
                    'ALE DYNAMIC section is NOT rejected, though: 4C reads it, '
                    'ignores it, and says nothing, so the deck runs and reproduces '
                    'the Eulerian answer. What does break is asking the fluid '
                    'ELEMENT block for ALE kinematics. Signal: with NA: ALE the run '
                    "parses and then aborts at the first assembly with 'Cannot find "
                    "state {} in discretization {}' from "
                    '4C_fem_discretization.hpp:1849 -- rendered as Cannot find '
                    'state dispnp in discretization fluid, a missing-state message that '
                    'names neither XFEM nor ALE. Keep NA: Euler; do not expect a '
                    'warning about a leftover ALE DYNAMIC block. (Audit 2026-06-02; '
                    'corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Ghost-penalty stabilisation matters for cut '
                    'elements with very small volume fractions, but its absence is '
                    'an ACCURACY failure, not a solver failure. Signal: with '
                    'GHOST_PENALTY_STAB: false (or GHOST_PENALTY_FAC: 0.0) the run '
                    "completes its Newton loop, reaches 'Checking results of N "
                    "tests' and reports different values; no condition number is "
                    'printed, no factorisation fails, and a direct solver such as '
                    "UMFPACK is untroubled. There is no 'Belos: condition number' "
                    "or 'solver diverged after 0 iterations' message in 4C. Detect "
                    'this by comparing against a reference, not by watching the '
                    'solver. (Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] The interface must be described by a level-set field '
                    'or a cutter boundary mesh. If no XFEM coupling condition is '
                    'present, 4C does NOT fall back to standard FEM. Signal: it '
                    "aborts in Cut::CutWizard::safety_checks with 'You have to call "
                    "PrepareCut() before you can call the Cut-routine' from "
                    '4C_cut_cutwizard.cpp, before any time step completes, so no '
                    'result test is reached and there is no non-enriched answer to '
                    'compare with. 4C prints no count of enriched elements at any '
                    'point. (Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] Two-phase level-set XFEM is not configured through a '
                    'material map. DESIGN XFEM LEVELSET TWOPHASE VOL CONDITIONS '
                    'takes only E, COUPLINGID, LEVELSETFIELDNO, BOOLEANTYPE and '
                    'COMPLEMENTARY -- there are no MAT_NEGATIVE or MAT_POSITIVE '
                    'keys anywhere in 4C, and adding them is rejected as unmatched '
                    'condition input. Signal: a correctly spelled two-phase '
                    'condition parses and then dies in Element::location_vector '
                    "with 'wrong number of nodes' from 4C_fem_general_element.cpp; "
                    'the only upstream decks that mention the condition leave it as '
                    'an empty list. Treat two-phase XFEM as unavailable rather than '
                    'mis-specified. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Numerical] Cut elements need a special integration rule, '
                    'chosen with VOLUME_GAUSS_POINTS_BY. Tessellation and '
                    'DirectDivergence are the usable values. Signal: MomentFitting '
                    'does not warn and does not fall back -- it terminates the '
                    'process with SIGSEGV inside '
                    'Core::FE::GaussPointsComposite::num_points, so the log ends '
                    'with the MPI runtime\'s signal block naming a segmentation '
                    'fault, signal 11, and no 4C-level '
                    'diagnostic at all. If a cut run dies without a PROC 0 ERROR '
                    'block, check this key first. (Audit 2026-06-02; corrected by '
                    'execution 2026-08-06.)'
                ),
                (
                    '[Numerical] The Nitsche penalty knob is NIT_STAB_FAC in XFLUID '
                    'DYNAMIC/STABILIZATION (default 35), with NIT_STAB_FAC_TANG for '
                    'the tangential term; the viscous scaling by element size is '
                    'applied internally through VISC_STAB_TRACE_ESTIMATE and '
                    'VISC_STAB_HK. There is no NITSCHE_PENALTY_PARAMETER key. '
                    "Signal: the wrong name fails with 'Could not match this input' "
                    "and 4C lists the section's real keys, including NIT_STAB_FAC. "
                    'Mis-setting the real key gives wrong results rather than a '
                    'solver stall, and on problems the space reproduces exactly it '
                    "changes nothing, because Nitsche's method is consistent. "
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Output] An XFEM fluid writes NO VTU files, per sub-domain or '
                    'otherwise. Signal: with IO/RUNTIME VTK OUTPUT and IO/RUNTIME '
                    'VTK OUTPUT/FLUID enabled, a plain Fluid problem produces '
                    'fluid-*.vtu plus a .pvd, while the same configuration on a '
                    'Fluid_XFEM problem produces neither, exits 0, and never '
                    'mentions that the requested output was skipped. Read XFEM '
                    'results from the legacy Ensight .result file, or set '
                    'OUTPUT_GMSH with GMSH_SOL_OUT and read the Gmsh .pos files. '
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
            ],
            "typical_experiments": [
                {
                    "name": "embedded_cylinder_3d",
                    "description": (
                        "Flow past a cylinder embedded in a background "
                        "fluid mesh via XFEM.  The cylinder surface is "
                        "described by a level-set or boundary mesh.  "
                        "Tests Nitsche coupling, ghost penalty, and "
                        "enriched integration."
                    ),
                    "template_variant": "xfem_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "xfem_3d",
                "description": (
                    "3-D XFEM fluid: flow with an embedded interface "
                    "(void or two-phase).  FLUID HEX8 elements, "
                    "Nitsche coupling, ghost-penalty stabilisation, "
                    "UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "xfem_3d") -> str:
        templates = {
            "xfem_3d": self._template_xfem_3d,
        }
        if variant == "default":
            variant = "xfem_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_xfem_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D XFEM Fluid — Flow with Embedded Interface
            #
            # A fluid domain with an embedded boundary (void, obstacle, or
            # two-phase interface) captured via XFEM.  The background mesh
            # does not conform to the interface.  Nitsche coupling enforces
            # the interface conditions weakly.
            #
            # Mesh: requires exodus file with:
            #   element_block 1 = background fluid (HEX8)
            #   node_set 1 = inlet
            #   node_set 2 = outlet
            #   node_set 3 = walls (no-slip)
            #
            # Cutter mesh: a SECOND file holding the boundary mesh, read as
            #   the structure discretisation, with
            #   element_block 1 = cutter body (HEX8)
            #   node_set 1 = the cutting surface
            #   (or drop it entirely and describe the interface with a
            #   level-set FUNCT instead -- see the note at STRUCTURE GEOMETRY)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D XFEM fluid -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_XFEM"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true

            # == Fluid dynamics =================================================
            FLUID DYNAMIC:
              TIMEINTEGR: "Np_Gen_Alpha"
              TIMESTEP: <fluid_timestep>
              NUMSTEP: <fluid_num_steps>
              MAXTIME: <fluid_max_time>
              LINEAR_SOLVER: 1
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == XFEM-specific settings =========================================
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

            # == Solver =========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "xfem_fluid_solver"

            # == Materials ======================================================
            MATERIALS:
              # Background fluid material
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>
              # Cutter (boundary-mesh) material -- the cutter is a structure
              # discretisation, so it needs a structural material even when it
              # only ever prescribes the interface geometry.
              - MAT: <cutter_material_id>
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <cutter_Young_modulus>
                  NUE: <cutter_Poisson_ratio>
                  DENS: <cutter_density>

            # == Boundary Conditions ============================================

            # Fluid: inlet velocity.
            # ENTITY_TYPE is REQUIRED whenever the geometry comes from a mesh
            # FILE: without it 4C aborts with "legacy_id condition N uses
            # legacy_id entity type but no legacy entities were defined".
            DESIGN SURF DIRICH CONDITIONS:
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

            # Inlet ramp function
            FUNCT<inlet_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == Geometry =======================================================
            FLUID GEOMETRY:
              FILE: "<fluid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    HEX8:
                      MAT: 1
                      NA: Euler

            # Cutter boundary mesh (defines the embedded interface).
            # There is no 'XFEM BOUNDARY GEOMETRY' section -- 4C aborts with
            # "Section 'XFEM BOUNDARY GEOMETRY' is not a valid section name."
            # The cutter is an ordinary STRUCTURE discretisation; what makes it
            # an XFEM interface is the coupling CONDITION below, not the
            # geometry section.  (Alternative, needing no second mesh: drop
            # this block and declare the interface as a level set -- a FUNCT
            # whose zero iso-surface is the interface, referenced by
            # LEVELSETFIELDNO from a DESIGN XFEM LEVELSET WEAK DIRICHLET VOL
            # CONDITIONS entry on the fluid volume.)
            STRUCTURE GEOMETRY:
              FILE: "<cutter_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: <cutter_material_id>
                      KINEM: nonlinear

            # The interface itself: weak (Nitsche) Dirichlet on the cutter
            # surface, plus how that surface moves.  Both blocks must carry the
            # SAME COUPLINGID.  Without an XFEM coupling condition 4C does not
            # fall back to standard FEM -- it aborts inside the cut wizard.
            DESIGN XFEM WEAK DIRICHLET SURF CONDITIONS:
              - E: <cutter_surface_node_set_id>
                ENTITY_TYPE: "node_set_id"
                COUPLINGID: 1
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [<interface_velocity_x>, <interface_velocity_y>, <interface_velocity_z>]
                FUNCT: [0, 0, 0]
            DESIGN XFEM DISPLACEMENT SURF CONDITIONS:
              - E: <cutter_surface_node_set_id>
                ENTITY_TYPE: "node_set_id"
                COUPLINGID: 1
                EVALTYPE: "zero"
                NUMDOF: 3
                ONOFF: [0, 0, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Result tests on an XFEM fluid are named XFLUID, not FLUID.  A
            # 'FLUID' entry parses but is never run: 4C then aborts with
            # "expected N tests but performed 0".
            RESULT DESCRIPTION:
              - XFLUID:
                  DIS: "fluid"
                  NODE: <result_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_velocity>
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

        # Reject the fabricated Nitsche key outright: 4C has no such
        # parameter and would abort with "Could not match this input".
        if params.get("NITSCHE_PENALTY_PARAMETER") is not None:
            issues.append(
                "NITSCHE_PENALTY_PARAMETER does not exist in 4C.  The "
                "Nitsche penalty factor is NIT_STAB_FAC (default 35) in "
                "XFLUID DYNAMIC/STABILIZATION, with NIT_STAB_FAC_TANG for "
                "the tangential term."
            )

        # Check Nitsche penalty factor (the real key)
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

        # Check ghost penalty factor
        gpf = params.get("GHOST_PENALTY_FAC")
        if gpf is not None:
            try:
                g = float(gpf)
                if g <= 0:
                    issues.append(
                        f"GHOST_PENALTY_FAC must be > 0, got {g}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"GHOST_PENALTY_FAC must be a positive number, "
                    f"got {gpf!r}."
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

        # Check cut integration scheme
        for key in ("VOLUME_GAUSS_POINTS_BY", "BOUNDARY_GAUSS_POINTS_BY"):
            scheme = params.get(key)
            if scheme is None:
                continue
            if scheme not in ("Tessellation", "DirectDivergence", "MomentFitting"):
                issues.append(
                    f"{key} must be one of 'Tessellation', "
                    f"'DirectDivergence', 'MomentFitting', got {scheme!r}.  "
                    f"It belongs in XFEM GENERAL, not in any XFLUID DYNAMIC "
                    f"section."
                )
            elif scheme == "MomentFitting":
                issues.append(
                    f"{key}: 'MomentFitting' terminates 4C with SIGSEGV "
                    f"inside Core::FE::GaussPointsComposite::num_points and "
                    f"prints no diagnostic.  Use 'Tessellation' or "
                    f"'DirectDivergence'."
                )

        return issues
