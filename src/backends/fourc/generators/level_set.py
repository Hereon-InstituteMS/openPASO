"""Level-set interface tracking generator for 4C.

Covers level-set methods for tracking interfaces in multi-phase or
free-surface problems.  The level-set field phi is advected by a prescribed
or computed velocity field.  Reinitialization (signed distance function)
maintains phi as a distance function for accurate interface location.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class LevelSetGenerator(BaseGenerator):
    """Generator for level-set interface tracking problems in 4C."""

    module_key = "level_set"
    display_name = "Level-Set Interface Tracking"
    problem_type = "Level_Set"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The level-set module tracks an interface as the zero "
                "contour of a scalar field phi.  phi > 0 represents one "
                "phase, phi < 0 the other.  The level-set field is "
                "advected by the advection equation "
                "d(phi)/dt + u . grad(phi) = 0 using the scalar transport "
                "framework.  Reinitialization periodically restores phi to "
                "a signed distance function to prevent numerical "
                "degradation.  The PROBLEM TYPE is 'Level_Set'.  The module "
                "uses SCALAR TRANSPORT DYNAMIC for the advection equation "
                "and LEVEL-SET CONTROL for reinitialization and other "
                "level-set-specific settings."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "LEVEL-SET CONTROL",
                "LEVEL-SET CONTROL/REINITIALIZATION",
                "SCALAR TRANSPORT DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "IO/RUNTIME VTK OUTPUT",
            ],
            "materials": {
                "MAT_scatra": {
                    "description": (
                        "Scalar transport material for the level-set field.  "
                        "DIFFUSIVITY is typically set to 0 (pure advection) "
                        "or a very small value for numerical regularisation."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Diffusion coefficient for the level-set "
                                "field.  0 for pure advection (standard); "
                                "small positive value for regularisation."
                            ),
                            "range": ">= 0 (typically 0)",
                        },
                    },
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "Level-set problems are typically moderate in size.  "
                        "Direct solvers work well."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": (
                    "Time step size.  Must satisfy CFL condition: "
                    "dt <= h / |u|_max where h is the mesh size."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "level_set_control": {
                "REINITIALIZATION": (
                    "Reinitialization method in LEVEL-SET CONTROL/"
                    "REINITIALIZATION.  Options: "
                    "'Signed_Distance_Function' (geometric, exact), "
                    "'Elliptic' (PDE-based, smoother), "
                    "'none' (no reinitialization)."
                ),
                "REINIT_INITIAL": (
                    "Set to true to reinitialise the level-set field at "
                    "the start of the simulation.  Useful when the initial "
                    "condition is not a signed distance function."
                ),
            },
            "velocity_field": {
                "VELOCITYFIELD": (
                    "Velocity field for advection.  Options: "
                    "'function' -- prescribed via VELFUNCNO referencing a "
                    "FUNCT section.  'Navier_Stokes' -- coupled with a "
                    "fluid solve."
                ),
                "VELFUNCNO": (
                    "Function number for the prescribed velocity field "
                    "(used when VELOCITYFIELD: 'function')."
                ),
            },
            "pitfalls": [
                (
                    '[Input] DIFFUSIVITY in the MAT_scatra material should be 0 for '
                    'a standard level-set advection: any diffusion smears the '
                    'interface and changes the physics, not just the numerics. '
                    'Signal: 4C never warns about this. A run with and a run '
                    'without diffusion both complete normally and both reach their '
                    'result test; the only difference is that the diffused field is '
                    'flatter at the interface, so the discrepancy shows up solely '
                    'against a reference solution. Note that some upstream '
                    'level-set decks are convection-DIFFUSION test cases whose '
                    'non-zero DIFFUSIVITY is intentional, so do not copy it '
                    'blindly. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Numerical] Reinitialisation is essential for long runs, or '
                    'the level-set field loses its signed-distance property. The '
                    'section is LEVEL-SET CONTROL with the sub-section LEVEL-SET '
                    'CONTROL/REINITIALIZATION, holding REINITIALIZATION, '
                    'REINIT_INITIAL, REINITINTERVAL, REINITBAND and '
                    'REINITBANDWIDTH. Signal: writing LEVELSET CONTROL without the '
                    'hyphen is rejected as not a valid section name, and the enum '
                    'is case-sensitive: the lower-case spelling '
                    'signed_distance_function fails with '
                    "'possible values: "
                    "EllipticEq|None|Signed_Distance_Function|Sussman'. Use "
                    'Signed_Distance_Function or EllipticEq exactly as spelled. '
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] VELOCITYFIELD in SCALAR TRANSPORT DYNAMIC must be '
                    "'function' with VELFUNCNO pointing at the velocity FUNCT. For "
                    'PROBLEMTYPE Level_Set that is the ONLY accepted value, and '
                    'omitting it does not quietly default to a stationary '
                    'interface. Signal: levelset_dyn checks the value before '
                    "anything is built and throws 'Other velocity fields than a "
                    'field given by a function not yet supported for level-set '
                    "problems' from 4C_levelset_dyn.cpp; 'Navier_Stokes' hits the "
                    'same throw. The run never reaches a time step, so there is no '
                    'time-independent field to notice. (Audit 2026-06-02; corrected '
                    'by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Stabilisation matters for advection-dominated '
                    'level-set transport; the default STABTYPE: SUPG with '
                    'DEFINITION_TAU: Taylor_Hughes_Zarins is what the upstream '
                    'level-set decks rely on. Signal: the failure mode of STABTYPE: '
                    'no_stabilization is NOT visible ringing. The run completes, '
                    'produces no NaN, and returns values that differ from the '
                    'stabilised reference by well under the several percent usually '
                    'quoted for oscillations, so it can only be caught by comparing '
                    'against a reference rather than by eyeballing the field. '
                    '(Audit 2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] The initial level-set field should be a signed '
                    'distance function. If it is not, set REINIT_INITIAL: true in '
                    'LEVEL-SET CONTROL/REINITIALIZATION to correct it before the '
                    'first time step. Signal: without it the raw initial field is '
                    'used exactly as written, so a field whose gradient magnitude '
                    'is not 1 produces nodal values scaled by that same wrong '
                    'gradient, and every interface-position result is off by the '
                    'corresponding factor. 4C prints no warning about the gradient '
                    'or the missing signed-distance property. (Audit 2026-06-02; '
                    'corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Keep the time step small relative to element size '
                    "over velocity for accuracy, but not for stability: 4C's scalar "
                    'transport is implicit, so there is no upwind stability bound '
                    'to break and no O(1) overshoot to look for. Signal: taking the '
                    'same end time in far fewer, larger steps completes normally, '
                    'produces no NaN, and prints no CFL or Courant warning of any '
                    'kind, while every pinned value ends up BELOW the reference. '
                    'The large step damps the profile rather than making it '
                    'oscillate, which is why only a step-size study or a reference '
                    'solution exposes it. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
            ],
            "typical_experiments": [
                {
                    "name": "zalesak_disc",
                    "description": (
                        "Zalesak's slotted disc rotation benchmark.  A "
                        "notched disc is advected by a solid-body rotation "
                        "velocity field.  After one full revolution the disc "
                        "should return to its original position.  Tests "
                        "advection accuracy and reinitialization quality."
                    ),
                    "template_variant": "advection_2d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "advection_2d",
                "description": (
                    "2-D level-set advection with prescribed velocity field "
                    "and geometric reinitialization.  Uses MAT_scatra with "
                    "DIFFUSIVITY: 0, SUPG stabilisation, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "advection_2d") -> str:
        templates = {
            "advection_2d": self._template_advection_2d,
        }
        if variant == "default":
            variant = "advection_2d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_advection_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Level-Set Advection with Reinitialization
            #
            # A level-set field is advected by a prescribed velocity field.
            # Periodic reinitialization restores the signed distance property.
            #
            # Mesh: exodus file with:
            #   element_block 1 = computational domain (QUAD4)
            # ---------------------------------------------------------------
            TITLE:
              - "2-D level-set advection -- generated template"
            PROBLEM TYPE:
              PROBLEMTYPE: "Level_Set"

            # == Level-Set Control =============================================
            LEVEL-SET CONTROL:
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              RESTARTEVERY: <restart_interval>
            LEVEL-SET CONTROL/REINITIALIZATION:
              REINITIALIZATION: "<reinitialization_method>"
              REINIT_INITIAL: <reinitialize_at_start>

            # == Scalar Transport (level-set advection) ========================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              MAXTIME: <end_time>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              RESTARTEVERY: <restart_interval>
              MATID: 1
              VELOCITYFIELD: "function"
              VELFUNCNO: <velocity_function_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_levelset_function_id>
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              DEFINITION_TAU: "<stabilization_tau_definition>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "levelset_solver"

            # == Materials =====================================================
            MATERIALS:
              - MAT: 1
                MAT_scatra:
                  DIFFUSIVITY: <levelset_diffusivity>

            # == Velocity field function =======================================
            FUNCT<velocity_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<velocity_x_expression>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<velocity_y_expression>"

            # == Initial level-set field =======================================
            FUNCT<initial_levelset_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_levelset_expression>"

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    QUAD4:
                      MAT: 1
                      TYPE: Std

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_levelset_value>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check diffusivity
        diffusivity = params.get("DIFFUSIVITY")
        if diffusivity is not None:
            try:
                d = float(diffusivity)
                if d < 0:
                    issues.append(
                        f"DIFFUSIVITY must be >= 0, got {d}."
                    )
                if d > 0:
                    issues.append(
                        f"DIFFUSIVITY = {d} > 0.  For standard level-set "
                        f"advection, DIFFUSIVITY should be 0.  Non-zero "
                        f"diffusion smears the interface."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DIFFUSIVITY must be a non-negative number, "
                    f"got {diffusivity!r}."
                )

        # Check reinitialization method
        reinit = params.get("REINITIALIZATION")
        if reinit is not None and reinit not in (
            "Signed_Distance_Function", "Elliptic", "none"
        ):
            issues.append(
                f"REINITIALIZATION must be 'Signed_Distance_Function', "
                f"'Elliptic', or 'none', got {reinit!r}."
            )

        # Check velocity field
        velfield = params.get("VELOCITYFIELD")
        if velfield is not None and velfield not in (
            "function", "Navier_Stokes", "zero"
        ):
            issues.append(
                f"VELOCITYFIELD must be 'function', 'Navier_Stokes', or "
                f"'zero', got {velfield!r}."
            )

        # Check TIMESTEP
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    issues.append(
                        f"TIMESTEP must be > 0, got {dt}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        # Check CONVTOL
        convtol = params.get("CONVTOL")
        if convtol is not None:
            try:
                ct = float(convtol)
                if ct <= 0:
                    issues.append(
                        f"CONVTOL must be > 0, got {ct}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CONVTOL must be a positive number, got {convtol!r}."
                )

        return issues
