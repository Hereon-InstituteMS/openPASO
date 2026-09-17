"""Lubrication generator for 4C.

Covers thin film lubrication problems governed by the Reynolds equation.
Solves for pressure distribution in a thin fluid film between two surfaces.
Applications include journal bearings, squeeze films, slider bearings,
elastohydrodynamic lubrication (EHL), and MEMS devices.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class LubricationGenerator(BaseGenerator):
    """Generator for thin film lubrication (Reynolds equation) problems in 4C."""

    module_key = "lubrication"
    display_name = "Lubrication (Reynolds Equation)"
    problem_type = "Lubrication"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The lubrication module solves the Reynolds equation for "
                "thin film flows.  The Reynolds equation is a 2-D PDE "
                "derived from the Navier-Stokes equations under the thin "
                "film approximation (gap height << lateral dimensions).  "
                "It governs the pressure distribution in a lubricant film "
                "between two surfaces.  The PROBLEM TYPE is 'Lubrication'.  "
                "The dynamics section is 'LUBRICATION DYNAMIC'.  Elements "
                "use the LUBRICATION element type (2-D QUAD4/TRI3 surface "
                "elements forming the film, embedded in 3-D space -- "
                "PROBLEM SIZE/DIM stays 3).  For a stand-alone Reynolds "
                "run set LUBRICATION DYNAMIC/PURE_LUB: true and prescribe "
                "the gap height with HEIGHTFEILD (note 4C's spelling) plus "
                "HFUNCNO, and the surface sliding velocity with "
                "VELOCITYFIELD plus VELFUNCNO; both take 'function' and a "
                "FUNCT id.  Coupling to structural deformation "
                "(elastohydrodynamic lubrication, EHL) is the other option "
                "and is a different PROBLEMTYPE.  Materials use "
                "MAT_lubrication, which carries only DENSITY and "
                "LUBRICATIONLAWID; the viscosity lives in the separate "
                "lubrication-law material that LUBRICATIONLAWID points to."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "LUBRICATION DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_lubrication": {
                    "description": (
                        "Lubricant material for the Reynolds equation.  It "
                        "holds ONLY the density and a pointer to a "
                        "lubrication-law material -- there is no viscosity "
                        "parameter on MAT_lubrication itself."
                    ),
                    "parameters": {
                        "LUBRICATIONLAWID": {
                            "description": (
                                "MAT id of the lubrication-law material "
                                "that supplies the viscosity (one of "
                                "MAT_lubrication_law_constant, "
                                "MAT_lubrication_law_barus, "
                                "MAT_lubrication_law_roeland)"
                            ),
                            "range": "existing MAT id",
                        },
                        "DENSITY": {
                            "description": "Lubricant density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_lubrication_law_constant": {
                    "description": (
                        "Constant-viscosity lubrication law.  Referenced by "
                        "MAT_lubrication/LUBRICATIONLAWID."
                    ),
                    "parameters": {
                        "VISCOSITY": {
                            "description": (
                                "Dynamic viscosity of the lubricant [Pa s]"
                            ),
                            "range": "> 0",
                        },
                    },
                },
                "MAT_lubrication_law_barus": {
                    "description": (
                        "Barus piezoviscous law, mu = mu_0 * exp(alpha*p)."
                    ),
                    "parameters": {
                        "ABSViscosity": {
                            "description": "Reference viscosity mu_0 [Pa s]",
                            "range": "> 0",
                        },
                        "PreVisCoeff": {
                            "description": (
                                "Pressure-viscosity coefficient alpha [1/Pa]"
                            ),
                            "range": ">= 0",
                        },
                    },
                },
                "MAT_lubrication_law_roeland": {
                    "description": "Roelands piezoviscous law.",
                    "parameters": {
                        "ABSViscosity": {
                            "description": "Reference viscosity mu_0 [Pa s]",
                            "range": "> 0",
                        },
                        "PreVisCoeff": {
                            "description": (
                                "Pressure-viscosity coefficient alpha [1/Pa]"
                            ),
                            "range": ">= 0",
                        },
                        "RefPress": {
                            "description": "Roelands reference pressure [Pa]",
                            "range": "> 0",
                        },
                        "RefVisc": {
                            "description": "Roelands reference viscosity [Pa s]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "The Reynolds equation leads to a well-conditioned "
                        "system that is efficiently solved by direct "
                        "solvers.  For large 2-D meshes, iterative solvers "
                        "with AMG can be used."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": "Time step size for transient lubrication.",
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
                "CONVTOL": (
                    "Tolerance of the Newton loop on the pressure."
                ),
                "ITEMAX": "Maximum Newton iterations per time step.",
                "LINEAR_SOLVER": "Id of the SOLVER <n> block to use.",
                "PURE_LUB": (
                    "true for a stand-alone Reynolds run (height and "
                    "velocity come from FUNCTs); false when the height "
                    "and velocity come from an EHL coupling.  There is "
                    "NO SOLVERTYPE key in LUBRICATION DYNAMIC -- the "
                    "Reynolds problem is always solved with the implicit "
                    "Newton loop."
                ),
            },
            "film_height": {
                "prescribed": (
                    "Film height is prescribed with HEIGHTFEILD: "
                    "\"function\" plus HFUNCNO: <FUNCT id> (HEIGHTFEILD "
                    "is 4C's own spelling; choices are EHL|function|zero, "
                    "default zero).  Suitable for slider bearings with "
                    "known geometry."
                ),
                "surface_velocity": (
                    "Surface sliding velocity is prescribed the same way, "
                    "with VELOCITYFIELD: \"function\" plus VELFUNCNO: "
                    "<FUNCT id> (choices EHL|function|zero, default zero). "
                    " There is no SURFACE_VELOCITY key."
                ),
                "function_shape": (
                    "Both FUNCTs are vector valued: give COMPONENT 0/1/2 "
                    "entries, as in 4C's own tests/input_files/"
                    "lubrication_sb_2d.4C.yaml."
                ),
                "coupled": (
                    "For EHL, the film height comes from structural "
                    "deformation (HEIGHTFEILD: \"EHL\", PURE_LUB false). "
                    "This requires coupling with a structural field (not "
                    "covered in stand-alone lubrication)."
                ),
            },
            "pitfalls": [
                (
                    "[Numerical] Reynolds equation is valid "
                    "ONLY for thin films (gap << lateral "
                    "dimension), and 4C does NOT check that "
                    "for you. Signal: none — a gap comparable "
                    "to the lateral extent produces no "
                    "warning, converges in the same number of "
                    "Newton steps as a valid case, and "
                    "returns a finite, plausible-looking "
                    "pressure that simply follows the "
                    "lubrication scaling p ~ mu*U*L/h^2 out "
                    "of the regime where it means anything. "
                    "Check h/L yourself, and switch to "
                    "PROBLEMTYPE: Fluid + the FLUID element for "
                    "thicker gaps -- not FLUID3, which 4C "
                    "rejects outright with \"Fluid element "
                    "types FLUID2 and FLUID3 are no longer in "
                    "use. Switch to FLUID.\" from "
                    "fluid_ele/4C_fluid_ele.cpp. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[API] Lubrication elements are 2D "
                    "SURFACE elements (QUAD4, TRI3) in the "
                    "film plane — NOT 3D volume elements. "
                    "Signal: a 3D cell type is rejected by "
                    "the generic mesh reader, not by any "
                    "lubrication code — \"Element "
                    "'LUBRICATION' does not seem to know cell "
                    "type 'hex8'.\" from "
                    "core/fem/src/general/element/"
                    "4C_fem_general_element_definition.cpp, "
                    "before the Reynolds solver exists. "
                    "There is no 4C_lubrication_factory.cpp "
                    "and no 'unsupported element type' "
                    "message. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] Film height MUST be specified, "
                    "via HEIGHTFEILD + HFUNCNO in LUBRICATION "
                    "DYNAMIC or by coupling to structural "
                    "deformation. Signal: omitting it aborts "
                    "with 'Function with index -1 (i.e. input "
                    "FUNCT-1) not available.' from "
                    "core/utils/src/functions/"
                    "4C_utils_function_manager.hpp, raised "
                    "from Lubrication::TimIntImpl::"
                    "set_height_field_pure_lub — the message "
                    "is about the FUNCT id, not the film. A "
                    "height of zero is NOT a singularity and "
                    "gives no NaN: h enters the operator "
                    "multiplicatively (h^3 grad p) and never "
                    "divides, so h -> 0 zeroes the pressure "
                    "instead of blowing it up. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Cavitation (sub-ambient "
                    "pressure) requires SPECIAL treatment, "
                    "and 4C already has the knob: "
                    "LUBRICATION DYNAMIC/PENALTY_CAVITATION, "
                    "whose default is 0, i.e. OFF. Signal: "
                    "with the penalty off, a diverging gap "
                    "returns a large negative pressure and no "
                    "diagnostic of any kind; setting the "
                    "penalty drives it back to ambient. Do "
                    "not go looking for an Elrod-Adams "
                    "option — the string 'elrod' appears "
                    "nowhere in 4C. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] Reynolds equation BCs are "
                    "pressure DIRICHLET conditions. At "
                    "least one boundary must have a "
                    "prescribed pressure for well-posedness. "
                    "Signal: none — this is a silent-wrong "
                    "failure. A deck with no pressure "
                    "Dirichlet still runs every time step, "
                    "the direct solver reports no 'zero "
                    "pivot' and nothing 'singular', and you "
                    "get back an order-of-magnitude wrong "
                    "pressure. Pin pressure at one boundary "
                    "(typically outflow) and verify it is "
                    "there yourself. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] Lubrication units must be "
                    "consistent. Reynolds uses viscosity "
                    "[Pa s], film height [m], surface "
                    "velocity [m/s]. Signal: none — 4C "
                    "prints no unit or scaling diagnostic. "
                    "Pressure is exactly linear in viscosity "
                    "and in surface velocity, so a "
                    "poise-vs-Pa.s mix-up is exactly 10x. "
                    "Film height is the trap: pressure goes "
                    "as 1/h^2, not 1/h^3 — the h^3 sits "
                    "inside the divergence and one power is "
                    "spent on the second gradient — so an "
                    "mm-vs-m slip moves the pressure by the "
                    "SQUARE of the length factor, not the "
                    "cube. Verify with a Couette-flow sanity "
                    "check. (Corrected by execution "
                    "2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "slider_bearing_2d",
                    "description": (
                        "A 2-D slider bearing with a linearly varying "
                        "film height.  An analytic solution exists for "
                        "the pressure distribution.  Tests the Reynolds "
                        "equation solver, prescribed film height, and "
                        "pressure boundary conditions."
                    ),
                    "template_variant": "slider_bearing_2d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "slider_bearing_2d",
                "description": (
                    "2-D slider bearing with prescribed linear film "
                    "height.  QUAD4 lubrication elements, constant "
                    "viscosity lubricant, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "slider_bearing_2d") -> str:
        templates = {
            "slider_bearing_2d": self._template_slider_bearing_2d,
        }
        if variant == "default":
            variant = "slider_bearing_2d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_slider_bearing_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Slider Bearing (Lubrication / Reynolds Equation)
            #
            # A slider bearing with a linearly varying film height.
            # The lower surface moves at constant velocity, the upper
            # surface is stationary.  Pressure builds up due to the
            # converging gap.
            #
            # Mesh: exodus file with:
            #   element_block 1 = lubrication film, a QUAD4 SURFACE in 3-D
            #                     space (the film is a 2-D manifold; the
            #                     discretisation itself stays 3-D)
            #   node_set 1 = inlet boundary (pressure Dirichlet)
            #   node_set 2 = outlet boundary (pressure Dirichlet)
            # ---------------------------------------------------------------
            TITLE:
              - "2-D slider bearing (Reynolds equation) -- generated template"
            PROBLEM SIZE:
              # Keep DIM 3: the film is a QUAD4 surface embedded in 3-D.
              # DIM 2 aborts in the mesh reader on any film mesh with a
              # non-zero third coordinate.
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Lubrication"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>

            # == Lubrication dynamics ==========================================
            # There is NO SOLVERTYPE key here -- the Reynolds problem always
            # uses the implicit Newton loop (CONVTOL / ITEMAX control it).
            LUBRICATION DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              CONVTOL: <newton_tolerance>
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              # Stand-alone Reynolds run: height and velocity from FUNCTs.
              PURE_LUB: true
              VELOCITYFIELD: "function"
              VELFUNCNO: 1
              HEIGHTFEILD: "function"   # 4C's own spelling, not HEIGHTFIELD
              HFUNCNO: 2
              # Cavitation penalty; default 0 means OFF.
              PENALTY_CAVITATION: <cavitation_penalty>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "lubrication_solver"

            # == Materials =====================================================
            # MAT_lubrication holds only DENSITY + LUBRICATIONLAWID.  The
            # viscosity lives in the lubrication-law material it points to.
            MATERIALS:
              - MAT: 1
                MAT_lubrication:
                  LUBRICATIONLAWID: 2
                  DENSITY: <lubricant_density>
              - MAT: 2
                MAT_lubrication_law_constant:
                  VISCOSITY: <lubricant_dynamic_viscosity>

            # Surface sliding velocity (vector valued, one COMPONENT per dim)
            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<surface_velocity_x>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

            # Film height (linear: h = h_in - (h_in - h_out) * x / L)
            FUNCT2:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<film_height_expression>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

            # == Boundary Conditions ===========================================

            # Pressure Dirichlet at inlet and outlet.  There is no
            # lubrication-specific Dirichlet section: the lubrication field
            # uses the generic DESIGN ... DIRICH CONDITIONS with NUMDOF 1
            # (the single pressure dof).
            DESIGN LINE DIRICH CONDITIONS:
              - E: <inlet_boundary_id>
                ENTITY_TYPE: node_set_id
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<inlet_pressure>]
                FUNCT: [0]
              - E: <outlet_boundary_id>
                ENTITY_TYPE: node_set_id
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<outlet_pressure>]
                FUNCT: [0]

            # == Geometry ======================================================
            LUBRICATION GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  LUBRICATION:
                    QUAD4:
                      MAT: 1

            RESULT DESCRIPTION:
              - LUBRICATION:
                  DIS: "lubrication"
                  NODE: <result_node_id>
                  QUANTITY: "pre"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check viscosity.  It belongs to the lubrication-law material
        # (MAT_lubrication_law_constant/VISCOSITY), NOT to MAT_lubrication.
        if params.get("DYNVISCOSITY") is not None:
            issues.append(
                "DYNVISCOSITY is not a lubrication parameter. Put the "
                "viscosity on MAT_lubrication_law_constant as VISCOSITY "
                "and point MAT_lubrication/LUBRICATIONLAWID at it."
            )
        viscosity = params.get("VISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"VISCOSITY must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VISCOSITY must be a positive number, "
                    f"got {viscosity!r}."
                )

        # Check density
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

        # Surface velocity / film height are prescribed through FUNCT ids,
        # not through a scalar key.  SURFACE_VELOCITY and HEIGHT_FUNCTION
        # do not exist in LUBRICATION DYNAMIC.
        for bogus, real in (
            ("SURFACE_VELOCITY", 'VELOCITYFIELD: "function" + VELFUNCNO'),
            ("HEIGHT_FUNCTION", 'HEIGHTFEILD: "function" + HFUNCNO'),
        ):
            if params.get(bogus) is not None:
                issues.append(
                    f"{bogus} is not a LUBRICATION DYNAMIC key. Use "
                    f"{real} instead."
                )
        if params.get("SOLVERTYPE") is not None:
            issues.append(
                "SOLVERTYPE is not a LUBRICATION DYNAMIC key. The Reynolds "
                "problem always uses the implicit Newton loop; tune it with "
                "CONVTOL / ITEMAX."
            )

        # Check the function ids that DO exist
        for funckey in ("HFUNCNO", "VELFUNCNO"):
            funcno = params.get(funckey)
            if funcno is not None:
                try:
                    n = int(funcno)
                    if n < 1:
                        issues.append(
                            f"{funckey} must reference an existing FUNCT "
                            f"section (>= 1), got {n}."
                        )
                except (TypeError, ValueError):
                    issues.append(
                        f"{funckey} must be an integer FUNCT id, "
                        f"got {funcno!r}."
                    )

        # Check timestep
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
                    f"TIMESTEP must be a positive number, "
                    f"got {timestep!r}."
                )

        return issues
