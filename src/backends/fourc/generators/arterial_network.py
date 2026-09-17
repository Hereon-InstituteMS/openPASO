"""Arterial Network (1-D blood flow) generator for 4C.

Covers 1-D blood flow simulation in arterial networks using the reduced-order
model derived from cross-sectional averaging of the Navier-Stokes equations.
Solves for pressure, flow rate, and vessel cross-sectional area along
arterial segments connected at junctions.  Applications include pulse wave
propagation, arterial hemodynamics, and cardiovascular system modeling.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ArterialNetworkGenerator(BaseGenerator):
    """Generator for 1-D arterial network blood flow problems in 4C."""

    module_key = "arterial_network"
    display_name = "Arterial Network (1-D Blood Flow)"
    problem_type = "ArterialNetwork"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The arterial network module solves 1-D blood flow equations "
                "in networks of compliant arterial segments.  The governing "
                "equations are derived from cross-sectional averaging of the "
                "Navier-Stokes equations, yielding a hyperbolic system for "
                "cross-sectional area A, flow rate Q, and pressure p.  "
                "Arterial segments are connected at junction nodes where "
                "mass conservation and pressure continuity are enforced.  "
                "Terminal (outlet) boundaries use Windkessel (lumped "
                "parameter) models to represent the downstream vasculature.  "
                "The PROBLEM TYPE is 'ArterialNetwork'.  The dynamics "
                "section is 'ARTERIAL DYNAMIC'.  Elements use the ART "
                "element type (1-D LINE2 elements), whose element line "
                "carries MAT, GP, TYPE and DIAM.  Materials use "
                "MAT_CNST_ART (upper case) which defines vessel wall "
                "properties: YOUNG, NUE, TH, DENS, VISCOSITY, PEXT1, "
                "PEXT2.  The reference cross-section is NOT a material "
                "input - it follows from the element's DIAM."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "ARTERIAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "DESIGN NODE 1D ARTERY PRESCRIBED CONDITIONS",
                "DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS",
                "DESIGN NODE 1D ARTERY JUNCTION CONDITIONS",
                "DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_CNST_ART": {
                    "description": (
                        "Constant arterial wall material.  Spelled in "
                        "UPPER CASE - 4C's YAML keys are case-sensitive "
                        "and 'MAT_cnst_art' does not exist.  Defines the "
                        "mechanical properties of the arterial wall for "
                        "the 1-D model.  It carries NO geometry: the "
                        "reference cross-section comes from the element's "
                        "DIAM, not from the material."
                    ),
                    "parameters": {
                        "VISCOSITY": {
                            "description": (
                                "Dynamic blood viscosity [Poise or Pa s]"
                            ),
                            "range": "> 0",
                        },
                        "DENS": {
                            "description": "Blood density [g/cm^3 or kg/m^3]",
                            "range": "> 0",
                        },
                        "YOUNG": {
                            "description": (
                                "Young's modulus of arterial wall "
                                "[dyn/cm^2 or Pa]"
                            ),
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio of artery fiber",
                            "range": "[0, 0.5]",
                        },
                        "TH": {
                            "description": "Wall thickness [cm or m]",
                            "range": "> 0",
                        },
                        "PEXT1": {
                            "description": (
                                "Fixed external pressure at the first node "
                                "of the element.  Required - it has no "
                                "default.  Commonly 0."
                            ),
                        },
                        "PEXT2": {
                            "description": (
                                "Fixed external pressure at the second node "
                                "of the element.  Required - it has no "
                                "default.  Commonly 0."
                            ),
                        },
                    },
                    "optional_parameters": {
                        "VISCOSITYLAW": "CONSTANT (default) or BLOOD.",
                        "BLOOD_VISC_SCALE_DIAM_TO_MICRONS": (
                            "Diameter scaling for the BLOOD viscosity law; "
                            "default 1.0."
                        ),
                        "VARYING_DIAMETERLAW": (
                            "CONSTANT (default) or BY_FUNCTION."
                        ),
                        "VARYING_DIAMETER_FUNCTION": (
                            "Function id for the varying-diameter law; "
                            "default -1."
                        ),
                        "COLLAPSE_THRESHOLD": (
                            "Diameter below which the element counts as "
                            "collapsed; default -1.0."
                        ),
                    },
                },
            },
            "element_parameters": {
                "ART": {
                    "description": (
                        "The 1-D artery element.  Cell type LINE2.  Under "
                        "ARTERY GEOMETRY/ELEMENT_BLOCKS it takes MAT, GP, "
                        "TYPE and DIAM; in the legacy ARTERY ELEMENTS "
                        "string section the same tokens appear inline, "
                        "e.g. '1 ART LINE2 1 2 MAT 1 GP 5 TYPE LinExp "
                        "DIAM 24.0'."
                    ),
                    "parameters": {
                        "MAT": "Material id (points at a MAT_CNST_ART).",
                        "GP": "Number of Gauss points along the element.",
                        "TYPE": (
                            "Artery formulation, e.g. LinExp or "
                            "PressureBased."
                        ),
                        "DIAM": (
                            "Vessel diameter.  THIS is the reference "
                            "geometry: A0 = pi*DIAM^2/4.  There is no "
                            "AREA0 input anywhere in 4C."
                        ),
                    },
                },
            },
            "openpaso_level_inputs": {
                "AREA0": (
                    "openPASO-level convenience input ONLY - it is NOT a 4C "
                    "key and must never be written into a deck.  If you "
                    "prefer to think in reference cross-sectional area, "
                    "give AREA0 and convert it with "
                    "ArterialNetworkGenerator.area0_to_diam(AREA0), which "
                    "returns DIAM = 2*sqrt(AREA0/pi) for the element line."
                ),
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "1-D arterial network systems are small and "
                        "efficiently solved by direct solvers."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": (
                    "Time step size.  Must satisfy CFL condition for "
                    "the pulse wave speed c = sqrt(E*h/(2*rho*A0)).  "
                    "Typical: 1e-4 to 1e-3 s."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": (
                    "Maximum simulation time.  One cardiac cycle is "
                    "approximately 0.8-1.0 s."
                ),
            },
            "boundary_conditions": {
                "inlet": (
                    "DESIGN NODE 1D ARTERY PRESCRIBED CONDITIONS.  Inflow "
                    "at the aortic root: a prescribed flow rate or "
                    "pressure waveform.  Entries take E, VAL and curve "
                    "(the function ids)."
                ),
                "outlet": (
                    "DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS.  4C has "
                    "NO Windkessel outlet for ArterialNetwork - there is "
                    "no R, C or R_d to tune anywhere.  Terminal reflection "
                    "is controlled by the single coefficient in this "
                    "section (VAL: [0] is non-reflecting)."
                ),
                "terminals": (
                    "DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS declares "
                    "which terminal nodes are inlets and which are "
                    "outlets, via terminaltype."
                ),
                "junction": (
                    "DESIGN NODE 1D ARTERY JUNCTION CONDITIONS at "
                    "bifurcation points.  Entries share a ConditionID and "
                    "carry Kr.  Mass conservation and continuity of total "
                    "pressure are enforced automatically."
                ),
            },
            "pitfalls": [
                (
                    "[Numerical] 4C polices the CFL number of the 1D artery "
                    "scheme itself and the limit is tighter than dt <= "
                    "dx/c_max: the quantity it requires below 1 is "
                    "sqrt(3)*|lambda|_max*dt/dx, so use dt < "
                    "dx/(sqrt(3)*c_max). Signal: exceeding it is a HARD ABORT "
                    "in the first element evaluation - 'CFL number at element {} "
                    "is {}', the element id and the CFL value being filled in by "
                    "FOUR_C_THROW - from "
                    "art_net/4C_art_net_artery_ele_calc_lin_exp.cpp:1054 "
                    "- not a NaN and not an energy message. No time step is "
                    "written and no result test runs. (Audit 2026-08-06, "
                    "verified by execution.)"
                ),
                (
                    "[Input] There is no AREA0 input in 4C's ArterialNetwork. "
                    "The reference cross-section is A0 = pi*DIAM^2/4, taken "
                    "from the DIAM token on the ART element line; MAT_CNST_ART "
                    "carries VISCOSITY, DENS, YOUNG, NUE, TH, PEXT1, PEXT2 and "
                    "nothing else. Signal: adding AREA0 to the material fails "
                    "to match section 'MATERIALS'. Getting DIAM wrong is worse "
                    "than linear - the reference area is quadratic in it, and "
                    "the wave speed and junction impedances move together - so "
                    "match DIAM to the vessel. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
                (
                    "[Input] The material is MAT_CNST_ART in UPPER CASE. "
                    "4C's YAML keys are case-sensitive and 'MAT_cnst_art' "
                    "matches nothing. Signal: the lower-case spelling gives "
                    "exactly the same abort as an invented parameter - "
                    "\"Failed to match specification in section 'MATERIALS'\" "
                    "from global_data/4C_global_data_read.cpp followed by "
                    "\"Could not match this input\" from "
                    "core/io/src/4C_io_input_spec_builders.cpp - so a case "
                    "slip and a fabricated key are indistinguishable from "
                    "the message. 4C does NOT suggest the right spelling. "
                    "(Audit 2026-08-07, verified by execution.)"
                ),
                (
                    "[Input] The artery result test is ARTNET, not ARTERY, "
                    "and it accepts exactly three quantities: 'area', "
                    "'pressure' and 'flowrate'. Signal: 'ARTERY' under "
                    "RESULT DESCRIPTION aborts at parse time with \"Could "
                    "not match this input\"; a wrong QUANTITY survives the "
                    "whole simulation and only dies at the very end with "
                    "\"Quantity 'X' not supported in result-test of artery "
                    "transport problems\" from "
                    "art_net/4C_art_net_artery_resulttest.cpp - after every "
                    "time step has been computed and written. Names like "
                    "'one_d_artery_pressure' do not exist. (Audit "
                    "2026-08-07, verified by execution.)"
                ),
                (
                    "[Input] 4C's ArterialNetwork has NO Windkessel outlet: "
                    "there is no R, C or R_d to tune anywhere. art_net "
                    "registers only JUNCTION, PRESCRIBED (flow / pressure / "
                    "velocity / area / characteristicWave, forced or "
                    "absorbing), REFLECTIVE, IN_OUTLET and the porofluid/scatra "
                    "coupling conditions. Signal: writing a DESIGN NODE 1D "
                    "ARTERY WINDKESSEL CONDITIONS' aborts with \"is not a valid "
                    "section name.\" from core/io/src/4C_io_input_file.cpp. "
                    "Terminal reflection is controlled by the single "
                    "coefficient in DESIGN NODE 1D ARTERY REFLECTIVE "
                    "CONDITIONS. (Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Junction DEGREE is not checked. A junction id can "
                    "gather four branch nodes as happily as three; the run "
                    "completes and only the numbers change. Signal: the two "
                    "checks that do exist are in "
                    "art_net/4C_art_net_art_junction.cpp - \"An arterial "
                    "junction is supposed to have at least two nodes!\" and "
                    "\"Junction (N) has all of its nodes defined as outlets\" (or "
                    "inlets). 'inconsistent junction connectivity' does not "
                    "exist in 4C. Verify each junction degree yourself. (Audit "
                    "2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Blood is typically Newtonian with VISCOSITY ~ "
                    "0.03-0.04 Poise; non-Newtonian shear thinning is not "
                    "captured by the 1D model. Signal: viscosity does NOT "
                    "change the pulse wave speed at all - c = "
                    "sqrt(sqrt(pi)*E*th/(1-nu^2)*sqrt(A)/(2*A0*rho)) contains "
                    "no viscosity, and 4C's CFL diagnostic is identical to the "
                    "last digit for blood and for water. What viscosity changes "
                    "is the friction, hence the flow. Set it from blood, but "
                    "look for its effect in the flow, not the wave speed. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Numerical] The 1D model assumes axisymmetric flow and a "
                    "smoothly varying reference area. A local bulge - a "
                    "saccular aneurysm caricature built by widening one "
                    "segment's DIAM - does not give a smooth field with a "
                    "missing stagnation region: it destroys the solution. "
                    "Signal: a small bulge lets the run finish with an "
                    "unphysical flow that has changed sign and grown by orders "
                    "of magnitude; a larger one kills 4C with a bare SIGFPE "
                    "inside ArteryEleCalcLinExp::evaluate_wf_and_wb, shell "
                    "status 136 with NO 'PROC 0 ERROR' banner to grep for. "
                    "Nothing about validity is ever printed. Use 3D FSI for "
                    "aneurysm mechanics. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "single_artery_1d",
                    "description": (
                        "A single arterial segment with prescribed "
                        "inflow at one end and a reflective terminal "
                        "at the other.  Tests pulse wave propagation, "
                        "wave reflection, and pressure-flow "
                        "relationship."
                    ),
                    "template_variant": "single_artery_1d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "single_artery_1d",
                "description": (
                    "Single arterial segment with prescribed inflow "
                    "and a reflective terminal.  MAT_CNST_ART material, "
                    "1-D ART LINE2 elements, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "single_artery_1d") -> str:
        templates = {
            "single_artery_1d": self._template_single_artery_1d,
        }
        if variant == "default":
            variant = "single_artery_1d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_single_artery_1d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 1-D Arterial Network -- Single Artery Segment
            #
            # A single compliant artery with a prescribed flow rate at the
            # inlet and a reflective terminal at the outlet.  The pulse
            # wave propagates along the artery and reflects from the
            # terminal according to its reflection coefficient.
            #
            # NOTE: 4C has NO Windkessel (RCR) outlet for ArterialNetwork.
            # Terminal behaviour is one reflection coefficient, nothing more.
            #
            # NOTE: the reference cross-section is NOT a material input.
            # It follows from the element DIAM as A0 = pi*DIAM^2/4.
            #
            # Mesh: 1-D line mesh with:
            #   element_block 1 = artery segment (ART / LINE2)
            #   node_set 1 = inlet node
            #   node_set 2 = outlet node
            # ---------------------------------------------------------------
            TITLE:
              - "1-D arterial network -- generated template"
            PROBLEM SIZE:
              DIM: 1
            PROBLEM TYPE:
              PROBLEMTYPE: "ArterialNetwork"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>

            # == Arterial dynamics =============================================
            ARTERIAL DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "artery_solver"

            # == Materials =====================================================
            # MAT_CNST_ART is UPPER CASE. It carries no geometry: PEXT1 and
            # PEXT2 are required and have no defaults.
            MATERIALS:
              - MAT: 1
                MAT_CNST_ART:
                  VISCOSITY: <blood_viscosity>
                  DENS: <blood_density>
                  YOUNG: <arterial_wall_Young_modulus>
                  NUE: <arterial_wall_Poisson_ratio>
                  TH: <wall_thickness>
                  PEXT1: <external_pressure_node1>
                  PEXT2: <external_pressure_node2>

            # == Inflow waveform function ======================================
            FUNCT<inflow_function_id>:
              - SYMBOLIC_FUNCTION_OF_TIME: "<inflow_waveform_expression>"

            # == Boundary Conditions ===========================================

            # Inlet: prescribed flow rate / pressure waveform
            DESIGN NODE 1D ARTERY PRESCRIBED CONDITIONS:
              - E: <inlet_node_set_id>
                VAL: [<inlet_prescribed_value>, 0]
                curve: [<inflow_function_id>, null]

            # Outlet: terminal reflection. 4C has no Windkessel here -
            # this single coefficient is the whole terminal model.
            # 0 = non-reflecting.
            DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS:
              - E: <outlet_node_set_id>
                VAL: [<terminal_reflection_coefficient>]
                curve: [null]

            # Declare which terminals are inlets and which are outlets
            DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS:
              - E: <inlet_node_set_id>
                terminaltype: "inlet"
              - E: <outlet_node_set_id>
                terminaltype: "outlet"

            # == Geometry ======================================================
            # Element type is ART (not ARTERY). DIAM sets the reference
            # cross-section: A0 = pi*DIAM^2/4.
            ARTERY GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  ART:
                    LINE2:
                      MAT: 1
                      GP: <num_gauss_points>
                      TYPE: "<artery_formulation>"
                      DIAM: <vessel_diameter>

            # Result test discretisation is ARTNET. QUANTITY is one of
            # exactly: "area", "pressure", "flowrate".
            RESULT DESCRIPTION:
              - ARTNET:
                  DIS: "artery"
                  NODE: <result_node_id>
                  QUANTITY: "pressure"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
              - ARTNET:
                  DIS: "artery"
                  NODE: <result_node_id>
                  QUANTITY: "flowrate"
                  VALUE: <expected_flow_rate>
                  TOLERANCE: <result_tolerance>
        """)

    # -- openPASO-level helpers -----------------------------------------------

    @staticmethod
    def area0_to_diam(area0: float) -> float:
        """Convert a reference cross-sectional area to the element DIAM.

        4C has no AREA0 input.  The reference cross-section of an ART
        element follows from its DIAM as A0 = pi*DIAM^2/4, so the
        inverse is DIAM = 2*sqrt(A0/pi).  Use this when you prefer to
        specify area; write the RESULT into the element's DIAM token,
        never an AREA0 key into the deck.
        """
        import math

        a = float(area0)
        if a <= 0:
            raise ValueError(
                f"AREA0 must be > 0 to convert to DIAM, got {a}."
            )
        return 2.0 * math.sqrt(a / math.pi)

    @staticmethod
    def diam_to_area0(diam: float) -> float:
        """Reference cross-section of an ART element: A0 = pi*DIAM^2/4."""
        import math

        d = float(diam)
        if d <= 0:
            raise ValueError(f"DIAM must be > 0, got {d}.")
        return math.pi * d * d / 4.0

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(
                        f"Arterial wall YOUNG must be > 0, got {e}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check wall thickness
        th = params.get("TH")
        if th is not None:
            try:
                t = float(th)
                if t <= 0:
                    issues.append(f"Wall thickness TH must be > 0, got {t}.")
            except (TypeError, ValueError):
                issues.append(
                    f"TH must be a positive number, got {th!r}."
                )

        # Check vessel diameter -- this is 4C's real reference geometry,
        # and it lives on the ART element line, not in the material.
        diam = params.get("DIAM")
        if diam is not None:
            try:
                dm = float(diam)
                if dm <= 0:
                    issues.append(f"Vessel DIAM must be > 0, got {dm}.")
            except (TypeError, ValueError):
                issues.append(
                    f"DIAM must be a positive number, got {diam!r}."
                )

        # AREA0 is an openPASO-level convenience input, NOT a 4C key.  It is
        # accepted here and converted, but it must never reach the deck.
        area0 = params.get("AREA0")
        if area0 is not None:
            try:
                a = float(area0)
                if a <= 0:
                    issues.append(
                        f"AREA0 (reference area) must be > 0, got {a}."
                    )
                elif diam is not None:
                    try:
                        implied = self.area0_to_diam(a)
                        if abs(implied - float(diam)) > 1e-6 * max(
                            implied, 1.0
                        ):
                            issues.append(
                                f"AREA0 ({a}) implies DIAM "
                                f"{implied:.6g} (DIAM = 2*sqrt(A0/pi)), "
                                f"but DIAM was given as {float(diam):.6g}. "
                                "4C reads only DIAM; drop AREA0 or make "
                                "them consistent."
                            )
                    except (TypeError, ValueError):
                        pass
            except (TypeError, ValueError):
                issues.append(
                    f"AREA0 must be a positive number, got {area0!r}."
                )

        # Check blood density
        dens = params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(f"Blood DENS must be > 0, got {d}.")
            except (TypeError, ValueError):
                issues.append(
                    f"DENS must be a positive number, got {dens!r}."
                )

        # Check viscosity
        visc = params.get("VISCOSITY")
        if visc is not None:
            try:
                mu = float(visc)
                if mu <= 0:
                    issues.append(
                        f"Blood VISCOSITY must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VISCOSITY must be a positive number, got {visc!r}."
                )

        # 4C's ArterialNetwork has NO Windkessel outlet.  Reject these
        # loudly rather than silently accepting them into a deck that
        # would abort with "is not a valid section name."
        for wk_param in ("R_PROXIMAL", "R_DISTAL", "C", "P_VENOUS"):
            if params.get(wk_param) is not None:
                issues.append(
                    f"{wk_param} is a Windkessel parameter, and 4C's "
                    "ArterialNetwork has no Windkessel outlet - there is "
                    "no R, C or R_d anywhere. Terminal behaviour is the "
                    "single coefficient in DESIGN NODE 1D ARTERY "
                    "REFLECTIVE CONDITIONS. Writing 'DESIGN POINT "
                    "WINDKESSEL CONDITIONS' aborts with \"is not a valid "
                    "section name.\""
                )

        # Reflection coefficient
        refl = params.get("REFLECTION_COEFFICIENT")
        if refl is not None:
            try:
                r = float(refl)
                if not -1.0 <= r <= 1.0:
                    issues.append(
                        "Terminal reflection coefficient should lie in "
                        f"[-1, 1] (0 = non-reflecting), got {r}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    "REFLECTION_COEFFICIENT must be a number, "
                    f"got {refl!r}."
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
