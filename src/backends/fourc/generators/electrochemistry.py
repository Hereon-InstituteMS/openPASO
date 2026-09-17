"""Electrochemistry generator for 4C.

Covers electrochemical transport problems governed by the Nernst-Planck
equation with electroneutrality constraints.  Solves for ionic
concentrations and electric potential in electrolyte systems.  Used for
battery electrolyte modeling, rotating disk electrodes, and
diffusion-migration problems.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ElectrochemistryGenerator(BaseGenerator):
    """Generator for electrochemistry (Nernst-Planck) problems in 4C."""

    module_key = "electrochemistry"
    display_name = "Electrochemistry (Nernst-Planck / ELCH)"
    problem_type = "Electrochemistry"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The electrochemistry module solves the Nernst-Planck "
                "equation for ionic transport in electrolyte systems.  It "
                "couples diffusion, migration (electric field-driven "
                "transport), and optionally convection of multiple ionic "
                "species.  The electric potential is determined by an "
                "electroneutrality condition.  The PROBLEM TYPE is "
                "'Electrochemistry'.  The module uses SCALAR TRANSPORT "
                "DYNAMIC for the transport equations and ELCH CONTROL for "
                "electrochemistry-specific settings (temperature, "
                "electroneutrality method, diffusion-conduction formulation).  "
                "Materials use MAT_ion for individual ionic species wrapped "
                "in MAT_matlist.  Each ion has a diffusivity and valence."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "SCALAR TRANSPORT DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "ELCH CONTROL",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "FLUID DYNAMIC",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
                "IO/RUNTIME VTK OUTPUT",
            ],
            "materials": {
                "MAT_ion": {
                    "description": (
                        "Single ionic species material.  Defines the "
                        "diffusion coefficient and charge valence of an "
                        "ion in the electrolyte."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Diffusion coefficient D_i of the ionic "
                                "species [m^2/s]"
                            ),
                            "range": "> 0",
                        },
                        "VALENCE": {
                            "description": (
                                "Charge number z_i of the ionic species "
                                "(positive for cations, negative for anions)"
                            ),
                            "range": "integer, != 0",
                        },
                    },
                },
                "MAT_matlist": {
                    "description": (
                        "Material list that groups multiple MAT_ion species "
                        "into a single material for the scalar transport "
                        "field.  The number of species determines the "
                        "number of transported scalars."
                    ),
                    "parameters": {
                        "LOCAL": {
                            "description": "Local material flag (typically false)",
                            "range": "true/false",
                        },
                        "NUMMAT": {
                            "description": "Number of ionic species in the list",
                            "range": ">= 2",
                        },
                        "MATIDS": {
                            "description": (
                                "List of MAT_ion material IDs for each species"
                            ),
                            "range": "valid MAT IDs",
                        },
                    },
                },
                "MAT_electrode": {
                    "description": (
                        "Electrode material for Butler-Volmer kinetics at "
                        "electrode-electrolyte interfaces (S2I coupling).  "
                        "Defines concentration-dependent diffusion, "
                        "conductivity, and open-circuit potential."
                    ),
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "Robust direct solver for electrochemistry.  Works "
                        "well for moderate-size problems."
                    ),
                },
            },
            "time_integration": {
                "SOLVERTYPE": (
                    "'nonlinear' is required for electrochemistry due to "
                    "the nonlinear coupling between concentration and "
                    "potential fields."
                ),
                "TIMESTEP": "Time step size for the transport equation.",
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
                "THETA": (
                    "Time integration parameter for the one-step-theta "
                    "scheme.  theta=0.5 gives Crank-Nicolson (second-order), "
                    "theta=1.0 gives backward Euler (first-order, more stable)."
                ),
            },
            "elch_settings": {
                "TEMPERATURE": (
                    "Thermodynamic temperature in ELCH CONTROL.  In 4C "
                    "units this is often specified as T/F (temperature "
                    "divided by Faraday constant) for non-dimensionalised "
                    "formulations, e.g. 11604.506 for ~1 V."
                ),
                "EQUPOT": (
                    "Electroneutrality method.  Options: "
                    "'ENC' (electroneutrality constraint -- algebraic), "
                    "'divi' (divergence-based closure equation), "
                    "'Laplace' (Laplace equation for potential)."
                ),
                "DIFFCOND_FORMULATION": (
                    "Set true for concentrated solution theory "
                    "(diffusion-conduction formulation).  Set false for "
                    "dilute solution theory (Nernst-Planck)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] EQUPOT in ELCH CONTROL selects how "
                    "the electric potential is closed: 'ENC' "
                    "enforces electroneutrality as an algebraic "
                    "constraint, 'divi' solves an additional "
                    "equation for phi. It is NOT a free dial that "
                    "quietly changes the answer. Where both are "
                    "legal they agree: the upstream pair "
                    "elch_2D_tertiary_twoEqu_ENC_varParams_ndb_2iter "
                    "and its ..._divi_... twin differ in that one "
                    "line alone, assert the SAME reference values "
                    "including the potential, and both pass them — "
                    "and the ENC potential is neither zero nor "
                    "flat. Where the material demands 'divi', 4C "
                    "refuses the alternative outright. Signal: with "
                    "a Newman/diffusion-conduction material, ENC "
                    "aborts with 'Newman material must be combined "
                    "with divi closing equation for electric "
                    "potential!' from src/scatra_ele/"
                    "4C_scatra_ele_calc_service_elch_diffcond.cpp. "
                    "Let the material decide. (Corrected by "
                    "execution 2026-08-06; an earlier version "
                    "claimed an ENC run produces ZERO potential "
                    "drop at an electrode interface while divi "
                    "resolves the double layer — not what happens.)"
                ),
                (
                    "[Input] The material an ELCH element uses "
                    "comes from the ELEMENT LINE's 'MAT <id>', and "
                    "that id must be the MAT_matlist (or "
                    "MAT_elchmat) wrapping all ionic species in "
                    "MATIDS — never an individual MAT_ion. MATID in "
                    "SCALAR TRANSPORT DYNAMIC is a different key "
                    "(4C documents it as the material for automatic "
                    "mesh generation) and is inert for a deck whose "
                    "mesh is given explicitly: pointing it at a "
                    "MAT_ion, or deleting it, changes nothing and "
                    "the run still passes its result tests. Signal: "
                    "repointing an element line at a MAT_ion aborts "
                    "with 'Invalid material type!' from src/"
                    "scatra_ele/4C_scatra_ele_calc_service_elch_NP.cpp "
                    "in check_elch_element_parameter. (Corrected by "
                    "execution 2026-08-06; an earlier version put "
                    "the rule on MATID and quoted 'expected "
                    "matlist, got ion' from a '4C_scatra_factory.cpp'. "
                    "neither the string nor that file exists.)"
                ),
                (
                    "[Input] Number of transported scalars = "
                    "NUMMAT in MAT_matlist (one per ionic "
                    "species) plus one for the electric "
                    "potential, and every condition block must "
                    "declare that total in NUMDOF. Getting it wrong "
                    "is a hard abort in which 4C states the "
                    "expected number for you, so you do not have to "
                    "work it out: a 4-species matlist wants NUMDOF "
                    "5, and a block written with 4 gives '4 DOFs "
                    "given but 5 expected in Point Dirichlet "
                    "boundary condition' from core/fem/src/"
                    "discretization/4C_fem_discretization_utils_dbc.cpp. "
                    "The run never reaches its result tests. "
                    "Signal: that DOF-count line, naming both the "
                    "given and the expected count. (Corrected by "
                    "execution 2026-08-06; an earlier version "
                    "quoted 'INITIALFIELD component count mismatch' "
                    "and offered a silent phi = 0 as the "
                    "alternative. THAT STRING DOES NOT EXIST and "
                    "the failure is not silent.)"
                ),
                (
                    "[Input] Initial conditions for "
                    "concentrations + potential should use "
                    "INITIALFIELD: 'field_by_function' with "
                    "INITFUNCNO / STARTFUNCNO. Each scalar "
                    "component needs its OWN COMPONENT entry "
                    "in the FUNCT, and a short FUNCT is fatal "
                    "rather than quietly zero-filled: the first "
                    "evaluation asks for the component that is not "
                    "there and the run stops. Signal: 'There are N "
                    "expressions but tried to access component N' "
                    "from core/utils/src/functions/"
                    "4C_utils_function.cpp, before any result test. "
                    "(Corrected by execution 2026-08-06; an earlier "
                    "version claimed the missing component is "
                    "silently set to 0 and visible as a "
                    "discontinuous initial concentration plot. "
                    "There is no field to plot — the run aborts.)"
                ),
                (
                    "[Output] CALCFLUX_DOMAIN: 'total' in "
                    "SCALAR TRANSPORT DYNAMIC is what puts species "
                    "fluxes into the VTK output, and its absence is "
                    "an ABSENCE — there is no placeholder to "
                    "notice. Without it the .vtu carries only "
                    "phi_1 ... phi_N; with it, an additional "
                    "flux_domain_phi_1 ... flux_domain_phi_N per "
                    "species. The solution is unchanged either way, "
                    "so this is an output switch, not a physics "
                    "one. Signal: grep the .vtu for a field name "
                    "beginning flux_domain_ — that prefix is the "
                    "literal, in scatra/4C_scatra_resulttest.cpp, "
                    "and the species suffix is appended at run "
                    "time, so there is no full field name to grep "
                    "for in the source. If no such field is there "
                    "the key was not set. Enable it before you need it, "
                    "since recovering the flux means re-running. "
                    "(Corrected by execution 2026-08-06; an earlier "
                    "version said the flux fields 'show not "
                    "computed'. that string appears in neither the "
                    "log nor the .vtu.)"
                ),
                (
                    "[Input] For S2I (scatra-scatra "
                    "interface) coupling with Butler-Volmer "
                    "kinetics you need BOTH the dynamics "
                    "subsection 'SCALAR TRANSPORT DYNAMIC/S2I "
                    "COUPLING' and the interface conditions — but "
                    "the condition sections are named 'DESIGN S2I "
                    "KINETICS <SURF|LINE> CONDITIONS' and 'DESIGN "
                    "S2I MESHTYING <SURF|LINE> CONDITIONS', and "
                    "they come in pairs: every kinetics condition "
                    "needs a matching meshtying one. There is no "
                    "'DESIGN SURF S2I COUPLING CONDITIONS' section "
                    "on this build. Neither omission is survivable, "
                    "so there is no silently inert setup to look "
                    "for. Signal: dropping the kinetics conditions "
                    "gives \"For each 'S2IKinetics' or "
                    "'S2ISCLCoupling' condition a corresponding "
                    "'S2IMeshtying' or 'S2INoEvaluation' condition "
                    "has to be defined!\" from "
                    "src/scatra/4C_scatra_utils.cpp; dropping the "
                    "dynamics subsection gives 'Type of mortar "
                    "meshtying for scatra-scatra interface coupling "
                    "not recognized!' from "
                    "4C_scatra_timint_meshtying_strategy_s2i.cpp; "
                    "the invented section name gives 'is not a "
                    "valid section name.' (Corrected by execution "
                    "2026-08-06; an earlier version gave the wrong "
                    "section name and predicted a setup that "
                    "compiles but produces ~0 electrode current.)"
                ),
                (
                    "[Numerical] Use STABTYPE "
                    "'no_stabilization' for diffusion-dominated "
                    "ELCH; add SUPG only where convection is "
                    "significant (forced electrolyte flow). On the "
                    "diffusion-conduction framework "
                    "(DIFFCOND_FORMULATION: true — the "
                    "concentrated-solution route) this is not "
                    "advice you can ignore: 4C rejects stabilised "
                    "settings outright rather than degrading "
                    "quietly. Where the choice IS free (dilute "
                    "Nernst-Planck) it is not a small perturbation "
                    "either — on a convection-dominated deck the "
                    "two settings give order-unity differences in "
                    "the species concentrations, because the "
                    "stabilisation is doing real work there. "
                    "Signal: SUPG under DIFFCOND_FORMULATION "
                    "aborts with 'No stabilization is necessary for "
                    "solving the ELCH diffusion-conduction "
                    "framework!!' from "
                    "src/scatra/4C_scatra_timint_elch.cpp. "
                    "(Corrected by execution 2026-08-06; an earlier "
                    "version predicted a silent 5-20% damping of "
                    "concentration gradients, which is not what "
                    "either regime does.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "diffusion_migration_3d",
                    "description": (
                        "Diffusion-migration of binary electrolyte in a 3-D "
                        "domain.  Two ionic species with different "
                        "diffusivities and valences.  Tests electroneutrality "
                        "coupling and flux computation."
                    ),
                    "template_variant": "nernst_planck_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "nernst_planck_3d",
                "description": (
                    "3-D Nernst-Planck diffusion-migration problem with "
                    "binary electrolyte (cation + anion).  Uses MAT_ion "
                    "materials in MAT_matlist, ENC electroneutrality, "
                    "UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "nernst_planck_3d") -> str:
        templates = {
            "nernst_planck_3d": self._template_nernst_planck_3d,
        }
        if variant == "default":
            variant = "nernst_planck_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_nernst_planck_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Nernst-Planck Electrochemistry (Binary Electrolyte)
            #
            # Diffusion-migration of two ionic species (cation and anion)
            # in a 3-D domain with electroneutrality constraint.
            #
            # Mesh: exodus file with:
            #   element_block 1 = electrolyte domain (HEX8 or TET4)
            #   node_set 1 = Dirichlet boundary (fixed concentrations)
            #   node_set 2 = opposite boundary
            # ---------------------------------------------------------------
            TITLE:
              - "3-D electrochemistry (Nernst-Planck) -- generated template"
            PROBLEM TYPE:
              PROBLEMTYPE: "Electrochemistry"

            # == Scalar Transport (carries concentration + potential) ===========
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              MAXTIME: <end_time>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              RESTARTEVERY: <restart_interval>
              MATID: <matlist_material_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_condition_function_id>
              CALCFLUX_DOMAIN: "total"
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>
              EXPLPREDICT: <explicit_predictor_flag>

            # == Electrochemistry control ======================================
            ELCH CONTROL:
              TEMPERATURE: <thermodynamic_temperature>
              EQUPOT: "<electroneutrality_method>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "elch_solver"

            # == Materials =====================================================
            MATERIALS:
              # Cation
              - MAT: 1
                MAT_ion:
                  DIFFUSIVITY: <cation_diffusivity>
                  VALENCE: <cation_valence>
              # Anion
              - MAT: 2
                MAT_ion:
                  DIFFUSIVITY: <anion_diffusivity>
                  VALENCE: <anion_valence>
              # Material list wrapping all ionic species
              - MAT: <matlist_material_id>
                MAT_matlist:
                  LOCAL: false
                  NUMMAT: <number_of_species>
                  MATIDS: [1, 2]

            # == Initial condition function ====================================
            # One COMPONENT per scalar: species 1, species 2, potential
            FUNCT<initial_condition_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_concentration_1_expression>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_concentration_2_expression>"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_potential_expression>"

            # == Boundary Conditions ===========================================
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<boundary_concentrations_and_potential>]
                FUNCT: [<time_functions>]

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    HEX8:
                      MAT: <matlist_material_id>
                      TYPE: Std

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_concentration>
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
                if d <= 0:
                    issues.append(
                        f"DIFFUSIVITY must be > 0, got {d}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DIFFUSIVITY must be a positive number, "
                    f"got {diffusivity!r}."
                )

        # Check valence
        valence = params.get("VALENCE")
        if valence is not None:
            try:
                z = int(valence)
                if z == 0:
                    issues.append(
                        "VALENCE must be non-zero (charge number of the ion)."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VALENCE must be a non-zero integer, got {valence!r}."
                )

        # Check EQUPOT
        equpot = params.get("EQUPOT")
        if equpot is not None and equpot not in ("ENC", "divi", "Laplace"):
            issues.append(
                f"EQUPOT must be 'ENC', 'divi', or 'Laplace', "
                f"got {equpot!r}."
            )

        # Check TEMPERATURE
        temperature = params.get("TEMPERATURE")
        if temperature is not None:
            try:
                t = float(temperature)
                if t <= 0:
                    issues.append(
                        f"TEMPERATURE must be > 0 (thermodynamic temperature), "
                        f"got {t}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"TEMPERATURE must be a positive number, "
                    f"got {temperature!r}."
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

        # Check NUMMAT in matlist
        nummat = params.get("NUMMAT")
        if nummat is not None:
            try:
                nm = int(nummat)
                if nm < 2:
                    issues.append(
                        f"NUMMAT in MAT_matlist should be >= 2 (at least "
                        f"two ionic species for electroneutrality), got {nm}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUMMAT must be an integer >= 2, got {nummat!r}."
                )

        return issues
