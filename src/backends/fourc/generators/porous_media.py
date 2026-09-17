"""Generator for porous media flow physics module (pressure-based porofluid).

Covers single-phase and multiphase flow through rigid or deformable porous
media using the porofluid_dynamic section (lowercase!) in 4C.  The complex
nested material hierarchy (MAT_FluidPoroMultiPhase -> MAT_FluidPoroSinglePhase
-> density/viscosity/relative-permeability laws) is fully documented.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class PorousMediaGenerator(BaseGenerator):
    """Generator for porous media flow problems in 4C.

    Supports single-phase and multiphase pressure-based porofluid
    simulations with complex material hierarchies.
    """

    module_key = "porous_media"
    display_name = "Porous Media Flow (Pressure-Based Porofluid)"
    problem_type = "porofluid_pressure_based"

    # -- Knowledge -----------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The porous media module in 4C solves single-phase or multiphase "
                "flow through porous media using a pressure-based formulation.  "
                "CRITICAL: this module uses lowercase section names "
                "(porofluid_dynamic, NOT POROFLUID DYNAMIC).  The PROBLEM TYPE is "
                "'porofluid_pressure_based' (for rigid skeleton) or "
                "'porofluid_pressure_based_elasticity' (for deformable skeleton "
                "coupled with structural mechanics).  The material definition "
                "follows a complex nested hierarchy: MAT_FluidPoroMultiPhase -> "
                "MAT_FluidPoroSinglePhase -> density law + viscosity law + "
                "relative permeability law + DoF type.  Each sub-material is "
                "defined as a separate MATERIALS entry with its own MAT ID, "
                "referenced by ID from the parent material."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "porofluid_dynamic",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "STRUCTURAL DYNAMIC (for porofluid_pressure_based_elasticity)",
                "porofluid_elasticity_dynamic (for coupled problems)",
                "CLONING MATERIAL MAP (for coupled problems)",
                "IO",
                "NODE COORDS (inline mesh)",
                "TRANSPORT ELEMENTS (inline mesh)",
            ],
            "section_naming": {
                "CRITICAL": (
                    "Porous media sections use lowercase names unlike most "
                    "other 4C modules.  Use 'porofluid_dynamic' NOT "
                    "'POROFLUID DYNAMIC'.  Sub-sections also lowercase: "
                    "'porofluid_dynamic/time_integration', etc."
                ),
                "porofluid_dynamic": {
                    "total_simulation_time": "Maximum simulation time",
                    "time_integration/number_of_time_steps": "Number of time steps",
                    "time_integration/time_step_size": "Time step size",
                    "time_integration/theta": "Theta for one-step-theta scheme (1.0 = implicit Euler)",
                    "nonlinear_solver/linear_solver_id": "Reference to SOLVER N section",
                    "initial_condition/type": "'by_function' to set IC via FUNCT section",
                    "initial_condition/function_id": "FUNCT ID for initial condition",
                    "output/porosity": "Whether to output porosity field",
                    "flux_reconstruction/active": "Enable flux reconstruction",
                    "flux_reconstruction/solver_id": "Solver for flux reconstruction",
                },
            },
            "materials": {
                "MAT_FluidPoroMultiPhase (top-level)": {
                    "description": (
                        "Top-level porous media material.  Wraps one or more "
                        "single-phase materials and defines the intrinsic "
                        "permeability of the porous medium."
                    ),
                    "parameters": {
                        "LOCAL": {
                            "description": "Local assembly flag (typically false)",
                            "range": "true / false",
                        },
                        "PERMEABILITY": {
                            "description": "Intrinsic permeability of the porous medium [m^2]",
                            "range": "> 0",
                        },
                        "NUMMAT": {
                            "description": "Number of single-phase sub-materials",
                            "range": ">= 1",
                        },
                        "MATIDS": {
                            "description": "List of MAT IDs for each single-phase material",
                            "range": "e.g. [10] or [10, 20, 30]",
                        },
                        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE": {
                            "description": "Number of fluid phases sharing the pore space",
                            "range": ">= 1",
                        },
                    },
                },
                "MAT_FluidPoroSinglePhase": {
                    "description": (
                        "Single fluid phase within the pore space.  References "
                        "sub-materials for density law, viscosity law, relative "
                        "permeability law, and DoF type."
                    ),
                    "parameters": {
                        "DENSITYLAWID": {
                            "description": "MAT ID of the density law",
                            "range": "References a MAT_PoroDensityLaw* material",
                        },
                        "DENSITY": {
                            "description": "Reference density of the fluid phase",
                            "range": "> 0",
                        },
                        "RELPERMEABILITYLAWID": {
                            "description": "MAT ID of the relative permeability law",
                            "range": "References a MAT_FluidPoroRelPermeabilityLaw*",
                        },
                        "VISCOSITY_LAW_ID": {
                            "description": "MAT ID of the viscosity law",
                            "range": "References a MAT_FluidPoroViscosityLaw*",
                        },
                        "DOFTYPEID": {
                            "description": "MAT ID defining the primary DoF type for this phase",
                            "range": "References a MAT_FluidPoroSinglePhaseDoF*",
                        },
                    },
                },
                "MAT_PoroDensityLawExp": {
                    "description": (
                        "Exponential density law: rho = rho_0 * exp((p - p_0) / K).  "
                        "For nearly incompressible fluids use a large BULKMODULUS."
                    ),
                    "parameters": {
                        "BULKMODULUS": {
                            "description": "Fluid bulk modulus K",
                            "range": "> 0 (large for incompressible, e.g. 100--1e6)",
                        },
                    },
                },
                "MAT_FluidPoroViscosityLawConstant": {
                    "description": "Constant dynamic viscosity.",
                    "parameters": {
                        "VALUE": {
                            "description": "Dynamic viscosity mu [Pa.s]",
                            "range": "> 0 (e.g. 0.001 for water, 0.01 for oil)",
                        },
                    },
                },
                "MAT_FluidPoroRelPermeabilityLawConstant": {
                    "description": (
                        "Constant relative permeability (k_r = VALUE).  "
                        "Use VALUE = 1.0 for single-phase flow."
                    ),
                    "parameters": {
                        "VALUE": {
                            "description": "Relative permeability k_r (dimensionless)",
                            "range": "0 < k_r <= 1 (1.0 for single-phase)",
                        },
                    },
                },
                "MAT_FluidPoroSinglePhaseDofPressure": {
                    "description": (
                        "DoF type: primary variable is absolute pressure.  "
                        "The simplest choice for single-phase flow."
                    ),
                    "parameters": {
                        "PHASELAWID": {
                            "description": "MAT ID of a phase law (optional for single-phase)",
                            "range": "References a MAT_PhaseLaw* material",
                        },
                    },
                },
            },
            "dof_types": {
                "Pressure": (
                    "MAT_FluidPoroSinglePhaseDofPressure -- primary variable is "
                    "absolute pressure p.  Simplest for single-phase flow."
                ),
                "Saturation": (
                    "MAT_FluidPoroSinglePhaseDofSaturation -- primary variable is "
                    "saturation S.  Used in multiphase flow when one phase's "
                    "saturation is solved directly."
                ),
                "DiffPressure": (
                    "MAT_FluidPoroSinglePhaseDofDiffPressure -- primary variable "
                    "is a pressure difference (p_i - p_j).  Used in multiphase "
                    "flow for capillary pressure formulations."
                ),
            },
            "property_laws": {
                "PoroDensityLawExp": "Exponential density law (compressible fluid)",
                "PoroDensityLawConstant": "Constant density (incompressible)",
                "FluidPoroViscosityLawConstant": "Constant dynamic viscosity",
                "FluidPoroRelPermeabilityLawConstant": "Constant relative permeability",
                "FluidPoroRelPermeabilityLawExp": "Exponential relative permeability (S-dependent)",
            },
            "solver": {
                "recommended": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, robust for porous media problems.  "
                        "The porous media system can be stiff due to "
                        "permeability contrasts."
                    ),
                },
            },
            "time_integration": {
                "scheme": (
                    "One-step-theta method.  theta = 1.0 gives implicit Euler "
                    "(unconditionally stable, first-order accurate).  "
                    "theta = 0.5 gives Crank-Nicolson (second-order, may "
                    "oscillate)."
                ),
                "time_step_size": "Size of each time step.",
                "number_of_time_steps": "Total number of time steps.",
                "total_simulation_time": "Maximum simulation time.",
            },
            "mesh_format": {
                "note": (
                    "Most porous media examples in 4C use inline mesh format "
                    "(NODE COORDS + TRANSPORT ELEMENTS).  Exodus meshes can also "
                    "be used via TRANSPORT GEOMETRY with POROFLUID element type."
                ),
            },
            "pitfalls": [
                (
                    "[Input] The POROFLUIDMULTIPHASE element line takes MAT "
                    "and nothing else. 4C's own element spec for it is a "
                    "single required int MAT; there is no TYPE token, and "
                    "'TYPE PoroFluidMultiPhase' appears in none of the 2171 "
                    "upstream decks. Signal: a leftover token on an element "
                    "line is NOT ignored - 4C aborts with \"After parsing, "
                    "the line still contains 'TYPE PoroFluidMultiPhase'. "
                    "Parsed parameters: MAT : 1\" from "
                    "core/io/src/4C_io_input_spec.cpp. The message quotes "
                    "the surplus text back at you, so it is one of the "
                    "easier 4C errors to act on. (Audit 2026-08-07, "
                    "verified by execution: the same deck runs to "
                    "the exit banner \u2014 the literal is 'finished "
                    "normally', with the rank printed into it, so the "
                    "line reads processor 0 finished normally \u2014 "
                    "with the token "
                    "removed.)"
                ),
                (
                    "[Input] CRITICAL: Section names are lowercase in "
                    "porous media.  Use 'porofluid_dynamic' NOT "
                    "'POROFLUID DYNAMIC'.  Using uppercase does NOT "
                    "silently fall back to defaults -- it is a hard abort "
                    "before anything runs. Signal: \"Section 'POROFLUID "
                    "DYNAMIC' is not a valid section name.\" from "
                    "core/io/src/4C_io_input_file.cpp, exit 1 at parse. "
                    "(Verified by execution 2026-08-03. An earlier version "
                    "predicted a silent fallback with uniformly-default "
                    "result fields and quoted an 'unknown section' banner; "
                    "neither is real -- that string is not in the binary "
                    "and the run never starts.)"
                ),
                (
                    "[Input] The material hierarchy is deeply nested: "
                    "MAT_FluidPoroMultiPhase -> MAT_FluidPoroSinglePhase "
                    "-> density law + viscosity law + relative "
                    "permeability law + DoF type.  Each level is a "
                    "separate MATERIALS entry with its own MAT ID.  "
                    "Missing any sub-material causes a crash. Signal: "
                    "\"Material 'MAT <id>' could not be found\" from "
                    "mat/4C_mat_par_bundle.cpp, exit 1 while the "
                    "MATERIALS section is being read.  The stack names "
                    "the sub-factory (e.g. "
                    "Mat::PAR::PoroDensityLaw::create_density_law), so "
                    "the abort surfaces levels below the entry you "
                    "edited. (Verified by execution 2026-08-06; an "
                    "earlier version quoted `referenced material ID X "
                    "not found in MATERIALS`, which is not in the "
                    "binary.)"
                ),
                (
                    "[Numerical] PERMEABILITY must be > 0, but whether "
                    "4C SAYS SO depends on the material.  "
                    "MAT_FluidPoro (Poroelasticity) has an explicit "
                    "guard.  MAT_FluidPoroMultiPhase "
                    "(porofluid_pressure_based) has none. Signal: on "
                    "MAT_FluidPoro, 'zero or negative permeability' "
                    "from mat/4C_mat_fluidporo.cpp, exit 1.  On "
                    "MAT_FluidPoroMultiPhase there is NO 'PROC 0 ERROR' "
                    "banner at all: the nonlinear residual diverges and "
                    "the process dies on SIGFPE -- the MPI runtime names "
                    "the floating point exception, signal 8 -- inside "
                    "EvaluatorMassPressure::evaluate_matrix_and_assemble, "
                    "exit 136 -- a raw SIGFPE naming nothing you wrote. "
                    "(Verified by execution 2026-08-06; the quoted "
                    "`zero pivot detected` does not exist in 4C or in "
                    "Trilinos, and neither does `solver returned "
                    "status: -3`.)"
                ),
                (
                    "[Input] For single-phase flow, set "
                    "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE: 1 AND "
                    "NUMMAT: 1. Even single-phase flow requires the "
                    "full material hierarchy, and NUMMAT has no "
                    "default. Signal: omitting it gives \"Parameter "
                    "'NUMMAT' not found in container.\" from "
                    "global_data/4C_global_data_read.cpp, raised inside "
                    "Global::read_materials -- i.e. while the section "
                    "is read, not at a later consistency check. "
                    "(Verified by execution 2026-08-06; the quoted "
                    "'material list size mismatch' does not exist.)"
                ),
                (
                    "[Input] `initial_condition` is a REQUIRED group of "
                    "porofluid_dynamic and its `type` has no default -- "
                    "you cannot omit it and fall back to a silent zero "
                    "start.  If you want to start from zero, say so: "
                    "type: \"zero\". Signal: omitting the group aborts "
                    "at parse with 'Could not match this input' and a "
                    "match-tree line reading [X] Expected group "
                    "'initial_condition'. That line has no greppable "
                    "core: recursively_print_match_entries in "
                    "core/io/src/4C_io_input_spec_builders.cpp formats "
                    "it from four separate short literals -- the state "
                    "symbol, the state phrase, the spec type and the "
                    "name -- so nothing in it exists as one string in "
                    "the source. No time "
                    "step is ever started. (Verified by execution "
                    "2026-08-06.  An earlier version predicted a silent "
                    "wrong-answer trap -- an unphysical zero start with "
                    "a boundary-layer pressure shock at step 1; the run "
                    "does not get that far.)"
                ),
                (
                    "[Input] For coupled poroelasticity problems, a "
                    "CLONING MATERIAL MAP section is required to map "
                    "the structural material to the porofluid "
                    "material. Signal: 'At least one material pairing "
                    "required in --CLONING MATERIAL MAP.' from "
                    "core/fem/src/general/utils/"
                    "4C_fem_general_utils_createdis.hpp, raised inside "
                    "clone_discretization<PoroelastCloneStrategy>. Note "
                    "the message names the section in the retired "
                    "--SECTION dat spelling. (Verified by execution "
                    "2026-08-06.  An earlier version quoted "
                    "`CloningMaterialMap missing for porofluid -> "
                    "solid` and offered a silent decoupled run as the "
                    "alternative; neither is real -- the second field "
                    "is never created, so nothing runs.)"
                ),
                (
                    "[Input] The PROBLEM TYPE string is "
                    "'porofluid_pressure_based' (lowercase, with "
                    "underscores), not 'Porofluid' or "
                    "'POROFLUID_PRESSURE_BASED'.  The porofluid entries "
                    "are the only lower-case members of the enum. "
                    "Signal: 'Could not match this input' plus "
                    "\"'PROBLEMTYPE' has wrong value, possible "
                    "values: ...\" from "
                    "core/io/src/4C_io_input_spec_builders.cpp, which "
                    "does enumerate every legal string. (Verified by "
                    "execution 2026-08-06; the quoted phrase 'unknown "
                    "PROBLEMTYPE' does not exist in 4C.)"
                ),
                (
                    "[Numerical] BULKMODULUS in MAT_PoroDensityLawExp "
                    "sets fluid compressibility and CHANGES THE ANSWER, "
                    "but it does not destabilise the solve, and it has "
                    "no practical floor: 4C's own porofluid regression "
                    "deck ships a value orders of magnitude below the "
                    "'unphysical' threshold this entry used to assert. "
                    "Signal: raising MAT_PoroDensityLawExp's BULKMODULUS far "
                    "past the claimed 1e10 Pa "
                    "ceiling still completes every time step, never "
                    "reaches the nonlinear iteration limit, prints no "
                    "non-convergence message and needs no more "
                    "iterations than the soft case -- what moves is the "
                    "result, reported as 'Result check failed with {} "
                    "errors out of {} tests' "
                    "(4C_utils_result_test.cpp:181).  Choose it on physics "
                    "(water ~ 2.2e9 Pa, air ~ 1e5 Pa), not on solver "
                    "fear. (Verified by execution 2026-08-06; the "
                    "claimed 'iterative solvers stagnate' above 1e10 "
                    "and 'unphysical' below 1e6 are both false.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "single_phase_pressure_driven",
                    "description": (
                        "Single-phase Darcy flow through a 3D porous block.  "
                        "Pressure boundary conditions on inlet and outlet.  "
                        "Simplest porous media test case."
                    ),
                    "template_variant": "single_phase_3d",
                },
                {
                    "name": "multiphase_infiltration",
                    "description": (
                        "Two-phase flow (water displacing oil) in a porous "
                        "medium.  Uses saturation and pressure DoF types.  "
                        "Demonstrates the full material hierarchy."
                    ),
                },
            ],
        }

    # -- Templates -----------------------------------------------------------

    _TEMPLATES: dict[str, str] = {
        "single_phase_3d": """\
# ----------------------------------------------------------------
# Single-phase pressure-driven flow through a 3D porous block
#
# Simple Darcy flow with:
#   - Constant permeability, viscosity, and density
#   - Pressure BC: p = 1.0 at inlet, p = 0.0 at outlet
#   - Initial condition: p = 0.1 everywhere
#
# Material hierarchy:
#   MAT 1: MAT_FluidPoroMultiPhase (top-level, permeability)
#     -> MAT 10: MAT_FluidPoroSinglePhase (single phase)
#          -> MAT 101: MAT_FluidPoroSinglePhaseDofPressure (DoF = pressure)
#          -> MAT 102: MAT_PhaseLawConstraint (trivial phase law)
#          -> MAT 103: MAT_PoroDensityLawExp (density law)
#          -> MAT 104: MAT_FluidPoroViscosityLawConstant (viscosity)
#          -> MAT 105: MAT_FluidPoroRelPermeabilityLawConstant (k_r = 1)
#
# NOTE: Uses inline mesh (NODE COORDS + TRANSPORT ELEMENTS).
#       For Exodus mesh, replace with TRANSPORT GEOMETRY section.
# ----------------------------------------------------------------
TITLE:
  - "Single-phase Darcy flow through 3D porous block"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "porofluid_pressure_based"
DISCRETISATION:
  NUMSTRUCDIS: 0
  NUMALEDIS: 0
  NUMTHERMDIS: 0

# -- Porofluid dynamics (NOTE: lowercase section name!) ----------
porofluid_dynamic:
  total_simulation_time: 1.0
  time_integration:
    number_of_time_steps: 10
    time_step_size: 0.1
    theta: 1
  nonlinear_solver:
    linear_solver_id: 1
  output:
    porosity: false
  initial_condition:
    type: by_function
    function_id: 1
  flux_reconstruction:
    active: true
    solver_id: 1

IO:
  VERBOSITY: "Minimal"

# -- Solver ------------------------------------------------------
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "porofluid_solver"

# -- Material hierarchy ------------------------------------------
MATERIALS:
  # Top-level: multiphase container (single phase here)
  - MAT: 1
    MAT_FluidPoroMultiPhase:
      LOCAL: false
      PERMEABILITY: 1.0
      NUMMAT: 1
      MATIDS: [10]
      NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE: 1
  # Single fluid phase
  - MAT: 10
    MAT_FluidPoroSinglePhase:
      DENSITYLAWID: 103
      DENSITY: 1.0
      RELPERMEABILITYLAWID: 105
      VISCOSITY_LAW_ID: 104
      DOFTYPEID: 101
  # DoF type: absolute pressure
  - MAT: 101
    MAT_FluidPoroSinglePhaseDofPressure:
      PHASELAWID: 102
  # Phase law: trivial constraint (single-phase)
  - MAT: 102
    MAT_PhaseLawConstraint: {}
  # Density law: exponential (nearly incompressible)
  - MAT: 103
    MAT_PoroDensityLawExp:
      BULKMODULUS: 100.0
  # Viscosity: constant
  - MAT: 104
    MAT_FluidPoroViscosityLawConstant:
      VALUE: 0.01
  # Relative permeability: constant (k_r = 1 for single phase)
  - MAT: 105
    MAT_FluidPoroRelPermeabilityLawConstant:
      VALUE: 1.0

# -- Initial condition: uniform pressure p = 0.1 ----------------
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.1"

# -- Boundary conditions (pressure Dirichlet) --------------------
DESIGN SURF DIRICH CONDITIONS:
  # Inlet face: p = 1.0
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
  # Outlet face: p = 0.0
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]

# -- Inline mesh (simple 2x2x5 hex8 block) ----------------------
# Replace with TRANSPORT GEOMETRY + Exodus for production meshes
NODE COORDS:
  - "NODE 1  COORD 0.0 0.0 0.0"
  - "NODE 2  COORD 1.0 0.0 0.0"
  - "NODE 3  COORD 1.0 1.0 0.0"
  - "NODE 4  COORD 0.0 1.0 0.0"
  - "NODE 5  COORD 0.0 0.0 1.0"
  - "NODE 6  COORD 1.0 0.0 1.0"
  - "NODE 7  COORD 1.0 1.0 1.0"
  - "NODE 8  COORD 0.0 1.0 1.0"
  - "NODE 9  COORD 0.0 0.0 2.0"
  - "NODE 10 COORD 1.0 0.0 2.0"
  - "NODE 11 COORD 1.0 1.0 2.0"
  - "NODE 12 COORD 0.0 1.0 2.0"
TRANSPORT ELEMENTS:
  - "1 POROFLUIDMULTIPHASE HEX8 1 2 3 4 5 6 7 8 MAT 1"
  - "2 POROFLUIDMULTIPHASE HEX8 5 6 7 8 9 10 11 12 MAT 1"

# -- Design surface topology for BCs ----------------------------
DSURF-NODE TOPOLOGY:
  # Inlet (z = 0 face)
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  # Outlet (z = 2 face)
  - "NODE 9  DSURFACE 2"
  - "NODE 10 DSURFACE 2"
  - "NODE 11 DSURFACE 2"
  - "NODE 12 DSURFACE 2"

RESULT DESCRIPTION:
  - POROFLUIDMULTIPHASE:
      DIS: "porofluid"
      NODE: 1
      QUANTITY: "phi1"
      VALUE: 1.0
      TOLERANCE: 1e-06
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "single_phase_3d"
        if variant not in self._TEMPLATES:
            available = ", ".join(sorted(self._TEMPLATES))
            raise ValueError(
                f"Unknown template variant {variant!r} for {self.module_key}. "
                f"Available: {available}"
            )
        return self._TEMPLATES[variant]

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "single_phase_3d",
                "description": (
                    "Single-phase pressure-driven Darcy flow through a 3D "
                    "porous block.  Demonstrates the full material hierarchy "
                    "(MAT_FluidPoroMultiPhase -> MAT_FluidPoroSinglePhase -> "
                    "density/viscosity/permeability laws).  Uses inline mesh "
                    "with HEX8 elements and pressure Dirichlet BCs."
                ),
            },
        ]

    # -- Validation ----------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of porous media parameters.

        Expected keys in *params* (all optional):
            PERMEABILITY       - intrinsic permeability
            BULKMODULUS        - fluid bulk modulus (density law)
            VISCOSITY          - dynamic viscosity
            REL_PERMEABILITY   - relative permeability
            DENSITY            - reference fluid density
            NUMMAT             - number of sub-materials
            MATIDS             - list of sub-material IDs
            NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE - number of fluid phases
            time_step_size     - time step
            theta              - one-step-theta parameter
        """
        errors: list[str] = []

        # Check permeability
        perm = params.get("PERMEABILITY")
        if perm is not None:
            try:
                k = float(perm)
                if k <= 0:
                    errors.append(
                        f"PERMEABILITY must be > 0, got {k}.  "
                        f"Zero permeability means no flow is possible "
                        f"and the system becomes singular."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"PERMEABILITY must be a positive number, got {perm!r}."
                )

        # Check bulk modulus (density law)
        bulk = params.get("BULKMODULUS")
        if bulk is not None:
            try:
                K = float(bulk)
                if K <= 0:
                    errors.append(
                        f"BULKMODULUS must be > 0, got {K}."
                    )
                elif K > 1e8:
                    errors.append(
                        f"WARNING: BULKMODULUS = {K} is very large.  "
                        f"This may cause numerical issues.  Consider "
                        f"using MAT_PoroDensityLawConstant for "
                        f"incompressible fluids."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"BULKMODULUS must be a positive number, got {bulk!r}."
                )

        # Check viscosity
        visc = params.get("VISCOSITY") or params.get("VALUE")
        if visc is not None:
            try:
                mu = float(visc)
                if mu <= 0:
                    errors.append(
                        f"Viscosity must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Viscosity must be a positive number, got {visc!r}."
                )

        # Check relative permeability
        rel_perm = params.get("REL_PERMEABILITY")
        if rel_perm is not None:
            try:
                kr = float(rel_perm)
                if kr <= 0 or kr > 1:
                    errors.append(
                        f"Relative permeability must be in (0, 1], got {kr}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Relative permeability must be a number in (0, 1], "
                    f"got {rel_perm!r}."
                )

        # Check density
        density = params.get("DENSITY")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    errors.append(
                        f"DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check material hierarchy consistency
        nummat = params.get("NUMMAT")
        matids = params.get("MATIDS")
        numphases = params.get("NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE")

        if nummat is not None and matids is not None:
            try:
                nm = int(nummat)
                if isinstance(matids, list) and len(matids) != nm:
                    errors.append(
                        f"NUMMAT = {nm} but MATIDS has {len(matids)} entries.  "
                        f"These must match."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUMMAT must be a positive integer, got {nummat!r}."
                )

        if numphases is not None:
            try:
                np_ = int(numphases)
                if np_ < 1:
                    errors.append(
                        f"NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE must be >= 1, "
                        f"got {np_}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE must be a "
                    f"positive integer, got {numphases!r}."
                )

        # Check time integration parameter theta
        theta = params.get("theta")
        if theta is not None:
            try:
                th = float(theta)
                if th < 0 or th > 1:
                    errors.append(
                        f"theta must be in [0, 1], got {th}.  "
                        f"Use 1.0 for implicit Euler (stable, first-order) "
                        f"or 0.5 for Crank-Nicolson (second-order)."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"theta must be a number in [0, 1], got {theta!r}."
                )

        # Check time step
        dt = params.get("time_step_size")
        if dt is not None:
            try:
                step = float(dt)
                if step <= 0:
                    errors.append(
                        f"time_step_size must be > 0, got {step}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"time_step_size must be a positive number, got {dt!r}."
                )

        return errors
