"""Generator for scalar transport physics module (Poisson, heat conduction, advection-diffusion).

Covers stationary and transient scalar transport problems using the
SCALAR TRANSPORT DYNAMIC section in 4C.  Produces validated, working
.4C.yaml templates that an LLM can adapt to specific user requests.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class ScalarTransportGenerator(BaseGenerator):
    """Generator for scalar transport problems in 4C.

    Supports Poisson equation, steady-state / transient heat conduction,
    and advection-diffusion with prescribed velocity fields.
    """

    module_key = "scalar_transport"
    display_name = "Scalar Transport (Poisson / Heat / Advection-Diffusion)"
    problem_type = "Scalar_Transport"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The scalar transport module solves the generic advection-diffusion "
                "equation  d(phi)/dt + u . grad(phi) - div(kappa * grad(phi)) = f  "
                "for a scalar field phi.  Special cases include the Poisson equation "
                "(stationary, zero velocity), transient heat conduction, and "
                "advection-dominated transport with stabilisation (SUPG).  "
                "The PROBLEM TYPE is 'Scalar_Transport' and the dynamics section "
                "is 'SCALAR TRANSPORT DYNAMIC' (NOT 'SCATRA DYNAMIC').  "
                "Elements use the TRANSP category and go in the section "
                "'TRANSPORT ELEMENTS' (with NODE COORDS) for an inline mesh, "
                "or in 'TRANSPORT GEOMETRY' -> ELEMENT_BLOCKS for an external "
                "Exodus mesh.  The material is MAT_scatra (DIFFUSIVITY), NOT "
                "MAT_Fourier.  Boundary conditions carry NUMDOF: 1 with "
                "one-entry ONOFF / VAL / FUNCT arrays."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "SCALAR TRANSPORT DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                # Plus ONE mesh route - these three are alternatives, not
                # a list of requirements, and TRANSPORT GEOMETRY is the
                # one that needs an external file:
                #   inline   : NODE COORDS + TRANSPORT ELEMENTS
                #              + D*-NODE TOPOLOGY        <- no external file
                #   Exodus   : TRANSPORT GEOMETRY (FILE + ELEMENT_BLOCKS)
                #   generated: TRANSPORT DOMAIN (HEX/WEDGE cells only)
                "one of: NODE COORDS + TRANSPORT ELEMENTS | TRANSPORT GEOMETRY | TRANSPORT DOMAIN",
            ],
            "materials": {
                "MAT_scatra": {
                    "description": (
                        "Standard scalar transport material.  Defines the "
                        "diffusivity of the transported quantity."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": "Isotropic diffusion coefficient kappa",
                            "range": "> 0  (typical: 0.01 -- 100 depending on units)",
                        },
                    },
                },
                "MAT_Fourier": {
                    "description": (
                        "DO NOT USE THIS UNDER PROBLEMTYPE Scalar_Transport. "
                        "MAT_Fourier belongs to PROBLEMTYPE Thermo with THERMO "
                        "elements. A TRANSP element pointing at it aborts with "
                        "'Material type m_thermo_fourier is not supported!' from "
                        "scatra_ele/4C_scatra_ele_calc.cpp. Heat conduction "
                        "modelled INSIDE the scatra framework uses MAT_scatra "
                        "with the thermal diffusivity alpha = k / (rho * c_p) as "
                        "DIFFUSIVITY. Listed here only so the distinction is "
                        "explicit; the parameters below apply to the Thermo "
                        "problem type. (Verified by execution 2026-08-03: the "
                        "same 2D deck runs with MAT_scatra and aborts with "
                        "MAT_Fourier.)"
                    ),
                    "parameters": {
                        "CAPA": {
                            "description": "Volumetric heat capacity (rho * c_p)",
                            "range": "> 0",
                        },
                        "CONDUCT": {
                            "description": (
                                "Thermal conductivity.  Specified as a YAML mapping "
                                "with 'constant: [value]'."
                            ),
                            "range": "> 0",
                        },
                    },
                },
            },
            "time_integration": {
                "TIMEINTEGR": (
                    "Time integration scheme.  Options: "
                    "'Stationary' (steady-state solve, single step), "
                    "'BDF2' (second-order backward differentiation, robust for stiff problems), "
                    "'One_Step_Theta' (theta-method; theta=1.0 gives implicit "
                    "Euler, theta=0.5 gives Crank-Nicolson), "
                    "'Gen_Alpha'.  NOTE THE UNDERSCORES: this enum is spelled "
                    "'One_Step_Theta' and 'Gen_Alpha', unlike STRUCTURAL and "
                    "THERMAL DYNAMIC, whose DYNAMICTYPE uses CamelCase "
                    "('OneStepTheta', 'GenAlpha').  Writing 'OneStepTheta' here "
                    "aborts with 'Could not match this input' echoing the "
                    "SCALAR TRANSPORT DYNAMIC block."
                ),
                "SOLVERTYPE": (
                    "Use 'linear_full' for linear problems (Poisson, linear heat). "
                    "Use 'nonlinear' only when the equation contains nonlinear terms "
                    "(e.g. radiation, concentration-dependent diffusivity)."
                ),
                "VELOCITYFIELD": (
                    "Velocity field for advective transport.  "
                    "'zero' -- pure diffusion (Poisson / heat conduction).  "
                    "'function' -- prescribed via VELFUNCNO referencing a FUNCT section.  "
                    "'Navier_Stokes' -- coupled with a fluid solve."
                ),
                "TIMESTEP": "Time step size (only relevant for transient problems).",
                "NUMSTEP": "Number of time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, robust for problems up to ~50k DOFs.  "
                        "No preconditioner needed."
                    ),
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "description": (
                        "Iterative Krylov solver with algebraic multigrid "
                        "preconditioner.  Scalable to millions of DOFs."
                    ),
                },
            },
            "pitfalls": [
                (
                    "[Input] The dynamics section name is 'SCALAR TRANSPORT "
                    "DYNAMIC', NOT 'SCATRA DYNAMIC'.  Using the wrong name "
                    "does NOT silently fall back to defaults -- it is a hard "
                    "abort before anything runs. Signal: \"Section 'SCATRA "
                    "DYNAMIC' is not a valid section name.\" from "
                    "core/io/src/4C_io_input_file.cpp, exit 1 at parse. "
                    "(Verified by execution 2026-08-03. An earlier version "
                    "of this entry predicted a SILENT fallback with "
                    "uniformly-zero result fields and quoted an 'unknown "
                    "section: SCATRA DYNAMIC' parser banner; neither is "
                    "real -- that string is not in the binary and the run "
                    "never starts.)"
                ),
                (
                    "[Input] VELOCITYFIELD must be 'zero' (not omitted) "
                    "for pure diffusion, but it is OPTIONAL: 'zero' is "
                    "already its default and a deck that omits it runs "
                    "correctly. Set it explicitly to document intent, or "
                    "when you want 'function' (+ VELFUNCNO) or "
                    "'Navier_Stokes'. Signal: none — there is no error and "
                    "no warning for the omitted key. (An earlier version of "
                    "this entry claimed the omission raised 'requested "
                    "velocity field not found' or 'Vector velnp "
                    "uninitialized'; neither string exists in the 4C binary, "
                    "and a deck identical except for the deleted "
                    "VELOCITYFIELD line completed with exit 0. Corrected "
                    "against the real binary.)"
                ),
                (
                    "[Output] Scalar transport has NO runtime-VTK section at "
                    "all, and you do not need one: a Scalar_Transport run "
                    "writes <prefix>-vtk-files/scatra-*.vtu and "
                    "<prefix>-scatra.pvd automatically, with no IO section "
                    "in the deck whatsoever. Naming an output section is "
                    "FATAL, not merely ineffective. Signal: both "
                    "'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' and "
                    "'IO/RUNTIME VTK OUTPUT/SCATRA' abort at parse with "
                    "\"Section '<name>' is not a valid section name.\" from "
                    "core/io/src/4C_io_input_file.cpp and exit 1. The array "
                    "inside the .vtu is named phi_1, phi_2, ... "
                    "(Verified by execution 2026-08-03: both section names "
                    "were tried and both aborted; the automatic .vtu output "
                    "was confirmed on a deck containing no IO section. This "
                    "corrects an earlier entry that recommended the first "
                    "section name and claimed the failure was silent. "
                    "Re-verified 2026-08-07 at line 546 of the same file.)"
                ),
                (
                    "[Output] There is no key for it either, so do not go "
                    "looking for one to relocate into a valid section. "
                    "OUTPUT_SCATRA and PHI do not exist anywhere in 4C, and "
                    "the scatra .vtu cadence cannot be throttled from the "
                    "input file at all: IO/RUNTIME VTK OUTPUT is a valid "
                    "section, but neither its INTERVAL_STEPS nor SCALAR "
                    "TRANSPORT DYNAMIC's RESULTSEVERY changes how many "
                    "scatra files are written. Signal: writing "
                    "OUTPUT_SCATRA under IO/RUNTIME VTK OUTPUT is a hard "
                    "abort, 'Could not match this input' from "
                    "core/io/src/4C_io_input_spec_builders.cpp with the IO "
                    "block echoed; whereas INTERVAL_STEPS is accepted "
                    "silently and does nothing. A 6-step run produced the "
                    "same 7 scatra-*.vtu (+7 .pvtu) with no IO section, "
                    "with INTERVAL_STEPS 1, with INTERVAL_STEPS 3, with "
                    "RESULTSEVERY 2 and with RESULTSEVERY 3. If you need "
                    "fewer files, sub-sample the .pvd afterwards. "
                    "(Verified by execution 2026-08-07.)"
                ),
                (
                    "[Input] For Exodus-based meshes the geometry section "
                    "is 'TRANSPORT GEOMETRY' with ELEMENT_BLOCKS using "
                    "the TRANSP element category.  Do not use STRUCTURE "
                    "GEOMETRY. But note TRANSPORT GEOMETRY is only ONE of "
                    "three mesh routes and is NOT required: the inline route "
                    "(NODE COORDS + TRANSPORT ELEMENTS + D*-NODE TOPOLOGY) "
                    "needs no external file at all. Beware: writing "
                    "STRUCTURE GEOMETRY instead is NOT caught by "
                    "section-name validation, because every known field has "
                    "a GEOMETRY section and the name is perfectly valid — it "
                    "is simply never read for the scatra field, so the run "
                    "finishes with exit 0 even when the file it names does "
                    "not exist. What 4C does guard is exclusivity: the three "
                    "mesh routes may not be combined, and it names the "
                    "offenders. Signal: an unknown FIELD name is rejected "
                    "before anything runs with \"Section '<name>' is not a "
                    "valid section name.\"; two routes at once give "
                    "\"Multiple options to read mesh for discretization "
                    "'scatra'. Only one is allowed.\" from "
                    "core/io/src/4C_io_meshreader.cpp, followed by the "
                    "section names it found. A misdirected geometry section "
                    "is silent only while it names a field this problem type "
                    "does not instantiate: STRUCTURE GEOMETRY and THERMO "
                    "GEOMETRY are ignored in a Scalar_Transport run, but "
                    "FLUID GEOMETRY is NOT, because Scalar_Transport carries "
                    "a fluid discretization — that route is taken and the "
                    "missing file is reported by "
                    "core/io/src/4C_io_exodus.cpp with a throw of the form "
                    "File <path> does not exist. -- the path is "
                    "substituted at run time and the two literal runs "
                    "around it are too short to grep for. (An earlier "
                    "version of this entry quoted 'expected element category "
                    "TRANSP but got SOLID'; THAT STRING DOES NOT EXIST IN "
                    "THE 4C BINARY, and it claimed a misdirected section is "
                    "always silent. Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] Write NUMDOF 1 for scalar transport boundary "
                    "conditions (a single scalar field) — but understand "
                    "what NUMDOF is: it declares how long ONOFF/VAL/FUNCT "
                    "are, and it is checked against THEM, not against the "
                    "number of transported scalars. Over-declaring it is "
                    "NOT an error: NUMDOF 3 with three-entry arrays on a "
                    "one-scalar problem runs and returns the identical "
                    "answer, the surplus entries addressing DOFs the field "
                    "does not have and being dropped without comment. The "
                    "FIRST entry is the one that acts. What IS rejected is "
                    "an array whose length disagrees with NUMDOF. Signal: "
                    "\"Failed to match condition specification in section "
                    "'DESIGN ... DIRICH CONDITIONS'.\" from "
                    "core/fem/src/condition/4C_fem_condition_definition.cpp "
                    "plus 'Could not match this input', with the offending "
                    "block echoed. (Corrected by execution 2026-08-06; an "
                    "earlier version claimed the arrays must have exactly "
                    "one entry and quoted 'array size mismatch' or "
                    "'expected NUMDOF entries'. neither string exists in "
                    "the 4C binary and the over-declared deck runs fine.)"
                ),
                (
                    "[Numerical] When using BDF2 time "
                    "integration, the first step is "
                    "AUTOMATICALLY handled with a lower-"
                    "order scheme (backward Euler) — no "
                    "special start-up needed, and no way "
                    "to see it in the log, since 4C prints "
                    "the scheme name 'BDF2' for every step "
                    "and announces no switch. In source: "
                    "ScaTra::TimIntBDF2::set_old_part_of_"
                    "righthandside uses theta = 1.0 while "
                    "step_ == 1 and 2/3 afterwards "
                    "(src/scatra/4C_scatra_timint_bdf2.cpp). "
                    "Signal: with TIMEINTEGR: BDF2 in the SCALAR "
                    "TRANSPORT DYNAMIC section, a ONE-step run is "
                    "bit-identical to a one-step "
                    "One_Step_Theta run with THETA 1.0, "
                    "while two- and four-step runs differ — "
                    "that pair of observations is the cheap "
                    "way to confirm the start-up on any "
                    "build. Do not hand-roll a start-up "
                    "sequence. (Verified by execution "
                    "2026-08-06; an earlier version quoted "
                    "an unmeasured claim about a manual "
                    "two-step start doubling the first-step "
                    "error.)"
                ),
            ],
            # A COMPLETE, RUNNABLE scalar-transport deck with a
            # TIME-DEPENDENT Dirichlet BC. It is here because the only
            # worked FUNCT example an agent used to be shown in this
            # payload came from a structural block (NUMDOF: 3, three-entry
            # arrays) and copying it produces exactly the deck the NUMDOF
            # pitfall predicts will abort. Verified by execution
            # 2026-08-03: runs to completion, exit 0.
            "worked_example_time_dependent_bc": """\
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  SOLVERTYPE: "linear_full"
  TIMEINTEGR: "One_Step_Theta"   # underscores, not CamelCase
  THETA: 1.0
  TIMESTEP: 0.05
  NUMSTEP: 20
  MAXTIME: 1.0
  VELOCITYFIELD: "zero"
  INITIALFIELD: "zero_field"
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Scatra_Solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:                  # NOT MAT_Fourier
      DIFFUSIVITY: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1                    # ONE scalar DOF - not 3
    ONOFF: [1]                   # exactly one entry
    VAL: [1.0]                   # amplitude
    FUNCT: [1]                   # INDEX of FUNCT1 -> phi = 1.0 * t
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
TRANSPORT ELEMENTS:
  - "1 TRANSP QUAD4 1 2 3 4 MAT 1 TYPE Std"
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 2
      QUANTITY: "phi"
      VALUE: 0.0
      TOLERANCE: 1.0e30          # record mode: abs(diff) is the true value
""",

            "typical_experiments": [
                {
                    "name": "poisson_2d",
                    "description": (
                        "Steady-state Poisson equation on a 2D square domain "
                        "[0,1]x[0,1] with homogeneous Dirichlet BCs on all sides "
                        "and a constant source term.  Uses TIMEINTEGR: Stationary, "
                        "VELOCITYFIELD: zero, MAT_scatra with DIFFUSIVITY: 1.0."
                    ),
                    "template_variant": "poisson_2d",
                },
                {
                    "name": "heat_transient_2d",
                    "description": (
                        "Transient heat conduction on a 2D plate.  One edge is "
                        "heated (Dirichlet T=100), opposite edge held at T=0.  "
                        "Uses TIMEINTEGR: BDF2, MAT_scatra with DIFFUSIVITY "
                        "representing thermal diffusivity alpha = k/(rho*c_p)."
                    ),
                    "template_variant": "heat_transient_2d",
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    _TEMPLATES: dict[str, str] = {
        "poisson_2d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
TITLE:
  - "Poisson equation on unit square -- stationary diffusion"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: <timestep>
  MAXTIME: <end_time>
  NUMSTEP: <number_of_steps>
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: <diffusivity>
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge1>]
    FUNCT: [0]
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge2>]
    FUNCT: [0]
  - E: 3
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge3>]
    FUNCT: [0]
  - E: 4
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<boundary_value_edge4>]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 5
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<source_term_value>]
    FUNCT: [0]
TRANSPORT GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      TRANSP:
        QUAD4:
          MAT: 1
          TYPE: Std
  FILE: mesh.e
  SHOW_INFO: detailed_summary
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 1
      QUANTITY: "phi"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
        "heat_transient_2d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
TITLE:
  - "Transient heat conduction on 2D plate -- BDF2 time integration"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "BDF2"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: <timestep>
  MAXTIME: <end_time>
  NUMSTEP: <number_of_steps>
  LINEAR_SOLVER: 1
# NO output section. Scalar transport has no runtime-VTK section and no
# runtime-VTK switch: there is no SCALAR TRANSPORT DYNAMIC/RUNTIME VTK
# OUTPUT section, no IO/RUNTIME VTK OUTPUT/SCATRA sub-section, and no
# OUTPUT_SCATRA or PHI key anywhere in 4C. The run writes
# <prefix>-vtk-files/scatra-*.vtu and <prefix>-scatra.pvd by itself, one
# file set per step. IO/RUNTIME VTK OUTPUT is a valid section but its
# INTERVAL_STEPS does not throttle scatra output.
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: <diffusivity>
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<hot_boundary_temperature>]
    FUNCT: [0]
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [<cold_boundary_temperature>]
    FUNCT: [0]
TRANSPORT GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      TRANSP:
        QUAD4:
          MAT: 1
          TYPE: Std
  FILE: mesh.e
  SHOW_INFO: detailed_summary
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 1
      QUANTITY: "phi"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "poisson_2d"
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
                "name": "poisson_2d",
                "description": (
                    "Stationary Poisson equation on a 2D unit square with "
                    "homogeneous Dirichlet BCs and a constant Neumann source."
                ),
            },
            {
                "name": "heat_transient_2d",
                "description": (
                    "Transient heat conduction on a 2D plate with BDF2 time "
                    "integration.  Hot edge (T=100) and cold edge (T=0)."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        # Check DIFFUSIVITY
        diffusivity = params.get("DIFFUSIVITY")
        if diffusivity is not None:
            try:
                d = float(diffusivity)
                if d <= 0:
                    errors.append(
                        f"DIFFUSIVITY must be > 0, got {d}.  A non-positive "
                        f"diffusivity is unphysical and will cause divergence."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DIFFUSIVITY must be a positive number, got {diffusivity!r}."
                )

        # Check CAPA (if Fourier material)
        capa = params.get("CAPA")
        if capa is not None:
            try:
                c = float(capa)
                if c <= 0:
                    errors.append(
                        f"CAPA (heat capacity) must be > 0, got {c}."
                    )
            except (TypeError, ValueError):
                errors.append(f"CAPA must be a positive number, got {capa!r}.")

        # Check CONDUCT (if Fourier material)
        conduct = params.get("CONDUCT")
        if conduct is not None:
            if isinstance(conduct, dict):
                vals = conduct.get("constant", [])
                if isinstance(vals, list):
                    for v in vals:
                        try:
                            if float(v) <= 0:
                                errors.append(
                                    f"CONDUCT values must be > 0, got {v}."
                                )
                        except (TypeError, ValueError):
                            errors.append(
                                f"CONDUCT values must be positive numbers, "
                                f"got {v!r}."
                            )
            else:
                try:
                    if float(conduct) <= 0:
                        errors.append(
                            f"CONDUCT must be > 0, got {conduct}."
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"CONDUCT must be a positive number or dict with "
                        f"'constant: [value]', got {conduct!r}."
                    )

        # Check TIMESTEP for transient problems
        timestep = params.get("TIMESTEP")
        timeintegr = params.get("TIMEINTEGR", "Stationary")
        if timeintegr != "Stationary" and timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    errors.append(
                        f"TIMESTEP must be > 0 for transient problems, got {dt}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        # Warn about VELOCITYFIELD
        velocityfield = params.get("VELOCITYFIELD")
        if velocityfield is not None and velocityfield not in (
            "zero", "function", "Navier_Stokes"
        ):
            errors.append(
                f"VELOCITYFIELD must be 'zero', 'function', or 'Navier_Stokes', "
                f"got {velocityfield!r}."
            )

        return errors
