"""Generator for contact mechanics physics module (Penalty, Uzawa, Nitsche).

Covers mortar-based contact between deformable bodies using the
CONTACT DYNAMIC + MORTAR COUPLING sections in 4C.  Produces validated,
working .4C.yaml templates for LLM-driven contact simulation setup.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class ContactGenerator(BaseGenerator):
    """Generator for contact mechanics problems in 4C.

    Supports penalty, Uzawa, and Nitsche contact enforcement strategies
    with mortar-based interface coupling between deformable 3D bodies.
    """

    module_key = "contact"
    display_name = "Contact Mechanics (Penalty / Uzawa / Nitsche)"
    problem_type = "Structure"

    # -- Knowledge -----------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The contact mechanics module in 4C models frictionless or "
                "frictional contact between deformable bodies using mortar-based "
                "surface coupling.  It adds two mandatory sections on top of the "
                "standard structural analysis: CONTACT DYNAMIC (enforcement "
                "strategy and parameters) and MORTAR COUPLING (mortar algorithm "
                "settings).  Contact surfaces are defined via "
                "DESIGN SURF MORTAR CONTACT CONDITIONS 3D, where each contact "
                "interface consists of a Slave and a Master surface pair sharing "
                "the same InterfaceID.  The PROBLEM TYPE remains 'Structure'.  "
                "Quasi-static contact problems typically require load stepping "
                "(many small time steps with DYNAMICTYPE GenAlpha or Statics) to "
                "achieve convergence as the contact zone evolves."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "MORTAR COUPLING",
                "CONTACT DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
                # Plus ONE mesh route - these three are alternatives, not
                # three requirements, and only the middle one needs a file:
                #   inline   : NODE COORDS + STRUCTURE ELEMENTS
                #              + D*-NODE TOPOLOGY
                #   Exodus   : STRUCTURE GEOMETRY (FILE + ELEMENT_BLOCKS)
                #   generated: STRUCTURE DOMAIN (HEX/WEDGE cells only)
                "one of: NODE COORDS + STRUCTURE ELEMENTS "
                "| STRUCTURE GEOMETRY | STRUCTURE DOMAIN",
            ],
            "materials": {
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material for contact bodies.  "
                        "Use different material IDs for different bodies to "
                        "allow distinct stiffness values."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5",
                        },
                        "DENS": {
                            "description": (
                                "Mass density.  Set > 0 for dynamic contact; "
                                "can be 0 for quasi-static if using Statics."
                            ),
                            "range": ">= 0",
                        },
                    },
                },
                "MAT_ElastHyper + ELAST_CoupNeoHooke": {
                    "description": (
                        "Neo-Hookean hyperelastic material for large-deformation "
                        "contact.  Recommended when large strains are expected "
                        "near the contact zone."
                    ),
                    "parameters": {
                        "YOUNG": {"description": "Young's modulus", "range": "> 0"},
                        "NUE": {"description": "Poisson's ratio", "range": "0 < nu < 0.5"},
                        "DENS": {"description": "Mass density", "range": ">= 0"},
                    },
                },
            },
            "contact_strategies": {
                "Penalty": {
                    "description": (
                        "Penalty enforcement: adds a stiff penalty spring when "
                        "penetration is detected.  Simple and robust for small "
                        "penetration.  Requires tuning PENALTYPARAM: too low "
                        "allows excessive penetration; too high causes "
                        "ill-conditioning."
                    ),
                    "key_parameter": "PENALTYPARAM",
                    "typical_range": "1e2 -- 1e5 (problem-dependent)",
                },
                "Uzawa": {
                    "description": (
                        "DO NOT SELECT THIS. 'Uzawa' is in the STRATEGY enum "
                        "that `4C --parameters` prints, but the code does not "
                        "implement it: choosing it aborts at setup with "
                        "'This contact strategy is not yet considered!' from "
                        "contact/src/4C_contact_strategy_factory.cpp. Being "
                        "present in --parameters means the parser accepts the "
                        "word, not that the method exists. Use Penalty or "
                        "Lagrange instead. (Verified by execution "
                        "2026-08-03.)"
                    ),
                    "key_parameter": "PENALTYPARAM (initial penalty for augmentation)",
                    "typical_range": "1e2 -- 1e4",
                },
                "Nitsche": {
                    "description": (
                        "Nitsche's method: variationally consistent penalty-like "
                        "formulation.  Combines accuracy of Lagrange multipliers "
                        "with the simplicity of penalty.  Requires consistent "
                        "linearization."
                    ),
                    "key_parameter": "PENALTYPARAM (Nitsche stabilization parameter)",
                    "typical_range": "1e2 -- 1e5",
                },
            },
            "mortar_coupling": {
                "ALGORITHM": (
                    "'Mortar' -- the mortar method for contact surface coupling.  "
                    "This is the standard and recommended algorithm."
                ),
                "LM_SHAPEFCN": (
                    "'Standard' -- standard Lagrange multiplier shape "
                    "functions.  'Dual' is the DEFAULT and is what STRATEGY "
                    "Lagrange with the default SYSTEM: Condensed requires."
                ),
                "LM_DUAL_CONSISTENT": (
                    "Default 'boundary'. MUST be set to 'none' for any "
                    "STRATEGY other than Lagrange while LM_SHAPEFCN is Dual, "
                    "or 4C aborts with 'Consistent dual shape functions in "
                    "boundary elements only for Lagrange multiplier "
                    "strategy.' Setting LM_SHAPEFCN: 'Standard' instead is an "
                    "equally valid escape. The MORTAR COUPLING section must "
                    "also be PRESENT and NON-EMPTY: 'MORTAR COUPLING: {}' "
                    "still aborts. (Verified by execution 2026-08-03 across "
                    "the four LM_SHAPEFCN x LM_DUAL_CONSISTENT combinations "
                    "under STRATEGY Penalty.)"
                ),
            },
            "contact_conditions": {
                "format": (
                    "DESIGN SURF MORTAR CONTACT CONDITIONS 3D defines contact "
                    "surface pairs.  Each interface requires TWO entries with "
                    "the SAME InterfaceID: one with Side: Slave and one with "
                    "Side: Master."
                ),
                "slave_selection": (
                    "The slave surface should generally be the finer mesh or "
                    "the softer body.  This ensures better constraint "
                    "satisfaction."
                ),
                "example": (
                    "- NODE_SET_NAME: contact_slave\n"
                    "  InterfaceID: 1\n"
                    "  Side: Slave\n"
                    "- NODE_SET_NAME: contact_master\n"
                    "  InterfaceID: 1\n"
                    "  Side: Master"
                ),
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, robust for contact problems up to ~50k "
                        "DOFs.  Recommended for initial testing and debugging."
                    ),
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "description": (
                        "Iterative solver with AMG preconditioner.  Contact "
                        "problems can be ill-conditioned; may need tighter "
                        "tolerances or saddle-point preconditioners."
                    ),
                },
            },
            "time_integration": {
                "quasi_static": (
                    "For quasi-static contact: use DYNAMICTYPE 'GenAlpha' with "
                    "small load steps (NUMSTEP = 10--100) to gradually establish "
                    "contact.  Sudden load application often fails to converge."
                ),
                "dynamic_contact": (
                    "For dynamic contact (impact): use DYNAMICTYPE 'GenAlpha' with "
                    "DENS > 0 in all materials.  Time step must be small enough "
                    "to capture the contact event."
                ),
                "TIMESTEP": "Time step size (load stepping parameter for quasi-static).",
                "NUMSTEP": "Number of load/time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "pitfalls": [
                (
                    "[Input] Both MORTAR COUPLING and CONTACT DYNAMIC sections are REQUIRED, "
                    "and each one aborts in a different place, neither of them the input "
                    "parser. Signal: dropping CONTACT DYNAMIC gets to the solver factory "
                    "before it fails -- 'no linear solver defined for meshtying/contact "
                    "problem. Please set LINEAR_SOLVER in CONTACT DYNAMIC to a valid "
                    "number!' from structure_new/src/linear_solver/"
                    "4C_structure_new_solver_factory.cpp, exit 1. Dropping MORTAR COUPLING "
                    "lets its defaults stand and they contradict a penalty strategy, so the "
                    "contact strategy factory stops instead, on the consistent-dual-shape "
                    "check. Its greppable core is 'Consistent dual shape functions in "
                    "boundary elements only for Lagrange'; the source splits the literal "
                    "there and the printed sentence continues with the words multiplier "
                    "strategy. It comes from "
                    "contact/src/4C_contact_strategy_factory.cpp, exit 1. Neither "
                    "message names the missing SECTION, and neither run silently ignores "
                    "the contact conditions. (Audit 2026-06-02; the quoted 'MORTAR COUPLING "
                    "section required for contact' does not exist in 4C, and neither does "
                    "'CONTACT DYNAMIC section missing' -- both falsified by execution "
                    "2026-08-09 on the shipped contact_penalty_3d deck, dropping each "
                    "section in turn.) "
                ),
                (
                    "[Input] Each contact interface needs BOTH a Slave and a Master surface "
                    "with the same InterfaceID. Signal: marking both surfaces Slave gives "
                    "'Master side missing in contact condition group!' from "
                    "4C_contact_utils.cpp, while listing only ONE surface gives 'Not enough "
                    "contact conditions in discretization' -- which never names a side, so it "
                    "reads like a missing-section problem rather than a missing-partner one. "
                    "The log line 'no master partner found for interface X' quoted earlier "
                    "does not exist in 4C, and neither does "
                    "'MortarInterface: InterfaceID X has 0 master elements'. "
                    "(Executed 2026-08-06.) "
                ),
                (
                    "[Numerical] PENALTYPARAM tuning is critical, and both ends fail "
                    "differently. Signal: too low, and the run converges happily while the bodies "
                    "interpenetrate -- there is no warning, only the geometry tells you. Too "
                    "high: Newton stops, with 'The nonlinear solver did not converge!' from "
                    "4C_solver_nonlin_nox_problem.cpp and NO condition-number warning of any "
                    "kind, so the promised ill-conditioning message is not an observable. "
                    "Treat 1e3 as a first bracket rather than a default: on a plain two-block "
                    "penalty problem it still leaves penetration above the five-per-cent-of- "
                    "edge-length rule this entry recommends. Raise it until the penetration "
                    "you measure is acceptable, then stop -- the divergence edge is only a "
                    "couple of decades further up. (Executed 2026-08-06.) "
                ),
                (
                    "[Numerical] Load stepping is NOT a safety property of quasi-static "
                    "contact. Signal: there is none -- the single-step run does not misbehave. The "
                    "earlier advice that applying the full load in one step "
                    "'almost always causes Newton divergence' is wrong. On a two-block mortar "
                    "penalty problem the single-step run converges at every load level tested "
                    "and costs fewer total Newton iterations than the stepped run; at the "
                    "harshest level it is the STEPPED run that fails, with 'The nonlinear "
                    "solver did not converge!'. More steps means more intermediate "
                    "configurations, each with its own active-set transition, and one of "
                    "those can be the one that breaks. What is true is that a contact solve "
                    "can fail where the active set changes sharply. If it does, do not reach "
                    "for a smaller step size first -- check the initial geometry and "
                    "PENALTYPARAM. The claimed 'NOX hits StatusTest::MaxIters at the first "
                    "step' does not appear. (Executed 2026-08-06.) "
                ),
                (
                    "[Numerical] With non-matching meshes, which side is Slave changes the "
                    "LOAD DISTRIBUTION across the interface. Signal: making the coarse face "
                    "the slave lets the mortar projection concentrate the transfer at the "
                    "coarse nodes, and the fine face's own displacement becomes several times "
                    "less uniform between its corners and its centre; making the fine face "
                    "the slave evens it out. The active-set size follows the slave node "
                    "count, which is the quickest way to confirm which side 4C took. The "
                    "convergence half of the older advice did not reproduce: swapping the "
                    "sides left the Newton iteration count unchanged. Rule of thumb stands -- "
                    "slave = finer mesh OR softer material. (Executed 2026-08-06.) "
                ),
                (
                    "[Numerical] KINEM must be 'nonlinear' for contact, and the danger is "
                    "that 4C accepts 'linear' in complete silence: the deck parses, the "
                    "contact interface is built, the active set fills, every step converges, "
                    "and no log line mentions kinematics. Signal: none at run time. The two "
                    "kinematics give different answers and the disagreement GROWS with the "
                    "deformation, so a single small-deformation run looks fine and the error "
                    "only surfaces once the load is raised. Compare a KINEM linear against a "
                    "KINEM nonlinear run at two load levels rather than trusting either "
                    "alone. (Executed 2026-08-06.) "
                ),
                (
                    "[Input] A mortar contact surface is a design SURFACE built from nodes -- "
                    "by default an index into the inline DSURF-NODE TOPOLOGY list. Conditions "
                    "also accept an explicit ENTITY_TYPE of node_set_id, element_block_id or "
                    "a NODE_SET_NAME, and all three of those resolve against a MESH FILE. "
                    "Signal: using them on a deck with inline topology gives \"Cannot apply "
                    "condition 'Contact' to element block N / to node set N which is not "
                    "specified in the mesh file.\" from 4C_fem_condition.cpp; declaring the "
                    "node groups as DVOLUMEs instead of DSURFACEs gives 'not in range [0:0[' "
                    "and, on the next line, 'DSurface condition on non existent DSurface?' "
                    "immediately followed by 'Could not read set from entity type.' — two "
                    "adjacent literals at 4C_fem_condition.cpp:135-136 which the compiler "
                    "concatenates, so the joined sentence exists only at run time. "
                    "Note that element_block_id is a legitimate entity type, "
                    "not a forbidden one -- the refusal above is about the absent mesh file. "
                    "The parser errors 'expected node set for contact surface, got element "
                    "set' and 'MortarInterface needs DNODE not DELE' do not exist. (Executed "
                    "2026-08-06.) "
                ),
                (
                    "[Numerical] Contact surfaces must not overlap in the reference "
                    "configuration. A gap and an exactly touching start both solve; an "
                    "overlap of a fraction of an element breaks the first Newton solve before "
                    "a single step is finalised, with a full active set already present at "
                    "step 0. Signal: 'The nonlinear solver did not converge!' from "
                    "4C_solver_nonlin_nox_problem.cpp, and a NOX force norm at Nonlinear "
                    "Solver Step 0 an order of magnitude above the same deck started touching "
                    "-- but NOTHING that names the cause. There is no MortarInterface message "
                    "about penetration; the log mentions neither penetration nor overlap, so "
                    "this failure is indistinguishable from any other non-convergence and has "
                    "to be found by inspecting the geometry. The quoted 'initial penetration "
                    "X.X > tolerance' does not exist. (Executed 2026-08-06.) "
                ),
            ],
            "typical_experiments": [
                {
                    "name": "two_block_compression",
                    "description": (
                        "Two 3D blocks pressed together.  Bottom block fixed, "
                        "top block displaced downward.  Penalty contact.  "
                        "Classic benchmark for contact algorithm verification."
                    ),
                    "template_variant": "penalty_3d",
                },
                {
                    "name": "hertz_contact",
                    "description": (
                        "Hertzian contact: cylinder or sphere on flat surface.  "
                        "Analytical solution available for contact pressure "
                        "distribution and contact radius.  Good for validation."
                    ),
                },
            ],
        }

    # -- Templates -----------------------------------------------------------

    _TEMPLATES: dict[str, str] = {
        "penalty_3d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
# ----------------------------------------------------------------
# 3D Contact: Two blocks in compression with penalty enforcement
#
# Geometry (requires Exodus mesh with named node sets):
#   - Block 1 (bottom): fixed on bottom face
#   - Block 2 (top): pushed downward via prescribed displacement
#   - Contact between top-of-block1 (master) and bottom-of-block2 (slave)
#
# The mesh must define node sets:
#   wall    -- bottom face of block 1 (fully fixed)
#   pushing -- top face of block 2 (prescribed displacement)
#   slave   -- bottom face of block 2 (contact slave)
#   master  -- top face of block 1 (contact master)
# ----------------------------------------------------------------
TITLE:
  - "Contact -- two 3D blocks in compression with penalty method"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"

IO:
  STRUCT_STRESS: "Cauchy"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: <output_interval_steps>
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
  ELEMENT_MAT_ID: true

# -- Structural dynamics (quasi-static via load stepping) --------
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: <load_step_size>
  NUMSTEP: <number_of_load_steps>
  MAXTIME: <end_time>
  RESTARTEVERY: <restart_interval>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  LINEAR_SOLVER: 1

# -- Mortar coupling (required for contact) ----------------------
MORTAR COUPLING:
  ALGORITHM: Mortar
  LM_SHAPEFCN: Standard

# -- Contact strategy --------------------------------------------
CONTACT DYNAMIC:
  STRATEGY: Penalty
  PENALTYPARAM: <penalty_parameter>
  LINEAR_SOLVER: 1

# -- Solver ------------------------------------------------------
SOLVER 1:
  SOLVER: "UMFPACK"

# -- Materials (two distinct bodies) -----------------------------
MATERIALS:
  # Block 1 (bottom, stiffer)
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus_block1>
      NUE: <Poisson_ratio_block1>
      DENS: <density_block1>
  # Block 2 (top, softer)
  - MAT: 2
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus_block2>
      NUE: <Poisson_ratio_block2>
      DENS: <density_block2>

# -- Loading function (gradual compression) ----------------------
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "compression"
  - VARIABLE: 0
    NAME: "compression"
    TYPE: "linearinterpolation"
    NUMPOINTS: <number_of_interpolation_points>
    TIMES: [<interpolation_times>]
    VALUES: [<interpolation_values>]

# -- Contact surfaces --------------------------------------------
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - NODE_SET_NAME: slave
    InterfaceID: 1
    Side: Slave
  - NODE_SET_NAME: master
    InterfaceID: 1
    Side: Master

# -- Boundary conditions -----------------------------------------
DESIGN SURF DIRICH CONDITIONS:
  # Bottom of block 1: fully fixed
  - NODE_SET_NAME: wall
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  # Top of block 2: prescribed compression in x
  - NODE_SET_NAME: pushing
    NUMDOF: 3
    ONOFF: [1, 0, 0]
    VAL: [<prescribed_displacement>, 0.0, 0.0]
    FUNCT: [1, 0, 0]

# -- Geometry (Exodus mesh) --------------------------------------
STRUCTURE GEOMETRY:
  FILE: contact_two_blocks.e
  SHOW_INFO: detailed_summary
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear
    - ID: 2
      SOLID:
        HEX8:
          MAT: 2
          KINEM: nonlinear

RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispx"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "penalty_3d"
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
                "name": "penalty_3d",
                "description": (
                    "Two 3D blocks in compression with penalty contact.  "
                    "Bottom block fixed, top block pushed down.  "
                    "Demonstrates MORTAR COUPLING, CONTACT DYNAMIC, and "
                    "DESIGN SURF MORTAR CONTACT CONDITIONS 3D sections.  "
                    "Uses quasi-static load stepping with GenAlpha."
                ),
            },
        ]

    # -- Validation ----------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of contact parameters.

        Expected keys in *params* (all optional):
            PENALTYPARAM   - penalty stiffness
            STRATEGY       - contact enforcement strategy
            YOUNG          - Young's modulus of materials
            NUE            - Poisson's ratio
            KINEM          - kinematics type
            slave_surface  - whether slave surface is defined
            master_surface - whether master surface is defined
            TIMESTEP       - time step / load step size
            NUMSTEP        - number of load steps
        """
        errors: list[str] = []

        # Check penalty parameter
        penalty = params.get("PENALTYPARAM")
        if penalty is not None:
            try:
                p = float(penalty)
                if p <= 0:
                    errors.append(
                        f"PENALTYPARAM must be > 0, got {p}.  "
                        f"A non-positive penalty parameter is unphysical."
                    )
                elif p < 10:
                    errors.append(
                        f"WARNING: PENALTYPARAM = {p} is very low.  "
                        f"This will allow large penetration.  Typical "
                        f"range: 1e2 to 1e5."
                    )
                elif p > 1e8:
                    errors.append(
                        f"WARNING: PENALTYPARAM = {p} is very high.  "
                        f"This may cause ill-conditioning and Newton "
                        f"divergence.  Typical range: 1e2 to 1e5."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"PENALTYPARAM must be a positive number, got {penalty!r}."
                )

        # Check strategy
        strategy = params.get("STRATEGY")
        valid_strategies = {"Penalty", "Uzawa", "Nitsche"}
        if strategy is not None and strategy not in valid_strategies:
            errors.append(
                f"STRATEGY must be one of {sorted(valid_strategies)}, "
                f"got {strategy!r}."
            )

        # Check slave/master pairs
        has_slave = params.get("slave_surface")
        has_master = params.get("master_surface")
        if has_slave is not None and has_master is not None:
            if has_slave and not has_master:
                errors.append(
                    "Contact interface has a Slave surface but no Master.  "
                    "Each interface requires BOTH Slave and Master surfaces "
                    "with the same InterfaceID."
                )
            elif has_master and not has_slave:
                errors.append(
                    "Contact interface has a Master surface but no Slave.  "
                    "Each interface requires BOTH Slave and Master surfaces "
                    "with the same InterfaceID."
                )

        # Check kinematics
        kinem = params.get("KINEM")
        if kinem is not None and str(kinem).lower() == "linear":
            errors.append(
                "KINEM is 'linear' but contact problems require 'nonlinear' "
                "kinematics to correctly compute the geometric nonlinearity "
                "of the contact gap."
            )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    errors.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                errors.append(f"YOUNG must be a positive number, got {young!r}.")

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    errors.append(
                        f"NUE (Poisson's ratio) must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                errors.append(f"NUE must be a number in (0, 0.5), got {nue!r}.")

        # Check load stepping
        numstep = params.get("NUMSTEP")
        if numstep is not None:
            try:
                ns = int(numstep)
                if ns < 2:
                    errors.append(
                        f"WARNING: NUMSTEP = {ns}.  Contact problems almost "
                        f"always require load stepping (NUMSTEP >= 5) for "
                        f"Newton convergence.  Consider increasing NUMSTEP."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUMSTEP must be a positive integer, got {numstep!r}."
                )

        # Check TIMESTEP
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    errors.append(
                        f"TIMESTEP must be > 0, got {dt}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        return errors
