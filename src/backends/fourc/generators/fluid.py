"""Fluid dynamics (incompressible Navier-Stokes) generator for 4C.

Covers SUPG/PSPG-stabilised incompressible Navier-Stokes on fixed (Euler)
or moving (ALE) grids.  Provides two template variants: a 2-D channel flow
and a 2-D lid-driven cavity.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FluidGenerator(BaseGenerator):
    """Generator for incompressible Navier-Stokes fluid problems in 4C."""

    module_key = "fluid"
    display_name = "Fluid Dynamics (Incompressible Navier-Stokes)"
    problem_type = "Fluid"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Incompressible Navier-Stokes solver using residual-based "
                "SUPG/PSPG stabilisation.  Supports fixed Eulerian grids "
                "(NA: Euler) and arbitrary-Lagrangian-Eulerian moving grids "
                "(NA: ALE, required for FSI coupling).  Velocity and pressure "
                "are solved in a monolithic system; the pressure DOF is the "
                "last DOF per node (NUMDOF includes pressure)."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "FLUID DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "FLUID GEOMETRY",
            ],
            "optional_sections": [
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid material.  4C solves incompressible "
                        "Navier-Stokes; density and dynamic viscosity fully "
                        "characterise the fluid."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity mu [Pa s]",
                            "range": "> 0 (water ~1e-3, air ~1.8e-5)",
                        },
                        "DENSITY": {
                            "description": "Fluid density rho [kg/m^3]",
                            "range": "> 0 (water ~1000, air ~1.2)",
                        },
                    },
                },
            },
            "solver": {
                "small_2d": {
                    "type": "UMFPACK (direct)",
                    "when": "Small 2-D problems (< ~50 k DOFs)",
                },
                "large_or_3d": {
                    "type": "Belos (iterative) with block preconditioner",
                    "when": "Larger problems or 3-D",
                    "notes": (
                        "Use AZTOL ~1e-8, AZSUB 300. "
                        "Block preconditioner via MueLu XML recommended."
                    ),
                },
            },
            "time_integration": {
                "schemes": [
                    "Np_Gen_Alpha (recommended -- second-order, A-stable, controllable dissipation)",
                    "BDF2 (second-order, strongly A-stable)",
                    "OneStepTheta (first-order with theta=1, i.e. backward Euler)",
                    "Stationary (pseudo-time stepping to steady state)",
                ],
                "key_parameters": {
                    "TIMEINTEGR": "Time integration scheme name",
                    "TIMESTEP": "Time step size dt",
                    "NUMSTEP": "Number of time steps",
                    "MAXTIME": "Maximum simulation time",
                    "ITEMAX": "Max nonlinear iterations per step (default 10)",
                },
            },
            "element_options": {
                "NA": {
                    "Euler": "Fixed grid (standard CFD)",
                    "ALE": "Moving grid (required for FSI)",
                },
                "NUMDOF": {
                    "2D": "3 (vx, vy, pressure)",
                    "3D": "4 (vx, vy, vz, pressure)",
                },
            },
            "dimensionless_numbers": {
                "Reynolds": "Re = rho * U * L / mu  (inertia vs. viscous forces)",
            },
            "pitfalls": [
                # Every Signal: below was produced by running
                #   LD_LIBRARY_PATH=/opt/4C-dependencies/lib stdbuf -oL -eL \
                #     {FOURC_BINARY} <deck>.yaml <out>
                # on 4C 2026.2.0-dev (commit 89519cfe76), mutating one key at a
                # time in the upstream deck tests/input_files/
                # f2_stokes_residualbased.4C.yaml (2-D, 16 FLUID QUAD4,
                # Stokes, UMFPACK, 5 pinned RESULT DESCRIPTION values).
                (
                    "[Input] A fluid Dirichlet condition counts the PRESSURE dof: "
                    "NUMDOF is 3 in 2-D (vx, vy, p) and 4 in 3-D. Writing the "
                    "velocity count instead is the common slip. Signal: the check "
                    "is ONE-SIDED, so only the too-few direction is caught. "
                    "NUMDOF: 2 with 2-entry ONOFF/VAL/FUNCT on a DESIGN LINE "
                    "DIRICH of the Stokes deck aborts at exit 1 reading 2 DOFs "
                    "given but 3 expected in Line Dirichlet boundary condition, "
                    "from core/fem/src/discretization/"
                    "4C_fem_discretization_utils_dbc.cpp:292. That sentence is "
                    "NOT one literal: the line there is a format string of the "
                    "shape <given> DOFs given but <expected> expected in "
                    "<condition name>, so both counts and the name arrive at run "
                    "time. Two things are greppable — the 14-character run DOFs "
                    "given but, which has exactly one hit in the whole tree "
                    "(that line), and the condition name, which is a string of "
                    "its own: 'Line Dirichlet boundary condition' in "
                    "core/legacy_enum_definitions/"
                    "4C_legacy_enum_definitions_conditions.cpp:22, next to the "
                    "Point, Surface and Volume spellings. The message names "
                    "the entity kind (Point|Line|Surface|Volume) but never the "
                    "field. NUMDOF: 4 with 4-entry arrays on the same condition "
                    "is ACCEPTED: the run reaches 'processor 0 finished normally' "
                    "at exit 0 and the surplus entry is dropped without a word. A "
                    "3-D block pasted into a 2-D deck is therefore invisible. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Numerical] 4C's FLUID QUAD4/HEX8 are equal-order "
                    "velocity-pressure elements, so they are inf-sup unstable and "
                    "lean entirely on PSPG for the pressure. Losing it does not "
                    "make the solver complain — it destroys the pressure while "
                    "leaving the velocity right, which is why it gets mistaken "
                    "for a boundary-condition bug. Signal: setting PSPG: false in "
                    "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION (or STABTYPE: "
                    "no_stabilization, which gives bit-identical numbers) on the "
                    "Stokes deck keeps velx/vely correct to 4.08e-15 while the "
                    "pinned pressures explode to -2.84365229115640386e+03 at node "
                    "19 (expected 0.25), -6.95780745456379373e+04 at node 25 "
                    "(expected 0.5) and -1.88680047107724422e+04 at node 1 "
                    "(expected -0.5) — node-to-node sign and magnitude swings, "
                    "i.e. a checkerboard, not an offset. There is no solver "
                    "warning and no divergence; the only reason the run stops is "
                    "'Result check failed with 3 errors out of 5 tests' from "
                    "core/utils/src/result_test/4C_utils_result_test.cpp:181. "
                    "Without a RESULT DESCRIPTION you get exit 0 and a garbage "
                    "pressure field. PSPG is isolated as the cause because SUPG "
                    "and GRAD_DIV are already false in that deck and cannot be "
                    "turned on to confound it: setting SUPG: true on a Stokes "
                    "deck aborts with 'Having SUPG-stabilization switched on (by "
                    "default?) for Stokes problems, does not make sense! Please "
                    "turn on brain before using 4C!' from "
                    "src/fluid_ele/4C_fluid_ele_parameter.cpp:188. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Numerical] When the velocity is Dirichlet on the WHOLE "
                    "boundary (lid-driven cavity, and the Stokes skew-flow deck), "
                    "the pressure is fixed only up to an additive constant. Fix "
                    "it with one pressure dof — ONOFF: [1, 1, 1] on a DESIGN "
                    "POINT DIRICH — or with a DESIGN SURF MODE FOR KRYLOV SPACE "
                    "PROJECTION block. Signal: leaving the null space open is "
                    "SILENT. Deleting the Krylov-projection section from the "
                    "Stokes deck still runs to completion, and UMFPACK prints no "
                    "singular-matrix, zero-pivot or rank-deficiency message of "
                    "any kind; the tell is that every pinned pressure is off by "
                    "the SAME number — abs(diff) = 3.91001280071499053e+01 at "
                    "node 19, 3.91001280071499053e+01 at node 25 and "
                    "3.91001280071499124e+01 at node 1, agreeing to 15 digits — "
                    "while velx/vely stay exact. Pressure DIFFERENCES are right, "
                    "the level is arbitrary. Pinning one node (VAL: [1, 1, 0.5] "
                    "with ONOFF: [1, 1, 1]) turns all five result tests CORRECT "
                    "at exit 0. If your pressures look uniformly shifted, look "
                    "for this before you re-derive the physics. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Syntax] ONOFF, VAL and FUNCT must each hold exactly NUMDOF "
                    "entries — the arrays are sized by the schema, not padded. "
                    "Signal: NUMDOF: 3 with 2-entry arrays is rejected before any "
                    "mesh is built with \"Failed to match condition specification "
                    "in section 'DESIGN LINE DIRICH CONDITIONS'\" — the section "
                    "name is in SINGLE quotes in the real output — from "
                    "core/fem/src/condition/4C_fem_condition_definition.cpp:79, "
                    "followed by 'Could not match this input' from "
                    "core/io/src/4C_io_input_spec_builders.cpp:633, the offending "
                    "YAML block echoed, and a candidate dump whose useful lines "
                    "are \"[!] Candidate parameter 'ONOFF' has incorrect size\" "
                    "and the same for 'VAL' and 'FUNCT'; exit 1. Grep for "
                    "'has incorrect size' — that is the line that names the array. "
                    "Note this is a DIFFERENT failure from a wrong NUMDOF with "
                    "consistent arrays, which passes the schema and is caught (or "
                    "not) much later by the dof-count check. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] In an exodus-based deck the element category inside "
                    "FLUID GEOMETRY/ELEMENT_BLOCKS must be FLUID. It is NOT "
                    "schema-enforced per field: 4C builds ONE ELEMENT_BLOCKS spec "
                    "and reuses it for STRUCTURE, FLUID, ALE, THERMO and the rest "
                    "(global_legacy_module/4C_global_legacy_module_"
                    "validparameters.cpp, add_geometry_section over known_fields), "
                    "so SOLID under FLUID GEOMETRY parses. Signal: the mistake "
                    "surfaces late and blames something else. On the upstream "
                    "tests/tutorials/preconditioner/tutorial_prec_fsi.4C.yaml, "
                    "swapping the fluid block's FLUID: for SOLID: (adding the "
                    "KINEM the SOLID spec requires) while keeping the fluid "
                    "material SEGFAULTS with no 4C diagnostic at all — exit "
                    "139, and the four signal lines you then see belong to "
                    "OpenMPI, not to 4C: its libopen-pal handler prints the "
                    "process-received-signal block naming the segmentation "
                    "fault, signal 11, with signal code address-not-mapped, 1, "
                    "and a nil failing address. That block changes with the MPI "
                    "build and the signal name comes from libc, so do not grep "
                    "for it; grep for exit status 139 and for the absence of any "
                    "PROC 0 ERROR block. The last named "
                    "frames are Discret::Elements::SolidEleCalc<...>::setup and "
                    "Core::IO::MeshReader::read_and_partition — and note that "
                    "OpenMPI prints those frames mangled "
                    "(_ZN5FourC7Discret8Elements12SolidEleCalc...), unlike the "
                    "demangled backtrace inside a 4C error block, so search the "
                    "mangled form for the substring. Point the same "
                    "SOLID block at a structural material instead and the mesh "
                    "reads fine, then the ALE clone aborts with 'no matching "
                    "material ID (2) in map' from core/fem/src/general/utils/"
                    "4C_fem_general_utils_createdis.hpp:663 — a CLONING MATERIAL "
                    "MAP message that says nothing about element categories. Only "
                    "a key the wrong category does not own (NA: ALE under SOLID) "
                    "is caught at parse, as 'Could not match this input' from "
                    "core/io/src/4C_io_input_spec_builders.cpp:633 followed by "
                    "the match tree. In that tree the two useful lines carry the "
                    "exclamation marker and read Candidate group, naming FLUID "
                    "GEOMETRY, and nested one level under it Candidate list, "
                    "naming ELEMENT_BLOCKS. Neither line exists as a literal in "
                    "the code that prints it: 4C composes it in one std::format "
                    "at core/io/src/4C_io_input_spec_builders.cpp:352 out of a "
                    "state symbol, a state phrase from the table at :349, the "
                    "spec type and the spec name, so the longest run of it there "
                    "is the single word Candidate. Grepping the tree for "
                    "Candidate group does hit — in the expected-output block of "
                    "4C's own spec unit test, core/io/tests/"
                    "4C_io_input_spec_test.cpp — while Candidate list has zero "
                    "hits anywhere, which tells you about that test's coverage "
                    "and nothing about your run. Anchor on a neighbouring line "
                    "that IS "
                    "a literal instead — '[!] The following data remains unused:' "
                    "from the same file at :332, which is printed for the block "
                    "that failed. Reproduce with tests/tutorials/preconditioner/"
                    "tutorial_prec_fsi.4C.yaml, FLUID: swapped for SOLID: with "
                    "NA: ALE left in place. "
                    "(Verified by execution 2026-08-09; re-executed 2026-08-10.)"
                ),
                (
                    "[Input] Use NA: Euler on FLUID elements of a pure-fluid "
                    "problem; NA: ALE only when an ALE field actually exists "
                    "(PROBLEMTYPE Fluid_Ale, Fluid_Structure_Interaction, ...). "
                    "NA: ALE makes the element ask for the mesh-displacement "
                    "state every time it is evaluated, and under PROBLEMTYPE: "
                    "Fluid nothing ever provides it. Signal: the abort names the "
                    "missing STATE VECTOR, never the NA keyword, and which of "
                    "three messages you get depends on what touches the element "
                    "first. On the Stokes deck with PHYSICAL_TYPE: Incompressible "
                    "it is 'Cannot find state {} in discretization {}' "
                    "from core/fem/src/discretization/4C_fem_discretization.hpp:"
                    "1849, which the run renders as Cannot find state dispnp in "
                    "discretization fluid. With PHYSICAL_TYPE: Stokes it is "
                    "intercepted earlier "
                    "by 'ALE with Oseen or Stokes seems to be a tricky "
                    "combination. Think deep before removing FOUR_C_THROW!' from "
                    "src/fluid_ele/4C_fluid_ele_calc.cpp:1738. If the deck also "
                    "carries a DESIGN SURF MODE FOR KRYLOV SPACE PROJECTION, the "
                    "projection setup gets there first with 'Cannot get state "
                    "vector dispnp' from src/fluid_ele/4C_fluid_ele_calc.cpp:7054. "
                    "All three exit 1 before the first time step. Search the log "
                    "for 'dispnp', not for 'ALE'. "
                    "(Verified by execution 2026-08-09.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "channel_flow_2d",
                    "description": (
                        "Poiseuille flow in a 2-D channel with parabolic inlet "
                        "velocity.  Good for verifying pressure drop and velocity "
                        "profile against the analytical solution."
                    ),
                },
                {
                    "name": "lid_driven_cavity",
                    "description": (
                        "Enclosed square cavity with a moving lid.  Classic CFD "
                        "benchmark (Ghia et al., 1982).  Tests recirculation, "
                        "pressure field, and stabilisation quality."
                    ),
                },
            ],
        }

    # ── Variants ──────────────────────────────────────────────────────

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "channel_2d",
                "description": (
                    "2-D channel flow (Poiseuille) with parabolic inlet, "
                    "no-slip walls, and natural outflow.  Uses UMFPACK, "
                    "Np_Gen_Alpha, and exodus mesh."
                ),
            },
            {
                "name": "cavity_2d",
                "description": (
                    "2-D lid-driven cavity with unit-velocity lid, no-slip "
                    "walls, and pressure pin.  Classic CFD benchmark."
                ),
            },
        ]

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "channel_2d") -> str:
        templates = {
            "channel_2d": self._template_channel_2d,
            "cavity_2d": self._template_cavity_2d,
        }
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_channel_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Channel Flow (Poiseuille)
            #
            # Domain: [0, 6] x [0, 1]  (L=6, H=1)
            # Inlet:  parabolic velocity profile  u(y) = 4*U_max*y*(1-y)
            # Walls:  no-slip (top / bottom)
            # Outlet: natural (do-nothing) boundary
            # ---------------------------------------------------------------
            TITLE:
              - "2-D channel flow (Poiseuille) -- generated template"
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid"

            # -- Fluid dynamics settings -----------------------------------
            FLUID DYNAMIC:
              LINEAR_SOLVER: 1
              TIMEINTEGR: "Np_Gen_Alpha"
              PREDICTOR: "explicit_second_order_midpoint"
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              RESTARTEVERY: <restart_interval>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # -- Solver (direct for small 2-D problems) --------------------
            SOLVER 1:
              SOLVER: "UMFPACK"

            # -- Material --------------------------------------------------
            MATERIALS:
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <dynamic_viscosity>
                  DENSITY: <fluid_density>

            # -- Mesh (exodus file generated separately) -------------------
            FLUID GEOMETRY:
              FILE: "channel_2d.e"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    QUAD4:
                      MAT: 1
                      NA: Euler

            # -- Boundary conditions ---------------------------------------
            # node_set_id 1 = inlet (left edge, x=0)
            DESIGN LINE DIRICH CONDITIONS:
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [<inlet_velocity>, 0.0, 0.0]
                FUNCT: [1, null, null]
              # node_set_id 2 = bottom wall (y=0)
              - E: 2
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 3 = top wall (y=1)
              - E: 3
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]

            # Parabolic inlet profile: u(y) = 4*y*(1-y)
            FUNCT1:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_profile_expression>"

            # -- VTK output ------------------------------------------------
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true
        """)

    @staticmethod
    def _template_cavity_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Lid-Driven Cavity
            #
            # Domain: [0, 1] x [0, 1]
            # Lid:    u_x = 1 on top edge (y=1)
            # Walls:  no-slip on left, right, bottom
            # Pressure: pinned to 0 at bottom-left corner
            # ---------------------------------------------------------------
            TITLE:
              - "2-D lid-driven cavity -- generated template"
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid"

            # -- Fluid dynamics settings -----------------------------------
            FLUID DYNAMIC:
              LINEAR_SOLVER: 1
              TIMEINTEGR: "Np_Gen_Alpha"
              PREDICTOR: "explicit_second_order_midpoint"
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              RESTARTEVERY: <restart_interval>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # -- Solver (direct for small 2-D problems) --------------------
            SOLVER 1:
              SOLVER: "UMFPACK"

            # -- Material --------------------------------------------------
            # Re = rho * U * L / mu
            MATERIALS:
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <dynamic_viscosity>
                  DENSITY: <fluid_density>

            # -- Mesh (exodus file generated separately) -------------------
            FLUID GEOMETRY:
              FILE: "cavity_2d.e"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    QUAD4:
                      MAT: 1
                      NA: Euler

            # -- Boundary conditions ---------------------------------------
            # node_set_id 1 = bottom wall (y=0, no-slip)
            DESIGN LINE DIRICH CONDITIONS:
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 2 = right wall (x=1, no-slip)
              - E: 2
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 3 = top lid (y=1, u_x = lid_velocity)
              - E: 3
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [<lid_velocity>, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 4 = left wall (x=0, no-slip)
              - E: 4
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]

            # Pin pressure at bottom-left corner (node_set_id 5)
            DESIGN POINT DIRICH CONDITIONS:
              - E: 5
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [0, 0, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]

            # -- VTK output ------------------------------------------------
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true
        """)

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate fluid-specific parameters.

        Checks:
        - viscosity > 0
        - density > 0
        - Reynolds number warning if very high (> 5000)
        - NUMDOF consistency with DIM
        """
        issues: list[str] = []

        viscosity = params.get("viscosity") or params.get("DYNVISCOSITY")
        density = params.get("density") or params.get("DENSITY")

        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"DYNVISCOSITY must be positive, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DYNVISCOSITY must be a number, got {viscosity!r}."
                )
        else:
            issues.append(
                "DYNVISCOSITY not provided -- required for MAT_fluid."
            )

        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(
                        f"DENSITY must be positive, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a number, got {density!r}."
                )
        else:
            issues.append(
                "DENSITY not provided -- required for MAT_fluid."
            )

        # Reynolds number check
        velocity = params.get("velocity") or params.get("U")
        length = params.get("length") or params.get("L")
        if (
            viscosity is not None
            and density is not None
            and velocity is not None
            and length is not None
        ):
            try:
                mu = float(viscosity)
                rho = float(density)
                U = float(velocity)
                L = float(length)
                if mu > 0:
                    Re = rho * U * L / mu
                    if Re > 5000:
                        issues.append(
                            f"Reynolds number Re = {Re:.0f} is very high.  "
                            f"Consider using finer mesh or turbulence model.  "
                            f"Laminar solver may not converge."
                        )
            except (TypeError, ValueError):
                pass

        # NUMDOF vs DIM
        dim = params.get("DIM") or params.get("dim")
        numdof = params.get("NUMDOF") or params.get("numdof")
        if dim is not None and numdof is not None:
            try:
                d = int(dim)
                n = int(numdof)
                expected = d + 1  # pressure adds one DOF
                if n != expected:
                    issues.append(
                        f"NUMDOF should be {expected} for {d}-D fluid "
                        f"(velocity DOFs + pressure), got {n}."
                    )
            except (TypeError, ValueError):
                pass

        return issues
