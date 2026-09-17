"""Particle-Structure Interaction (PASI) generator for 4C.

Covers coupling of a particle method (DEM, SPH, or peridynamics) with
structural finite elements.  Particles interact with the structural
surface via contact or coupling conditions.  Applications include
impact of granular material on structures, sloshing of particle-laden
fluids against deformable containers, and blast loading simulations.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class PASIGenerator(BaseGenerator):
    """Generator for Particle-Structure Interaction problems in 4C."""

    module_key = "pasi"
    display_name = "Particle-Structure Interaction (PASI)"
    problem_type = "Particle_Structure_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Particle-Structure Interaction (PASI) couples a particle "
                "method (DEM, SPH, or peridynamics) with a structural finite "
                "element discretisation.  The particles exert forces on the "
                "structural surface (impact, pressure) and the structure "
                "provides a moving boundary for the particles.  The PROBLEM "
                "TYPE is 'Particle_Structure_Interaction'.  Required sections "
                "include STRUCTURAL DYNAMIC for the FEM field, PARTICLE "
                "DYNAMIC for the particle field, and PASI DYNAMIC for the "
                "coupling parameters.  A BINNING STRATEGY section is needed "
                "for efficient particle-structure contact detection."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "PARTICLE DYNAMIC",
                "PASI DYNAMIC",
                "BINNING STRATEGY",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "PARTICLE DYNAMIC/DEM",
                "PARTICLE DYNAMIC/SPH",
                "PARTICLE DYNAMIC/PD",
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                # NOTE: there is no IO/RUNTIME VTK OUTPUT/PARTICLES
                # section in 4C. Writing one is a hard parse error.
                # There is also no WRITE_PARTICLE_RUNTIME_VTK key
                # anywhere: particle runtime output is unconditional,
                # written on every step for which the result flag is set.
                # The only WRITE_PARTICLE_* keys that exist are
                # WRITE_PARTICLE_ENERGY (PARTICLE DYNAMIC/DEM) and
                # WRITE_PARTICLE_WALL_INTERACTION (PARTICLE DYNAMIC/DEM
                # and PARTICLE DYNAMIC/SPH), and both are content
                # switches, not cadence switches.
            ],
            "materials": {
                "MAT_ParticleDEM": {
                    "description": (
                        "Discrete element method (DEM) particle material.  "
                        "It carries only geometry and mass: every contact "
                        "property (stiffness, damping, restitution, "
                        "friction, Young's modulus, Poisson ratio) lives "
                        "in the PARTICLE DYNAMIC/DEM section instead, not "
                        "in the material."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": (
                                "Particle radius; also the source of the "
                                "mass, m = INITDENSITY * (4/3) pi r^3"
                            ),
                            "range": "> 0",
                        },
                        "INITDENSITY": {
                            "description": "Particle mass density",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_ParticleWallDEM": {
                    "description": (
                        "Wall material referenced by the MAT entry of a "
                        "DESIGN SURFACE PARTICLE WALL condition (or by "
                        "PARTICLE_WALL_MAT for a bounding-box wall).  It "
                        "supplies the particle-to-WALL friction and "
                        "adhesion only; the particle-to-particle values "
                        "in PARTICLE DYNAMIC/DEM do not reach the wall.  "
                        "All three keys are optional and default to -1.0, "
                        "and MAT: -1 (no material at all) is legal as long "
                        "as no tangential, rolling or adhesion law is on."
                    ),
                    "parameters": {
                        "FRICT_COEFF_TANG": {
                            "description": "Wall tangential friction coefficient",
                            "range": "0 is legal here and means a frictionless wall",
                        },
                        "FRICT_COEFF_ROLL": {
                            "description": "Wall rolling friction coefficient",
                            "range": "0 is legal here",
                        },
                        "ADHESION_SURFACE_ENERGY": {
                            "description": "Wall adhesion surface energy",
                            "range": "0 or negative disables wall adhesion silently",
                        },
                    },
                },
                "MAT_ParticleSPHFluid": {
                    "description": (
                        "SPH fluid particle material.  Nine keys are "
                        "required and none has a default, so omitting any "
                        "one is a MATERIALS parse error."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": "Kernel support radius (2*dx cubic, 3*dx quintic)",
                            "range": "> 0",
                        },
                        "INITDENSITY": {
                            "description": "Reference mass density",
                            "range": "> 0",
                        },
                        "REFDENSFAC": {"description": "Reference density factor (GenTait only)"},
                        "EXPONENT": {"description": "Tait exponent (GenTait only)"},
                        "BACKGROUNDPRESSURE": {
                            "description": (
                                "Only read inside the transport-velocity "
                                "branch; inert without "
                                "TRANSPORTVELOCITYFORMULATION"
                            ),
                        },
                        "BULK_MODULUS": {
                            "description": (
                                "Sets the acoustic scale: the speed of "
                                "sound is derived as "
                                "sqrt(BULK_MODULUS / INITDENSITY).  There "
                                "is no SOUNDSPEED key."
                            ),
                            "range": "> 0",
                        },
                        "DYNAMIC_VISCOSITY": {"description": "Physical dynamic viscosity"},
                        "BULK_VISCOSITY": {"description": "Bulk viscosity"},
                        "ARTIFICIAL_VISCOSITY": {"description": "Artificial viscosity"},
                    },
                },
                "MAT_ElastHyper / MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material for the FEM "
                        "domain.  Any structural material supported by "
                        "4C can be used."
                    ),
                },
            },
            "solver": {
                "structure_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": "Solver for the structural FEM equations.",
                },
            },
            "time_integration": {
                "PASI DYNAMIC": (
                    "Controls the coupled time stepping.  TIMESTEP, "
                    "NUMSTEP, MAXTIME define the time loop, and they are "
                    "the ONLY ones that act: the particle and structure "
                    "fields are advanced together on this step.  COUPLING "
                    "also lives here and defaults to "
                    "partitioned_onewaycoup."
                ),
                "PARTICLE DYNAMIC": (
                    "Controls particle-specific settings: DYNAMICTYPE "
                    "(e.g. VelocityVerlet), INTERACTION, "
                    "PHASE_TO_MATERIAL_ID, GRAVITY_ACCELERATION and the "
                    "PARTICLE_WALL_* keys.  It DOES accept TIMESTEP, "
                    "NUMSTEP, RESULTSEVERY and RESTARTEVERY, but under "
                    "PASI those four are dead: PASI DYNAMIC's values win. "
                    "Upstream PASI decks leave them out entirely."
                ),
                "STRUCTURAL DYNAMIC": (
                    "Standard structural time integration.  For impact "
                    "problems, use explicit dynamics (GenAlpha with "
                    "appropriate spectral radius)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] BINNING STRATEGY is MANDATORY "
                    "for PASI — controls spatial partitioning "
                    "for particle-structure contact search. "
                    "Set BIN_SIZE_LOWER_BOUND to roughly the "
                    "particle diameter. Signal: the section "
                    "is declared OPTIONAL in 4C's input "
                    "spec, so omitting it parses cleanly, "
                    "sets up both fields and prints the "
                    "coupled time-stepping banner before "
                    "dying with `We need a discretization "
                    "at this point.` from "
                    "core/binstrategy/4C_binstrategy.cpp. "
                    "That sentence names neither binning "
                    "nor PASI nor the strategy, and reads "
                    "like a missing mesh, so grep for the "
                    "file name rather than for any word of "
                    "the symptom. (Audit 2026-06-02; "
                    "corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] PARTICLE DYNAMIC/INTERACTION "
                    "must be COMPATIBLE with the particle "
                    "materials and the wall treatment. DEM "
                    "particles use Hertz/hooke contact and "
                    "MAT_ParticleDEM; SPH uses boundary "
                    "particles and MAT_ParticleSPHFluid. "
                    "Nothing validates the combination. "
                    "Signal: switching INTERACTION to SPH "
                    "while the deck still carries "
                    "MAT_ParticleDEM and PARTICLE "
                    "DYNAMIC/DEM prints the time-stepping "
                    "banner and then takes a SEGMENTATION "
                    "FAULT — no PROC 0 ERROR block, no "
                    "message, no MPI_Abort banner, and a "
                    "signal exit status rather than 1.  "
                    "There is no 'interaction type "
                    "mismatch' string, and the run never "
                    "reaches a result test, so there is no "
                    "zero-force output to inspect either. "
                    "(Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Mesh] Structural elements at the "
                    "particle-structure interface need "
                    "sufficiently fine mesh resolution to "
                    "resolve individual particle contacts. "
                    "Signal: h_structure >> particle "
                    "diameter spreads each particle's "
                    "contact force over too few "
                    "structural DOFs — the interface "
                    "deflection becomes uniform across "
                    "every moving node instead of showing "
                    "distinct contact patches. h_structure "
                    "~ 0.5 * particle_diameter is a good "
                    "starting point but it is a WINDOW, "
                    "not a floor: refining the interface "
                    "well past it breaks the partitioned "
                    "coupling just as a too-coarse one "
                    "does, and both ends fail with the "
                    "same sentence, 'The partitioned PASI "
                    "solver did not converge in ITEMAX "
                    "steps!' from "
                    "pasi/4C_pasi_partitioned_twowaycoup"
                    ".cpp, which names neither the mesh "
                    "nor the particle diameter nor the "
                    "interface. Vary h in both directions "
                    "before blaming the tolerances. (Audit "
                    "2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] Time step must satisfy "
                    "BOTH the structural CFL AND the "
                    "particle stability limit (depends on "
                    "contact stiffness k_c and particle "
                    "mass m_p): dt < pi * sqrt(m_p/k_c) / "
                    "10.  Take min(dt_struct, dt_particle). "
                    "Signal: you never get to watch "
                    "particles oscillate.  With COUPLING: "
                    "partitioned_twowaycoup, 4C runs a "
                    "fixed-point iteration inside each "
                    "step, and once the particle response "
                    "stops being a contraction the "
                    "iteration fails outright: `The "
                    "partitioned PASI solver did not "
                    "converge in ITEMAX steps!` from "
                    "pasi/4C_pasi_partitioned_twowaycoup."
                    "cpp.  It aborts there, so no result "
                    "test is reached and no trajectory is "
                    "written, and the message names "
                    "neither the particle field nor the "
                    "contact stiffness nor the time step. "
                    "(Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] PASI requires SEPARATE geometry "
                    "sections for structural mesh AND "
                    "particle initial positions: NODE "
                    "COORDS plus STRUCTURE ELEMENTS for one "
                    "field, PARTICLES for the other. "
                    "Signal: a particle line placed in "
                    "STRUCTURE ELEMENTS is read as an "
                    "element line, so the parser trips on "
                    "the first token while trying to read "
                    "an element id — `Could not parse "
                    "'TYPE' as an integer value.` from "
                    "core/io/src/4C_io_value_parser.cpp. "
                    "That mentions neither particles nor "
                    "the PARTICLES section and reads like "
                    "an ordinary malformed-element error, "
                    "so it sends you to the wrong place; "
                    "there is no 'unexpected particle "
                    "entry' string in 4C. (Audit "
                    "2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] For SPH-structure coupling, "
                    "boundary handling at the structural "
                    "surface requires SPECIAL attention. "
                    "The knob is WALLFORMULATION in "
                    "PARTICLE DYNAMIC/SPH, and its default "
                    "is NoWallFormulation — so leaving it "
                    "out is not 'plain coupling', it is "
                    "switching the boundary treatment off. "
                    "Upstream's PASI SPH deck sets "
                    "VirtualParticleWallFormulation. "
                    "Signal: without it the deck is "
                    "accepted without a word, runs the "
                    "whole coupled loop and reaches every "
                    "result test, but the kernel support "
                    "reaches through the wall into empty "
                    "space, so wall-adjacent density comes "
                    "out DEFICIENT and the particle is no "
                    "longer held off the surface — its "
                    "normal velocity can reverse sign.  "
                    "Result-test a near-wall density "
                    "against INITDENSITY. (Audit "
                    "2026-06-02; confirmed by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] COUPLING defaults to "
                    "partitioned_onewaycoup, so a deck that "
                    "omits it is a ONE-WAY simulation: the "
                    "structure moves the particles and the "
                    "particle forces never come back. That is "
                    "the wrong default to inherit silently for "
                    "anyone who thinks 'PASI' means two-way. "
                    "Signal: nothing names the coupling — the "
                    "run does not print the word at all, "
                    "completes normally, and the only "
                    "distinguishing trace is that the "
                    "per-timestep partitioned-iteration lines "
                    "a two-way run emits are simply absent, "
                    "because a one-way scheme has no fixed-"
                    "point loop to report. Grep for 'iteration' "
                    "and expect a count of zero when the "
                    "coupling has silently degraded. The "
                    "two-way values are partitioned_twowaycoup, "
                    "partitioned_twowaycoup_disprelax and "
                    "partitioned_twowaycoup_disprelaxaitken; "
                    "all three need at least one CONVTOL* set, "
                    "since those default to -1.0. (Verified by "
                    "execution 2026-08-07.)"
                ),
                (
                    "[Numerical] Under PASI the time step "
                    "belongs to PASI DYNAMIC, and a TIMESTEP "
                    "written in PARTICLE DYNAMIC is INERT — "
                    "not merged, not warned about, not "
                    "honoured. Every upstream PASI deck leaves "
                    "PARTICLE DYNAMIC without a TIMESTEP for "
                    "exactly this reason. Signal: this is one "
                    "of the few places where 4C hands you the "
                    "answer directly. At startup it prints a "
                    "table headed 'Overview of chosen time "
                    "stepping:' with a PASI, a Particles and a "
                    "Structure column, followed by 'currently "
                    "equal for both structure and particle "
                    "field' — read the Particles column and "
                    "you will see the PASI step, whatever you "
                    "wrote in PARTICLE DYNAMIC. Setting a "
                    "conflicting particle step reproduces the "
                    "untouched run's result verdicts exactly, "
                    "so do not try to sub-cycle the particle "
                    "field that way; change PASI DYNAMIC's "
                    "TIMESTEP, which moves both fields "
                    "together. (Verified by execution "
                    "2026-08-07.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "dem_impact_3d",
                    "description": (
                        "DEM granular particles impacting a deformable "
                        "structural plate.  Tests particle-structure "
                        "contact force transfer, structural response to "
                        "distributed impact loads, and time step "
                        "stability."
                    ),
                    "template_variant": "dem_impact_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "dem_impact_3d",
                "description": (
                    "3-D DEM-structure interaction: granular particles "
                    "impacting a deformable plate.  Hertz contact, "
                    "SOLID HEX8 structure, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "dem_impact_3d") -> str:
        templates = {
            "dem_impact_3d": self._template_dem_impact_3d,
        }
        if variant == "default":
            variant = "dem_impact_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_dem_impact_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Particle-Structure Interaction (PASI) -- DEM Impact
            #
            # Discrete element particles (granular) impact a deformable
            # structural plate.  Particles use Hertz contact with the
            # structural surface.
            #
            # Mesh: exodus file with:
            #   element_block 1 = structural plate (HEX8)
            #   node_set 1 = plate fixed boundary
            # Particles: defined in PARTICLES section
            # ---------------------------------------------------------------
            TITLE:
              - "3-D particle-structure interaction -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Particle_Structure_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
              STRUCT_STRESS: "Cauchy"
              STRUCT_STRAIN: "GL"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true
            # No IO/RUNTIME VTK OUTPUT/PARTICLES section exists — writing
            # one is a hard parse error — and there is no
            # WRITE_PARTICLE_RUNTIME_VTK key either. Particle output is
            # written unconditionally at the PASI DYNAMIC RESULTSEVERY
            # interval.

            # == Structure (deformable plate) ==================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "GenAlpha"
              TIMESTEP: <structure_timestep>
              NUMSTEP: <structure_num_steps>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>
            STRUCTURAL DYNAMIC/GENALPHA:
              RHO_INF: <genalpha_rho_inf>

            # == Particle ======================================================
            # NO TIMESTEP / NUMSTEP / RESULTSEVERY / RESTARTEVERY here: they
            # are legal keys but PASI DYNAMIC's values win and these are
            # silently ignored. There is also no WRITE_PARTICLE_RUNTIME_VTK
            # key in 4C — particle runtime output is unconditional.
            PARTICLE DYNAMIC:
              INTERACTION: "DEM"
              GRAVITY_ACCELERATION: "<gx> <gy> <gz>"
              PHASE_TO_MATERIAL_ID: "phase1 1"
              PARTICLE_WALL_SOURCE: "DiscretCondition"
              PARTICLE_WALL_MOVING: true
              PARTICLE_WALL_LOADED: true
            # Contact properties live HERE, not in MAT_ParticleDEM.
            PARTICLE DYNAMIC/DEM:
              MAX_RADIUS: <max_particle_radius>
              MAX_VELOCITY: <max_expected_particle_velocity>
              REL_PENETRATION: <relative_penetration>
              COEFF_RESTITUTION: <coefficient_of_restitution>
              POISSON_RATIO: <particle_poisson_ratio>

            # == PASI coupling =================================================
            # COUPLING defaults to partitioned_onewaycoup, i.e. a deck that
            # omits it is a ONE-WAY run. The two-way schemes need at least
            # one CONVTOL* set, since those default to -1.0.
            PASI DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              COUPLING: partitioned_twowaycoup
              CONVTOLSCALEDDISP: <scaled_displacement_tolerance>
              CONVTOLRELATIVEDISP: <relative_displacement_tolerance>
              CONVTOLSCALEDFORCE: <scaled_force_tolerance>
              CONVTOLRELATIVEFORCE: <relative_force_tolerance>

            # == Binning strategy (spatial search) =============================
            # ONE key, DOMAINBOUNDINGBOX, and it is a whitespace-separated
            # STRING of six numbers "xmin ymin zmin xmax ymax zmax" — not a
            # YAML list and not a LOWER/UPPER pair.
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
            MATERIALS:
              # DEM particle material — geometry and mass ONLY.
              # Contact stiffness / restitution / friction are keys of
              # PARTICLE DYNAMIC/DEM, not of the material.
              - MAT: 1
                MAT_ParticleDEM:
                  INITRADIUS: <particle_radius>
                  INITDENSITY: <particle_density>
              # Wall material for the coupling surface. MAT_ParticleWallDEM
              # supplies wall friction/adhesion only; an empty {} (or
              # MAT: -1 on the condition) is legal for pure normal contact.
              - MAT: 4
                MAT_ParticleWallDEM: {}
              # Structural plate material
              - MAT: 2
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [3]
                  DENS: <structure_density>
              - MAT: 3
                ELAST_CoupNeoHooke:
                  YOUNG: <structure_Young_modulus>

            # == Boundary Conditions ===========================================

            # Structural: fixed boundary
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # The coupling surface itself. Required because
            # PARTICLE_WALL_SOURCE is "DiscretCondition". MAT points at the
            # MAT_ParticleWallDEM entry; MAT: -1 (no wall material) is legal
            # as long as no tangential, rolling or adhesion law is on.
            DESIGN SURFACE PARTICLE WALL:
              - E: <coupling_surface_id>
                MAT: 4

            # == Particles =====================================================
            # PARTICLES is a YAML list of whitespace-delimited STRINGS, not
            # a list of mappings. The grammar is
            #   TYPE <phasename> POS <x> <y> <z> [RAD <r>] [RIGIDCOLOR <c>]
            # with the phase name drawn from phase1 / phase2 /
            # boundaryphase / rigidphase / dirichletphase / neumannphase /
            # pdphase. There is no MAT, VELOCITY or RADIUS key here: the
            # material comes from PHASE_TO_MATERIAL_ID and the initial
            # velocity from INITIAL_VELOCITY_FIELD. Global particle ids are
            # assigned in file order starting at 0 — that is the ID a
            # RESULT DESCRIPTION PARTICLE entry refers to.
            PARTICLES:
              - "TYPE phase1 POS <x> <y> <z>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispz"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check particle density.  The particle materials spell it
        # INITDENSITY; DENS belongs to the STRUCTURAL material.
        dens = params.get("INITDENSITY") or params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(f"INITDENSITY must be > 0, got {d}.")
            except (TypeError, ValueError):
                issues.append(
                    f"INITDENSITY must be a positive number, "
                    f"got {dens!r}."
                )

        # Check particle radius.  MAT_ParticleDEM / MAT_ParticleSPHFluid
        # spell it INITRADIUS; there is no plain RADIUS particle key.
        radius = params.get("INITRADIUS")
        if radius is not None:
            try:
                r = float(radius)
                if r <= 0:
                    issues.append(
                        f"INITRADIUS must be > 0, got {r}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"INITRADIUS must be a positive number, got {radius!r}."
                )
        if params.get("RADIUS") is not None:
            issues.append(
                "RADIUS is not a particle-material key.  MAT_ParticleDEM, "
                "MAT_ParticlePD, MAT_ParticleSPHFluid and "
                "MAT_ParticleSPHBoundary all use INITRADIUS."
            )

        # Check contact Young's modulus.  For DEM contact this is
        # YOUNG_MODULUS in PARTICLE DYNAMIC/DEM; plain YOUNG is a
        # structural-material key (ELAST_CoupNeoHooke, ELAST_CoupSVK, ...).
        for young_key in ("YOUNG_MODULUS", "YOUNG"):
            young = params.get(young_key)
            if young is not None:
                try:
                    e = float(young)
                    if e <= 0:
                        issues.append(f"{young_key} must be > 0, got {e}.")
                except (TypeError, ValueError):
                    issues.append(
                        f"{young_key} must be a positive number, "
                        f"got {young!r}."
                    )

        # Check BIN_SIZE_LOWER_BOUND
        bin_size = params.get("BIN_SIZE_LOWER_BOUND")
        if bin_size is not None:
            try:
                bs = float(bin_size)
                if bs <= 0:
                    issues.append(
                        f"BIN_SIZE_LOWER_BOUND must be > 0, got {bs}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"BIN_SIZE_LOWER_BOUND must be a positive number, "
                    f"got {bin_size!r}."
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
