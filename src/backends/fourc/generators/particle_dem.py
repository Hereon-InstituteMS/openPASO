"""Generator for Discrete Element Method (DEM) simulations in 4C.

DEM is 4C's granular-mechanics particle method: rigid spheres that
interact only on contact, through a normal contact law plus optional
tangential (friction), rolling and adhesion laws.  It shares the
particle engine (binning, phases, PARTICLES section, result tests) with
SPH and peridynamics but has its own interaction stack under
``src/particle/src/interaction/4C_particle_interaction_dem*.cpp``.

The shape that surprises people is where the physics lives.  A DEM
material carries geometry and mass and nothing else; every contact
property -- stiffness, damping, restitution, friction, Young's modulus,
Poisson ratio -- is a key of the ``PARTICLE DYNAMIC/DEM`` section.  Wall
properties are the exception and live in a separate wall material.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ParticleDEMGenerator(BaseGenerator):
    """Generator for DEM (granular) particle simulations."""

    module_key = "particle_dem"
    display_name = "Discrete Element Method (DEM)"
    problem_type = "Particle"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Discrete Element Method (DEM) in 4C models granular media "
                "as rigid spheres that exchange momentum only through "
                "contact.  A normal contact law is always active; "
                "tangential (Coulomb friction), rolling-resistance and "
                "adhesion laws are optional additions.  Particles may also "
                "contact a wall, which is a surface of a structural "
                "discretisation carrying its own DEM wall material, and "
                "may be tied into rigid bodies.  Time integration is "
                "explicit."
            ),
            "problem_type": "Particle",
            "required_sections": [
                "PROBLEM TYPE",
                "IO",
                "BINNING STRATEGY",
                "PARTICLE DYNAMIC",
                "PARTICLE DYNAMIC/DEM",
                "MATERIALS",
                "PARTICLES",
                "RESULT DESCRIPTION",
            ],
            "section_details": {
                "PROBLEM TYPE": {"PROBLEMTYPE": '"Particle"'},
                "PARTICLE DYNAMIC": {
                    "INTERACTION": '"DEM"',
                    "PHASE_TO_DYNLOADBALFAC": (
                        'e.g. "phase1 1.0" -- this is what DECLARES the '
                        "phases; a phase named in PARTICLES but not here "
                        "aborts"
                    ),
                    "PHASE_TO_MATERIAL_ID": (
                        'e.g. "phase1 1" -- maps each phase to a MAT id'
                    ),
                    "TIMESTEP": "explicit step; see the critical-step pitfall",
                    "PARTICLE_WALL_SOURCE": (
                        '"NoParticleWall" (default), "DiscretCondition" '
                        '(a DESIGN SURFACE PARTICLE WALL on a structural '
                        'mesh) or "BoundingBox" (six walls on the '
                        "DOMAINBOUNDINGBOX)"
                    ),
                    "RIGID_BODY_MOTION": (
                        "bool; when true a rigidphase must exist and its "
                        "particles carry a RIGIDCOLOR token"
                    ),
                },
                "PARTICLE DYNAMIC/DEM": {
                    "MAX_RADIUS": "mandatory in practice -- the default 0.0 aborts",
                    "MIN_RADIUS": "optional floor, must not exceed MAX_RADIUS",
                    "MAX_VELOCITY": "needed by the automatic stiffness path",
                    "NORMALCONTACTLAW": (
                        "NormalLinearSpring (default) | NormalLinearSpringDamp | "
                        "NormalHertz | NormalLeeHerrmann | NormalKuwabaraKono | "
                        "NormalTsuji"
                    ),
                    "NORMAL_STIFF": "explicit stiffness; mutually exclusive with REL_PENETRATION",
                    "REL_PENETRATION": "automatic stiffness; needs MAX_VELOCITY",
                    "COEFF_RESTITUTION": "required by NormalLinearSpringDamp",
                    "NORMAL_DAMP": "required by LeeHerrmann / KuwabaraKono / Tsuji",
                    "TANGENTIALCONTACTLAW": "NoTangentialContact (default) | TangentialLinSpringDamp",
                    "ROLLINGCONTACTLAW": "NoRollingContact (default) | RollingViscous | RollingCoulomb",
                    "ADHESIONLAW": "NoAdhesion (default) | AdhesionVdWDMT | AdhesionRegDMT",
                    "INITIAL_RADIUS": (
                        "RadiusFromParticleMaterial (default) | "
                        "RadiusFromParticleInput | NormalRadiusDistribution | "
                        "LogNormalRadiusDistribution"
                    ),
                    "TENSION_CUTOFF": (
                        "default true; clamps the normal contact force to "
                        "compression only"
                    ),
                },
            },
            "materials": {
                "MAT_ParticleDEM": (
                    "INITRADIUS, INITDENSITY.  That is the whole material. "
                    "Mass follows as INITDENSITY * (4/3) pi r^3 and inertia "
                    "as 0.4 m r^2."
                ),
                "MAT_ParticleWallDEM": (
                    "FRICT_COEFF_TANG, FRICT_COEFF_ROLL, "
                    "ADHESION_SURFACE_ENERGY -- all optional, all defaulting "
                    "to -1.0.  Referenced from the MAT entry of a DESIGN "
                    "SURFACE PARTICLE WALL condition."
                ),
            },
            "contact_laws": {
                "normal": ["NormalLinearSpring", "NormalLinearSpringDamp",
                           "NormalHertz", "NormalLeeHerrmann",
                           "NormalKuwabaraKono", "NormalTsuji"],
                "tangential": ["NoTangentialContact", "TangentialLinSpringDamp"],
                "rolling": ["NoRollingContact", "RollingViscous", "RollingCoulomb"],
                "adhesion": ["NoAdhesion", "AdhesionVdWDMT", "AdhesionRegDMT"],
            },
            "time_integration": ["VelocityVerlet (default)", "SemiImplicitEuler"],
            "result_test_quantities": [
                "posx", "posy", "posz", "velx", "vely", "velz",
                "accx", "accy", "accz", "angvelx", "angvely", "angvelz",
                "radius", "density", "pressure", "temperature",
                "tempgradx", "tempgrady", "tempgradz", "pd_damage_phi",
            ],

            "pitfalls": [
                (
                    "[Input] Every mechanical contact property is a key "
                    "of the PARTICLE DYNAMIC/DEM section, never of the "
                    "material. MAT_ParticleDEM accepts exactly two "
                    "parameters, INITRADIUS and INITDENSITY, and both "
                    "are required with no default; YOUNG_MODULUS, "
                    "POISSON_RATIO, NORMAL_STIFF, COEFF_RESTITUTION and "
                    "the friction coefficients all live in the section. "
                    "Signal: putting a contact property in the material "
                    "does not warn about DEM at all. You get the generic "
                    "input-parser abort, \"Failed to match specification "
                    "in section 'MATERIALS'. The error was:\" from "
                    "global_data/4C_global_data_read.cpp followed by "
                    "\"Could not match this input\" — and then by a "
                    "candidate list with one block per material 4C "
                    "knows, which is every material in the code, each "
                    "block re-echoing your MAT_ParticleDEM entry under "
                    "\"The following data remains unused\". No line in "
                    "it says the key is unknown, and MAT_ParticleDEM is "
                    "not itself among the candidates it reports, so "
                    "scanning the wall for your material's name finds "
                    "nothing. Read any 'Failed to match specification in "
                    "section' abort on a particle deck as meaning the "
                    "material has no such parameter, check the entry "
                    "against the two-key list above, and move the key "
                    "into PARTICLE DYNAMIC/DEM. Note the "
                    "consequence for units: because the material carries "
                    "no stiffness, nothing in a DEM deck ties the contact "
                    "stiffness to a Young's modulus unless you set "
                    "YOUNG_MODULUS yourself for the laws that read it "
                    "(rolling-viscous and adhesion). (Verified by "
                    "execution 2026-08-07.)"
                ),
                (
                    "[Input] MAX_RADIUS has no usable default and is "
                    "effectively mandatory: it defaults to 0.0 and the "
                    "DEM setup rejects a non-positive maximum outright. "
                    "Signal: 'non-positive maximum allowed particle "
                    "radius!' from particle/src/interaction/"
                    "4C_particle_interaction_dem.cpp, before the first "
                    "step. Its partner MIN_RADIUS does default usefully "
                    "to 0.0, but if you set it above MAX_RADIUS you get "
                    "'minimum allowed particle radius larger than maximum "
                    "allowed particle radius!' from the same file. Both "
                    "bounds are checked for every INITIAL_RADIUS option, "
                    "not only the distributions, so every DEM deck needs "
                    "MAX_RADIUS whatever else it does. (Verified by "
                    "execution 2026-08-07.)"
                ),
                (
                    "[Input] The normal contact stiffness is an "
                    "exclusive-or, and BOTH ways of getting it wrong "
                    "produce the same message. Either set NORMAL_STIFF "
                    "and leave REL_PENETRATION alone, or set "
                    "REL_PENETRATION together with MAX_VELOCITY and leave "
                    "NORMAL_STIFF alone — 4C then derives the stiffness "
                    "from the maximum density, the maximum radius and "
                    "those two numbers. Signal: setting both, setting "
                    "neither, or setting REL_PENETRATION without "
                    "MAX_VELOCITY all abort with the single sentence "
                    "'specify either the relative penetration along with "
                    "the maximum velocity, or the normal stiffness, but "
                    "neither both nor none of them!' from "
                    "particle/src/interaction/"
                    "4C_particle_interaction_dem_contact_normal.cpp. "
                    "Because the message is identical in all three cases "
                    "it does not tell you which half you got wrong; check "
                    "the pair. (Verified by execution 2026-08-07.)"
                ),
                (
                    "[Numerical] The DEM critical time step IS computed, "
                    "from the smallest particle mass and the critical "
                    "normal stiffness, and it is checked on every single "
                    "step — but exceeding it is a WARNING and nothing "
                    "else. There is no abort, no step-size reduction, and "
                    "no non-zero exit from the time loop. Signal: 'Warning: "
                    "time step <dt> larger than critical time step "
                    "<dtcrit>!' from particle/src/interaction/"
                    "4C_particle_interaction_dem_contact.cpp, repeated "
                    "once per step and on every rank, so a long unstable "
                    "run buries the rest of the log. The run then finishes "
                    "and only the RESULT DESCRIPTION verdicts reveal that "
                    "the answer is wrong. Treat the first occurrence of "
                    "that line as fatal yourself: grep for 'critical time "
                    "step' and halve TIMESTEP until it stops appearing. "
                    "Note the printed limit already carries 4C's own "
                    "safety factor and is smaller when a tangential "
                    "contact law is active. (Verified by execution "
                    "2026-08-07.)"
                ),
                (
                    "[Numerical] INITIAL_RADIUS: RadiusFromParticleInput "
                    "takes the radius from a RAD token on each PARTICLES "
                    "line, and a line without that token keeps radius "
                    "zero. Signal: there is no diagnostic whatsoever — the "
                    "MIN_RADIUS/MAX_RADIUS bounds pass, because the "
                    "default MIN_RADIUS is 0.0 and zero is inside "
                    "[0, MAX_RADIUS] — and the run dies on a raw "
                    "SIGFPE -- OpenMPI's handler names the floating point "
                    "exception, signal 8, with signal code "
                    "floating-point divide-by-zero, 3 -- inside "
                    "ParticleInteractionDEM::compute_acceleration, exit "
                    "status 136, before the first printed time step and "
                    "with no 4C error block at all. The mechanism is that "
                    "mass is INITDENSITY * (4/3) pi r^3, so r = 0 gives "
                    "zero mass and the acceleration divides by it. Either "
                    "write RAD on every particle line or leave "
                    "INITIAL_RADIUS at its RadiusFromParticleMaterial "
                    "default, and set MIN_RADIUS above zero so the bounds "
                    "check can catch the case. (Verified by execution "
                    "2026-08-07.)"
                ),
                (
                    "[Numerical] The random radius options do not sample "
                    "the distribution you asked for. "
                    "NormalRadiusDistribution and "
                    "LogNormalRadiusDistribution draw once per particle "
                    "and then CLAMP the draw into [MIN_RADIUS, "
                    "MAX_RADIUS]; there is no rejection and no redraw, so "
                    "everything in the tails piles up exactly on the two "
                    "bounds and the realised distribution is the truncated "
                    "one with two point masses. Signal: 4C says nothing at "
                    "all about clamping — the tell is that a result test "
                    "on QUANTITY 'radius' returns a value bit-equal to "
                    "MIN_RADIUS or MAX_RADIUS rather than a generic "
                    "number, and 4C's own regression decks expect exactly "
                    "that for most of their particles. Choose "
                    "RADIUSDISTRIBUTION_SIGMA small against the bound "
                    "half-width, or widen the bounds, and check how many "
                    "radii land on a bound. Set RANDSEED in PROBLEM TYPE "
                    "or the draw changes run to run. (Verified by "
                    "execution 2026-08-07.)"
                ),
                (
                    "[Input] The tangential law is only implemented on "
                    "top of the two LINEAR normal laws. Combining "
                    "TANGENTIALCONTACTLAW: TangentialLinSpringDamp with "
                    "NormalHertz, NormalLeeHerrmann, NormalKuwabaraKono "
                    "or NormalTsuji is refused, because the tangential "
                    "stiffness is derived as a fixed multiple of the "
                    "normal one. Signal: 'tangential contact law only "
                    "valid with linear normal contact law!' from "
                    "particle/src/interaction/"
                    "4C_particle_interaction_dem_contact.cpp, at setup. "
                    "Rolling laws have no such restriction. If you need "
                    "friction, stay on NormalLinearSpring or "
                    "NormalLinearSpringDamp. (Verified by execution "
                    "2026-08-07.)"
                ),
                (
                    "[Input] A friction coefficient of zero is rejected, "
                    "not honoured, so 'frictionless' cannot be expressed "
                    "by setting the coefficient. Both FRICT_COEFF_TANG "
                    "and FRICT_COEFF_ROLL are validated as strictly "
                    "positive once their law is on, and their default of "
                    "-1.0 means omitting them fails the same test. Signal: "
                    "'invalid input parameter FRICT_COEFF_TANG for this "
                    "kind of contact law!' from "
                    "4C_particle_interaction_dem_contact_tangential.cpp, "
                    "or 'invalid input parameter FRICT_COEFF_ROLL for "
                    "this kind of contact law!' from "
                    "4C_particle_interaction_dem_contact_rolling.cpp — "
                    "both worded as if the value were malformed rather "
                    "than out of range. To switch friction off, remove "
                    "TANGENTIALCONTACTLAW / ROLLINGCONTACTLAW instead. "
                    "POISSON_RATIO is validated by the same two handlers "
                    "and by adhesion, never by a normal contact law: "
                    "'invalid input parameter POISSON_RATIO (expected in "
                    "range ]-1.0; 0.5])!'. (Verified by execution "
                    "2026-08-07.)"
                ),
                (
                    "[Input] Particle-to-wall friction and adhesion come "
                    "from MAT_ParticleWallDEM on the wall surface, not "
                    "from the PARTICLE DYNAMIC/DEM section, and the two "
                    "sets do not talk to each other. The section values "
                    "govern particle-to-particle contact only. Signal: "
                    "the asymmetry is the giveaway — zero is illegal in "
                    "the section and aborts, while FRICT_COEFF_TANG: 0.0 "
                    "in MAT_ParticleWallDEM is accepted silently and "
                    "simply makes that wall frictionless, changing the "
                    "answer with no message. Pointing the condition's MAT "
                    "at any other material gives 'cast to "
                    "Mat::ParticleWallMaterialDEM failed!' from "
                    "4C_particle_interaction_dem_contact.cpp — a message "
                    "that names a C++ class, not the input key you got "
                    "wrong. MAT: -1 (no wall material at all) is legal "
                    "and is what 4C's own coupled decks use, but only "
                    "while no tangential, rolling or adhesion law is "
                    "active; with adhesion on the cast is unconditional. "
                    "(Verified by execution 2026-08-07.)"
                ),
                (
                    "[Input] The adhesion block is a cluster of "
                    "individually mandatory keys whose defaults are "
                    "negative sentinels, and one of them wants a NEGATIVE "
                    "value. Turning ADHESIONLAW on makes "
                    "ADHESION_SURFACE_ENERGY and ADHESION_DISTANCE "
                    "required, AdhesionVdWDMT additionally requires "
                    "ADHESION_HAMAKER, and ADHESION_MAX_CONTACT_PRESSURE "
                    "is a pressure limit expressed as a negative number. "
                    "Signal: omitting a key produces a message describing "
                    "the sentinel value rather than the omission — 'negative "
                    "adhesion distance!' and 'negative hamaker constant!' "
                    "from 4C_particle_interaction_dem_adhesion.cpp and "
                    "..._adhesion_law.cpp both mean you did not set "
                    "the key. The sign trap reports itself as 'positive "
                    "adhesion maximum contact pressure!', i.e. a positive "
                    "value is the error. ADHESION_DISTANCE: 0.0 is "
                    "accepted, is not a no-op, and changes the answer "
                    "silently. (Verified by execution 2026-08-07.)"
                ),
                (
                    "[Numerical] Contact parameters that belong to a law "
                    "you are not using are read, accepted and ignored — "
                    "there is no unused-parameter warning anywhere in the "
                    "DEM stack. COEFF_RESTITUTION on the plain "
                    "NormalLinearSpring is the clean example: the law has "
                    "no damping term to restitute, and the deck runs "
                    "bit-identically with and without the key. Signal: "
                    "none — no warning, no change in any result-test "
                    "verdict, not even in the last printed digit. The "
                    "consequence is that a deck can look tuned while the "
                    "tuning parameter is inert, exactly as "
                    "BACKGROUNDPRESSURE is inert on the SPH side without "
                    "a transport velocity. Verify a parameter matters by "
                    "diffing the verdict lines of two runs, not by "
                    "reading the input back. (Verified by execution "
                    "2026-08-07.)"
                ),
            ],

            "typical_experiments": [
                {
                    "name": "Two-particle normal impact (1D)",
                    "description": (
                        "Two spheres approach on a line and rebound. The "
                        "smallest thing that exercises a normal contact "
                        "law, and the shape of most of 4C's own DEM "
                        "regression decks."
                    ),
                },
                {
                    "name": "Oblique impact with friction (2D)",
                    "description": (
                        "Adds TangentialLinSpringDamp so the tangential "
                        "history and the Coulomb bound are exercised; "
                        "needs POISSON_RATIO and FRICT_COEFF_TANG."
                    ),
                },
                {
                    "name": "Sphere rolling on a wall (2D)",
                    "description": (
                        "Rolling resistance against a DESIGN SURFACE "
                        "PARTICLE WALL, which is where the wall material "
                        "enters."
                    ),
                },
                {
                    "name": "Adhesive contact (1D)",
                    "description": (
                        "van-der-Waals-DMT or regularised DMT pull-off "
                        "between two spheres; the tension side of the "
                        "force curve, so TENSION_CUTOFF matters."
                    ),
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "normal_impact_1d",
                "description": (
                    "Two DEM spheres colliding head-on in 1D. Minimal "
                    "runnable-shaped DEM deck: one phase, one material, "
                    "one normal contact law."
                ),
            },
        ]

    def get_template(self, variant: str = "normal_impact_1d") -> str:
        if variant == "normal_impact_1d":
            return self._template_normal_impact_1d()
        raise ValueError(
            f"Unknown variant {variant!r} for {self.module_key}.  "
            f"Available: {[v['name'] for v in self.list_variants()]}"
        )

    @staticmethod
    def _template_normal_impact_1d() -> str:
        """Two spheres colliding on a line — the minimal DEM shape."""
        return textwrap.dedent("""\
            # 1D DEM normal impact — two spheres approaching on the x axis.
            #
            # FORMAT TEMPLATE — every <...> is a placeholder. Choose the
            # radius, density, stiffness and time step for your problem.
            #
            # No output section is needed or possible: 4C writes the
            # particle .pvd/.vtu series unconditionally at RESULTSEVERY.
            # There is no IO/RUNTIME VTK OUTPUT/PARTICLES section and
            # writing one is a hard parse error.

            PROBLEM TYPE:
              PROBLEMTYPE: "Particle"

            IO:
              STDOUTEVERY: <stdout_frequency>
              VERBOSITY: "Standard"

            BINNING STRATEGY:
              # Must be >= the DEM interaction distance, which is
              # 2 * MAX_RADIUS (plus ADHESION_DISTANCE when adhesion is on).
              BIN_SIZE_LOWER_BOUND: <bin_size>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            PARTICLE DYNAMIC:
              INTERACTION: "DEM"
              RESULTSEVERY: <output_frequency>
              RESTARTEVERY: <restart_frequency>
              TIMESTEP: <dt>
              NUMSTEP: <total_steps>
              # PHASE_TO_DYNLOADBALFAC DECLARES the phases. A phase used in
              # PARTICLES but missing here aborts with
              #   particle type '<name>' of initial particle not defined!
              PHASE_TO_DYNLOADBALFAC: "phase1 1.0"
              # PHASE_TO_MATERIAL_ID maps each phase to a MAT id. Omitting
              # it segfaults with no message at all — see the pitfalls.
              PHASE_TO_MATERIAL_ID: "phase1 1"

            PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS:
              INITIAL_VELOCITY_FIELD: "phase1 1"

            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<vx(x,y,z,t)>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

            PARTICLE DYNAMIC/DEM:
              # MAX_RADIUS has no usable default (0.0 aborts).
              MAX_RADIUS: <max_radius>
              MAX_VELOCITY: <max_expected_velocity>
              # Stiffness is an exclusive-or: EITHER NORMAL_STIFF ...
              NORMAL_STIFF: <normal_stiffness>
              # ... OR REL_PENETRATION together with MAX_VELOCITY.
              # Setting both, or neither, aborts.

            MATERIALS:
              # Geometry and mass only. Contact properties are section keys.
              - MAT: 1
                MAT_ParticleDEM:
                  INITRADIUS: <radius>
                  INITDENSITY: <density>

            RESULT DESCRIPTION:
              - PARTICLE:
                  ID: 0                 # ids follow PARTICLES file order from 0
                  QUANTITY: "posx"      # posx velx radius ... (see knowledge)
                  VALUE: <expected_value>
                  TOLERANCE: <tolerance>

            # A YAML list of STRINGS, not of mappings:
            #   TYPE <phase> POS <x> <y> <z> [RAD <r>] [RIGIDCOLOR <c>]
            PARTICLES:
              - "TYPE phase1 POS <x1> <y1> <z1>"
              - "TYPE phase1 POS <x2> <y2> <z2>"
        """)

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of DEM parameters.

        Expected keys in *params* (all optional):
            max_radius, min_radius, max_velocity, normal_stiff,
            rel_penetration, normal_contact_law, tangential_contact_law,
            rolling_contact_law, adhesion_law, frict_coeff_tang,
            frict_coeff_roll, poisson_ratio, coeff_restitution,
            normal_damp, adhesion_distance, adhesion_surface_energy,
            adhesion_hamaker, adhesion_max_contact_pressure,
            initial_radius, radiusdistribution_sigma,
            bin_size_lower_bound, density, dt
        """
        warnings: list[str] = []

        def _set(key: str) -> bool:
            v = params.get(key)
            return v is not None and v > 0.0

        max_radius = params.get("max_radius")
        min_radius = params.get("min_radius")

        if max_radius is None or max_radius <= 0.0:
            warnings.append(
                "ERROR: MAX_RADIUS must be > 0 (its default 0.0 aborts with "
                "'non-positive maximum allowed particle radius!')."
            )
        if (min_radius is not None and max_radius is not None
                and min_radius > max_radius):
            warnings.append(
                "ERROR: MIN_RADIUS > MAX_RADIUS aborts with 'minimum allowed "
                "particle radius larger than maximum allowed particle radius!'."
            )

        # The stiffness exclusive-or.
        stiff = _set("normal_stiff")
        relpen = _set("rel_penetration")
        vmax = _set("max_velocity")
        if stiff and relpen:
            warnings.append(
                "ERROR: NORMAL_STIFF and REL_PENETRATION are mutually "
                "exclusive; set exactly one."
            )
        elif not stiff and not relpen:
            warnings.append(
                "ERROR: set either NORMAL_STIFF, or REL_PENETRATION together "
                "with MAX_VELOCITY. Neither is not allowed."
            )
        elif relpen and not vmax:
            warnings.append(
                "ERROR: REL_PENETRATION needs MAX_VELOCITY > 0 to derive the "
                "normal stiffness."
            )

        normal = (params.get("normal_contact_law") or "")
        tangential = (params.get("tangential_contact_law") or "")
        rolling = (params.get("rolling_contact_law") or "")
        adhesion = (params.get("adhesion_law") or "")

        linear_normal = normal in ("", "NormalLinearSpring",
                                   "NormalLinearSpringDamp")
        if tangential and tangential != "NoTangentialContact":
            if not linear_normal:
                warnings.append(
                    f"ERROR: TANGENTIALCONTACTLAW {tangential!r} needs a "
                    f"linear normal law; {normal!r} aborts with 'tangential "
                    f"contact law only valid with linear normal contact law!'."
                )
            if not _set("frict_coeff_tang"):
                warnings.append(
                    "ERROR: FRICT_COEFF_TANG must be > 0 when a tangential "
                    "law is on. Zero is rejected — remove the law instead."
                )
        if rolling and rolling != "NoRollingContact":
            if not _set("frict_coeff_roll"):
                warnings.append(
                    "ERROR: FRICT_COEFF_ROLL must be > 0 when a rolling law "
                    "is on. Zero is rejected — remove the law instead."
                )
            if rolling == "RollingViscous" and not _set("max_velocity"):
                warnings.append(
                    "ERROR: RollingViscous needs MAX_VELOCITY > 0."
                )

        nu = params.get("poisson_ratio")
        needs_nu = ((tangential and tangential != "NoTangentialContact")
                    or (rolling and rolling != "NoRollingContact")
                    or (adhesion and adhesion != "NoAdhesion"))
        if needs_nu and (nu is None or nu <= -1.0 or nu > 0.5):
            warnings.append(
                "ERROR: POISSON_RATIO must lie in ]-1.0, 0.5] for tangential, "
                "rolling and adhesion laws (default -1.0 fails)."
            )

        if normal == "NormalLinearSpringDamp" and not _set("coeff_restitution"):
            warnings.append(
                "ERROR: NormalLinearSpringDamp requires COEFF_RESTITUTION > 0."
            )
        if normal in ("NormalLeeHerrmann", "NormalKuwabaraKono", "NormalTsuji"):
            if not _set("normal_damp"):
                warnings.append(
                    f"ERROR: {normal} requires NORMAL_DAMP > 0."
                )
        if (normal == "NormalLinearSpring"
                and params.get("coeff_restitution") is not None):
            warnings.append(
                "WARNING: COEFF_RESTITUTION is inert on NormalLinearSpring — "
                "the law has no damping term. The run is bit-identical "
                "without it, and 4C emits no unused-parameter warning."
            )

        if adhesion and adhesion != "NoAdhesion":
            if not _set("adhesion_surface_energy"):
                warnings.append(
                    "ERROR: ADHESION_SURFACE_ENERGY must be > 0 once "
                    "ADHESIONLAW is on."
                )
            ad = params.get("adhesion_distance")
            if ad is None or ad < 0.0:
                warnings.append(
                    "ERROR: ADHESION_DISTANCE must be >= 0 (its default -1.0 "
                    "aborts with 'negative adhesion distance!')."
                )
            if adhesion == "AdhesionVdWDMT" and not _set("adhesion_hamaker"):
                warnings.append(
                    "ERROR: AdhesionVdWDMT requires ADHESION_HAMAKER > 0 "
                    "(default -1.0 aborts with 'negative hamaker constant!')."
                )
            p = params.get("adhesion_max_contact_pressure")
            if p is not None and p > 0.0:
                warnings.append(
                    "ERROR: ADHESION_MAX_CONTACT_PRESSURE must be negative "
                    "or zero; a positive value aborts with 'positive adhesion "
                    "maximum contact pressure!'."
                )

        init_radius = params.get("initial_radius") or ""
        if init_radius in ("NormalRadiusDistribution",
                           "LogNormalRadiusDistribution"):
            if not _set("radiusdistribution_sigma"):
                warnings.append(
                    "ERROR: RADIUSDISTRIBUTION_SIGMA must be set and positive "
                    "for a radius distribution."
                )
            if min_radius is None or min_radius <= 0.0:
                warnings.append(
                    "WARNING: with MIN_RADIUS at 0.0 a random draw can be "
                    "clamped to a zero radius, which gives zero mass and a "
                    "silent divide-by-zero (SIGFPE) rather than an error."
                )
            else:
                warnings.append(
                    "WARNING: random radii are CLAMPED into [MIN_RADIUS, "
                    "MAX_RADIUS], not redrawn, so the tails pile up exactly "
                    "on the bounds. Check how many radii equal a bound."
                )
        if init_radius == "RadiusFromParticleInput":
            warnings.append(
                "WARNING: RadiusFromParticleInput needs a RAD token on EVERY "
                "PARTICLES line. A missing one leaves radius 0, passes the "
                "bounds check when MIN_RADIUS is 0, and crashes with a raw "
                "SIGFPE (exit 136) and no 4C message."
            )

        bin_size = params.get("bin_size_lower_bound")
        if bin_size is not None and max_radius is not None and max_radius > 0:
            needed = 2.0 * max_radius + (params.get("adhesion_distance") or 0.0)
            if bin_size < needed:
                warnings.append(
                    f"ERROR: BIN_SIZE_LOWER_BOUND = {bin_size} is below the "
                    f"DEM interaction distance {needed} (2*MAX_RADIUS plus "
                    f"ADHESION_DISTANCE). 4C aborts with 'the particle "
                    f"interaction distance is larger than the minimal bin "
                    f"size'."
                )

        return warnings
