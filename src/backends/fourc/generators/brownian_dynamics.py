"""Brownian dynamics generator for 4C.

Covers thermal fluctuations in beam/fiber networks (e.g., biopolymer networks).
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class BrownianDynamicsGenerator(BaseGenerator):
    """Generator for Brownian dynamics of fiber networks in 4C."""

    module_key = "brownian_dynamics"
    display_name = "Brownian Dynamics (Fiber Networks)"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Brownian dynamics for thermal fluctuations in beam/fiber networks.  "
                "Used for modeling biopolymer networks (actin, collagen) at the "
                "mesoscale where thermal forces are significant."
            ),
            "elements": ["BEAM3R LINE2 (Simo-Reissner beam with Brownian forces)"],
            "physics": {
                "thermal_forces": "Random forces from Fluctuation-Dissipation theorem",
                "viscous_drag": "Stokes drag on beam segments",
                "cross_links": "Beam-to-beam coupling via penalty/Lagrange",
            },
            "applications": ["actin network mechanics", "collagen fiber networks",
                             "polymer rheology", "cytoskeleton modeling"],
            "pitfalls": [
                (
                    "[Numerical] Time step must be SMALL "
                    "relative to the Brownian relaxation "
                    "time tau_B = ksi / kBT, and it has to "
                    "be chosen against that physical scale "
                    "because you CANNOT choose it by a "
                    "step-refinement study. Signal: with KT "
                    "= 0 the same deck step-converges "
                    "normally — halving dt leaves a node's "
                    "transverse displacement unchanged to "
                    "machine precision.  With KT > 0 and a "
                    "fixed RANDSEED, halving dt draws a "
                    "different realisation rather than a "
                    "better answer: the displacement moves "
                    "by more than its own magnitude and can "
                    "change sign.  So a 'halve dt and see "
                    "if it settles' check is meaningless "
                    "here and will let you accept any dt; "
                    "compute tau_B from VISCOSITY and KT "
                    "instead.  A single realisation also "
                    "cannot show the mean-square "
                    "displacement scaling — that needs an "
                    "ensemble over seeds. (Audit "
                    "2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] The temperature parameter is "
                    "KT in the BROWNIAN DYNAMICS section "
                    "and it controls the fluctuation "
                    "magnitude in the Langevin noise term. "
                    "Its default is 0.0, and BROWNDYNPROB: "
                    "true does NOT imply a temperature — "
                    "several upstream Polymer_Network decks "
                    "set BROWNDYNPROB and VISCOSITY with no "
                    "KT at all and are purely "
                    "deterministic runs. Signal: with KT "
                    "absent the transverse displacement of "
                    "a free filament node comes back at "
                    "machine zero, exactly, and 4C emits no "
                    "warning that the thermal forcing is "
                    "off — the deck looks like a Brownian "
                    "simulation and contains no thermal "
                    "physics.  Too-large KT instead "
                    "produces unphysical filament "
                    "stretching beyond bond extension.  "
                    "Note the units follow the rest of the "
                    "deck, so scale KT to the same system "
                    "as YOUNG and VISCOSITY rather than "
                    "pasting an SI value. (Audit "
                    "2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] Cross-link stiffness "
                    "dramatically affects network "
                    "response, and the knob is not on "
                    "MAT_Crosslinker itself: that material "
                    "points at a beam material through "
                    "MATNUM, and THAT beam's YOUNG is "
                    "k_xl.  Keep it a separate MAT entry "
                    "from the filament material so you can "
                    "vary one without the other. Signal: "
                    "the usable range is bracketed by two "
                    "hard aborts in different subsystems, "
                    "neither of which mentions stiffness. "
                    "Too soft, the linker stretches past "
                    "half the periodic box and 4C aborts "
                    "with `You are trying to set the "
                    "binding spot positions of this "
                    "crosslinker in at least one "
                    "direction` from beaminteraction/src/"
                    "crosslinking/4C_beaminteraction_"
                    "crosslinking_submodel_evaluator.cpp — "
                    "that is the end at which the network "
                    "deforms like a viscous fluid.  Too "
                    "stiff, the network "
                    "Newton stops converging: `The "
                    "nonlinear solver did not converge!` "
                    "from solver_nonlin_nox/"
                    "4C_solver_nonlin_nox_problem.cpp. "
                    "(Audit 2026-06-02; confirmed by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] Periodic boundary conditions "
                    "are needed for RVE analysis of "
                    "network rheology, and the knob is "
                    "PERIODICONOFF inside BINNING "
                    "STRATEGY, next to DOMAINBOUNDINGBOX "
                    "— NOT a 'DESIGN PERIODIC CONDITIONS' "
                    "block, which appears in no upstream "
                    "Polymer_Network deck.  The RVE drive "
                    "runs through PERIODIC BOUNDINGBOX "
                    "ELEMENTS plus DESIGN SURF DIRICH "
                    "CONDITIONS on the box corner nodes. "
                    "Signal: dropping PERIODICONOFF is not "
                    "a few-percent edge effect — the box "
                    "deformation then never reaches the "
                    "network and the filament response "
                    "comes back EXACTLY zero in every "
                    "component, while the run exits "
                    "normally with no warning.  A 'within "
                    "20%' screen passes that; a "
                    "zero-response check catches it "
                    "immediately. (Audit 2026-06-02; "
                    "corrected by execution 2026-08-06.)"
                ),
                (
                    "[Validation] A Brownian run is "
                    "DETERMINISTIC within one build and its "
                    "reference values are NOT PORTABLE between "
                    "builds. Both halves matter and they pull "
                    "in opposite directions. With RANDSEED "
                    "fixed, repeating the same deck on the "
                    "same binary reproduces every result to "
                    "the last printed digit, so a re-run tells "
                    "you nothing about statistical spread. The "
                    "random stream itself, however, depends on "
                    "the build, and on one and the same "
                    "binary SOME of 4C's own Brownian "
                    "regression decks reproduce the reference "
                    "values stored in them and others do not. "
                    "Signal: a RESULT DESCRIPTION on a "
                    "browndyn deck fails at the SCALE of the "
                    "answer rather than at roundoff — 'is "
                    "WRONG --> actresult=' differing in the "
                    "leading digits, not in the last ones — "
                    "while a non-Brownian beam deck passes on "
                    "the same binary, so the build is not "
                    "broken. That mixture is the point: a "
                    "failing Brownian reference value is not "
                    "evidence that YOUR deck is wrong. Before "
                    "debugging your own setup, run several of "
                    "4C's own beam*_browndyn_*.4C.yaml decks "
                    "unchanged and see how many reproduce. "
                    "Where they do not, the stored values have "
                    "to be regenerated on your build. Validate "
                    "against ensemble statistics over seeds, "
                    "never against a stored trajectory. "
                    "(Verified by execution 2026-08-07.)"
                ),
                (
                    "[Input] VISCOSITY has no usable default "
                    "and neither does BROWNDYNPROB, and "
                    "getting either wrong kills the run with "
                    "no message. VISCOSITY defaults to 0.0 and "
                    "the beam damping coefficients are "
                    "proportional to it, so the drag the "
                    "Langevin update divides by becomes zero. "
                    "BROWNDYNPROB defaults to false, and false "
                    "does not cleanly disable machinery a "
                    "structural deck has already been "
                    "configured for. Signal: neither produces "
                    "a 4C error block. Dropping either key "
                    "from a working deck kills the process with "
                    "SIGFPE and exit status 136. The only lines "
                    "you get are the MPI runtime's: OpenMPI's "
                    "handler in libopen-pal prints its "
                    "process-received-signal block naming the "
                    "floating point exception, signal 8, and the "
                    "floating-point divide-by-zero signal code, "
                    "3. That block is not 4C's and changes with "
                    "the MPI build, so do not grep for it; grep "
                    "for the exit status. There are "
                    "zero PROC 0 ERROR blocks, and nothing "
                    "naming VISCOSITY, BROWNDYNPROB or "
                    "damping. Set both explicitly in every "
                    "deck, and switch Brownian dynamics off by "
                    "removing the model from the structural "
                    "setup rather than by flipping the flag. "
                    "(Verified by execution 2026-08-07.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "brownian_3d", "description": "Brownian fiber network"}]

    def get_template(self, variant: str = "brownian_3d") -> str:
        # Not self-contained-runnable: Brownian dynamics of polymer
        # filaments needs a BEAM3R filament mesh inside a periodic box
        # plus a stochastic (statmech) integrator and a crosslinker
        # BINNING STRATEGY — all case-specific. Return a valid-YAML
        # reference stub (parses to a dict; documents what is required)
        # rather than a comment-only one-liner.
        return (
            "# =====================================================\n"
            "# 4C Brownian dynamics (variant: brownian_3d)\n"
            "# =====================================================\n"
            "# Not a self-contained runnable input. Requires:\n"
            "#   * a BEAM3R LINE2 filament mesh in a periodic box\n"
            "#   * BROWNIAN DYNAMICS section (thermal energy KT,\n"
            "#     damping, random seed)\n"
            "#   * STRUCTURAL DYNAMIC with a stochastic (statmech)\n"
            "#     integrator + a BINNING STRATEGY for crosslinkers\n"
            "#   * MAT_BeamReissnerElastHyper (filament cross-section)\n"
            "# Pitfalls (see knowledge() for the full set):\n"
            "#   * stochastic time step couples to KT and damping;\n"
            "#     too large breaks fluctuation-dissipation balance\n"
            "#   * results are statistical — one short run is not\n"
            "#     representative\n"
            "# =====================================================\n"
            "TITLE:\n"
            "  - \"4C Brownian dynamics reference stub\"\n"
            "PROBLEM TYPE:\n"
            "  PROBLEMTYPE: \"Structure\"\n"
        )

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
