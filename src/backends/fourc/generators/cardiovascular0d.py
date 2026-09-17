"""0-D cardiovascular / windkessel models generator for 4C.

Covers lumped-parameter heart models, windkessel afterload, closed-loop circulation.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class Cardiovascular0DGenerator(BaseGenerator):
    """Generator for 0-D cardiovascular models in 4C."""

    module_key = "cardiovascular0d"
    display_name = "0-D Cardiovascular (Windkessel)"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Lumped-parameter (0-D) cardiovascular models: windkessel afterload, "
                "closed-loop circulation, time-varying elastance heart model.  "
                "Coupled to 3-D fluid or structure via surface conditions."
            ),
            "models": {
                "windkessel_3element": "R_p, C, R_d — proximal resistance, compliance, distal resistance",
                "windkessel_4element": "R_p, L, C, R_d — adds inductance",
                "heart_time_varying_elastance": "E(t) model with active contraction",
                "closed_loop": "Full circulation: heart + arterial + venous + pulmonary",
            },
            "coupling": [
                "DESIGN SURF CARDIOVASCULAR0D CONDITIONS",
                "Coupled to 3-D fluid outflow or structural cavity volume",
            ],
            "applications": ["cardiac simulation", "hemodynamics", "afterload modeling",
                             "valve simulation", "ventricular assist device"],
            "pitfalls": [
                (
                    "[Input] Windkessel parameters live in the DESIGN SURF "
                    "CARDIOVASCULAR 0D N-ELEMENT WINDKESSEL CONDITIONS block, "
                    "one entry per cavity with C, R_p, Z_c, L, p_ref and p_0, "
                    "and they must match the vascular impedance. Signal: none - "
                    "arbitrary values are accepted without comment. The effect "
                    "is local to the cavity you mis-tuned, and compliance bites "
                    "much harder than resistance: a stiff Windkessel refuses "
                    "volume and nearly freezes the wall it loads. Tune from "
                    "Z_terminal = rho*c/A and a compliance estimate. (Audit "
                    "2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Time-varying elastance is set by E_at_min/E_at_max "
                    "and E_v_min/E_v_max per side plus the activation curves "
                    "Atrium_act_curve_l/r and Ventricle_act_curve_l/r, which "
                    "point at FUNCT blocks carrying the systolic and diastolic "
                    "timing. There is NO ELASTANCE_FUNCTION parameter and no "
                    "T_S or T_D. Signal: adding ELASTANCE_FUNCTION fails to "
                    "match the SYS-PUL CIRCULATION PARAMETERS section; setting "
                    "every max equal to its min is accepted silently and does "
                    "kill the pumping - ventricular pressure drops and the "
                    "inflow reverses. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
                (
                    "[API] Cavity volume is a surface integral over the CLOSED "
                    "cavity boundary, so the DSURF-NODE TOPOLOGY of the "
                    "cardiovascular0d surface must contain the whole boundary. "
                    "Signal: an open boundary gives a wrong volume, and 4C "
                    "prints it - the 'N V:' line under 'Cardiovascular0D output "
                    "id N' becomes the divergence-theorem integral over "
                    "whatever faces remain and stops changing between steps. "
                    "Downstream NOX usually gives up with \"The nonlinear solver "
                    "did not converge!\", and that message says nothing about "
                    "the surface, so the frozen volume is the real clue. (Audit "
                    "2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] The 0D compartments are seeded by the p_*_0 "
                    "parameters (p_at_l_0, p_v_l_0, p_ar_sys_0, ...). Signal: "
                    "none - zeroing them is accepted and every step converges "
                    "cleanly, but the model then spends cardiac cycles climbing "
                    "out of a warm-up transient and a single-cycle answer is "
                    "far from the settled one. There is no periodicity or "
                    "'still in warm-up' check, so seed physiological diastolic "
                    "pressures or run enough cycles and verify periodicity "
                    "yourself. (Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] There is no Cardiovascular0D PROBLEMTYPE; the 0D "
                    "model is always carried by a Structure problem through "
                    "CARDIOVASCULAR 0D-STRUCTURE COUPLING, and there is no "
                    "Fluid route. Signal: PROBLEMTYPE: \"Cardiovascular0D\" fails "
                    "to match and the parser prints the full enum, which does "
                    "not contain it. A standalone 0D circulation IS supported, "
                    "and that is how 4C's own closed-loop heart deck is built: "
                    "PROBLEMTYPE Structure with VENTRICLE_MODEL \"0D\" and a "
                    "single dummy SOLID element whose displacements are all "
                    "zero. (Audit 2026-08-06, verified by execution.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "windkessel_3d", "description": "3-element windkessel coupled to 3D"}]

    def get_template(self, variant: str = "windkessel_3d") -> str:
        if variant not in ("windkessel_3d", "default"):
            raise ValueError(f"Unknown variant {variant!r}")
        from ..inline_mesh import matched_cardiovascular0d_windkessel_input
        return matched_cardiovascular0d_windkessel_input()

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
