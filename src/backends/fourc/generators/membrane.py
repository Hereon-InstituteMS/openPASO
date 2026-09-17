"""Membrane element generator for 4C.

Covers thin membrane/shell elements for structural analysis.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class MembraneGenerator(BaseGenerator):
    """Generator for membrane/thin shell problems in 4C."""

    module_key = "membrane"
    display_name = "Membrane Elements"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thin membrane elements for inflatable structures, fabric, "
                "biological tissue.  No bending stiffness — pure in-plane stress."
            ),
            "elements": {
                # Verified against `4C --parameters` and by execution: the
                # element type carries the node count, there is no bare
                # "MEMBRANE" type, and every key below is REQUIRED.
                "surface in 3D": [
                    "MEMBRANE3 TRI3 / MEMBRANE6 TRI6 / MEMBRANE4 QUAD4 / "
                    "MEMBRANE9 QUAD9",
                    "line form: '<id> MEMBRANE4 QUAD4 <n1..n4> MAT m "
                    "KINEM nonlinear THICK t STRESS_STRAIN plane_stress'",
                    "MEMBRANESCATRA3/4/6/9 (scalar-transport coupled)",
                ],
            },
            "materials": [
                "MAT_Membrane_ElastHyper (NUMMAT, MATIDS, DENS) wrapping "
                "ELAST_* sub-materials — this wrapper is mandatory",
                "MAT_Membrane_ActiveStrain",
                "NOT MAT_ElastHyper: the general 3D wrapper parses but the "
                "element refuses it with 'The material does not support the "
                "evaluation of membranes'",
            ],
            "pitfalls": [
                (
                    "[Numerical] Zero bending stiffness decides whether the "
                    "deck runs at all, not just how the answer looks. A FLAT "
                    "membrane patch whose reference configuration carries no "
                    "in-plane stress has an identically empty out-of-plane "
                    "block in its tangent — nothing fills it, because there is "
                    "no bending term. Signal: the run dies on SIGFPE -- the "
                    "MPI runtime names the floating point exception, signal 8 "
                    "-- raised inside libumfpack's "
                    "triangular solve, exit 136, no time step finalised and "
                    "NOT ONE 'PROC 0 ERROR in' line, so there is no 4C "
                    "diagnostic to read. Related: a membrane will not accept a "
                    "plate-style normal surface traction either — a DESIGN "
                    "SURF NEUMANN with the load in the third dof slot is "
                    "refused with 'membrane pressure on 1st dof only!' from "
                    "src/membrane/4C_membrane_evaluate.cpp. Give the sheet "
                    "in-plane stretch (prescribed edge displacement) and the "
                    "same mesh solves. (Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] What stabilises a membrane is IN-PLANE "
                    "STRESS, and on a FLAT sheet neither of the two usual "
                    "recipes provides any. Adding an orthopressure DESIGN SURF "
                    "NEUMANN does not help; nor does STRUCTURAL DYNAMIC "
                    "PRESTRESS: 'MULF'. Both are loads, and neither puts "
                    "stress into the reference configuration. Signal: with "
                    "either of them and no prescribed in-plane stretch, the "
                    "run dies at step 0 with a floating-point exception inside "
                    "UMFPACK and no 4C-level message; prescribing an edge "
                    "stretch on the identical mesh makes it converge. Pressure "
                    "DOES stabilise once the surface is CURVED — a cylindrical "
                    "membrane driven by orthopressure alone runs — because "
                    "there the pressure generates in-plane stress. Direct LU "
                    "never prints the word 'singular' here; do not grep for "
                    "it. (Falsified and corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] There is NO wrinkling / tension-field material in "
                    "4C. 'MAT_MembraneWrinkling' does not exist, and the "
                    "membrane catalogue is exactly MAT_Membrane_ElastHyper and "
                    "MAT_Membrane_ActiveStrain — neither of which relaxes a "
                    "compressive state. A membrane that goes into compression "
                    "will therefore return non-physical negative principal "
                    "stresses, and detecting that is on you. Signal: asking "
                    "for the wrinkling material aborts at parse with 'Could "
                    "not match this input' and the MAT_MembraneWrinkling line "
                    "echoed back; substituting the ordinary MAT_ElastHyper "
                    "wrapper instead parses cleanly and is then refused by the "
                    "element with 'The material does not support the "
                    "evaluation of membranes'. (Falsified and corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] THICK is the membrane thickness and it is "
                    "REQUIRED on MEMBRANE3/4/6/9 — there is no default 1.0 to "
                    "be caught out by, so this cannot go wrong silently. "
                    "Signal: omitting it aborts at parse with \"Required value "
                    "'THICK' not found in input line\" from "
                    "core/io/src/4C_io_input_spec_builders.cpp, with no time "
                    "step run. KINEM and STRESS_STRAIN are required on the "
                    "same line for the same reason. Note that the response is "
                    "not a clean linear scaling in THICK once KINEM is "
                    "nonlinear: changing the thickness changes the deformed "
                    "shape, so do not back out a correction factor — re-run. "
                    "(Falsified and corrected by execution 2026-08-06.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "membrane_2d", "description": "Membrane under pressure loading"}]

    def get_template(self, variant: str = "membrane_2d") -> str:
        if variant not in ("membrane_2d", "default"):
            raise ValueError(f"Unknown variant {variant!r}")
        from ..inline_mesh import matched_membrane_2d_input
        return matched_membrane_2d_input()

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
