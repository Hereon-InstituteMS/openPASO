"""Reduced-dimensional lung model generator for 4C.

Couples reduced airways with alveolar tissue mechanics.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ReducedLungGenerator(BaseGenerator):
    """Generator for reduced lung model in 4C."""

    module_key = "reduced_lung"
    display_name = "Reduced Lung Model"
    problem_type = "ReducedLung"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Reduced-dimensional lung model coupling 1-D airway trees with "
                "0-D alveolar tissue compartments.  Used for whole-lung ventilation "
                "simulation."
            ),
            "coupling": "Reduced airways (1D) + alveolar acini (0D) + optional 3D parenchyma",
            "applications": ["ventilation simulation", "mechanical ventilation",
                             "lung disease modeling", "airway pressure distribution"],
            "pitfalls": [
                (
                    "[Input] Airway tree TOPOLOGY must be physiologically "
                    "reasonable - typically 16-23 generations (Weibel symmetric "
                    "model) - and 4C will not check it for you. Signal: the "
                    "reduced_lung setup banner REPORTS the tree it built "
                    "(Airways, Terminal Units, Connections, Bifurcations, "
                    "Boundary Conditions) but validates nothing: a chain with "
                    "zero bifurcations runs to completion with no warning and "
                    "simply returns a different total resistance and inlet "
                    "flow. Read that banner and check it against the lung you "
                    "meant. (Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Alveolar compliance varies with DISEASE STATE. In "
                    "4C the terminal-unit stiffness is reduced_dimensional_lung "
                    ".lung_tree.terminal_units.elasticity_model.linear.elastici "
                    "ty_e, and like every field in that block it takes "
                    "`constant:` or `from_file:` - so per-acinus heterogeneity "
                    "is a JSON field, not a code change. Signal: none - a "
                    "uniform value is accepted silently, yet two acini with the "
                    "same MEAN stiffness but different individual values split "
                    "the flow completely differently and change the total. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Numerical] Coupling between 1D airways and 0D acini via "
                    "flow/pressure matching carries a unit error straight "
                    "through: the coupled system is linear in the terminal-unit "
                    "elastance, so a factor-60 slip (s vs min) becomes a "
                    "factor-60 error in every flow in the tree, uniformly and "
                    "with no distortion. Signal: none - nothing at the "
                    "interface checks units, the run exits 0 and the answer "
                    "looks perfectly well behaved. Fix the units before you "
                    "trust the tidal volume. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "lung_1d", "description": "Reduced lung ventilation model"}]

    def get_template(self, variant: str = "lung_1d") -> str:
        return (
            "# =====================================================\n"
            "# 4C reduced-lung-tree (variant: lung_1d)\n"
            "# =====================================================\n"
            "# Not a self-contained runnable input. 4C reduced-lung\n"
            "# problems couple a 1-D airway tree (typically derived\n"
            "# from a patient CT) with 0-D alveolar acini and an\n"
            "# optional 3-D parenchyma. The required mesh +\n"
            "# topology depends on:\n"
            "#\n"
            "#   * airway-tree topology (NODE COORDS + ARTERY\n"
            "#     ELEMENTS sections)\n"
            "#   * alveolar compliance distribution per acinus\n"
            "#   * coupling type (1D-0D pressure-flow or 3D-0D\n"
            "#     surface-to-acinus mortar)\n"
            "#\n"
            "# Pitfalls (see knowledge() for the full set):\n"
            "#   * Airway-tree topology must be physiologically\n"
            "#     reasonable (Horsfield / Strahler order)\n"
            "#   * Alveolar compliance varies with disease state\n"
            "#     (emphysema, fibrosis, ARDS)\n"
            "#   * Coupling between 1-D airways and 0-D acini\n"
            "#     happens via flow/pressure matching at outlets\n"
            "# =====================================================\n"
            "TITLE:\n"
            "  - \"4C reduced-lung-tree reference stub\"\n"
            "PROBLEM TYPE:\n"
            "  PROBLEMTYPE: \"ReducedLung\"\n"
            "REDUCED DIMENSIONAL AIRWAYS DYNAMIC:\n"
            "  DYNAMICTYPE: \"OneStepTheta\"\n"
            "  TIMESTEP: 0.01\n"
            "  NUMSTEP: 100\n"
            "# Concrete airway-tree NODE COORDS / ARTERY ELEMENTS\n"
            "# / boundary BCs must be supplied per patient case.\n"
        )

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
