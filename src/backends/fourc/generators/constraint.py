"""Constraint/coupling condition generator for 4C.

Covers multi-point constraints, rigid body constraints, periodic BCs.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ConstraintGenerator(BaseGenerator):
    """Generator for constraint problems in 4C."""

    module_key = "constraint"
    display_name = "Constraints and Coupling"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Multi-point constraints, rigid body coupling, periodic boundary "
                "conditions, and general coupling of DOFs across discretizations."
            ),
            "condition_types": {
                "MPC": "Multi-point constraints (linear combinations of DOFs)",
                "Rigid body": "Rigid coupling of a set of nodes to a master node",
                "Periodic": "Periodic boundary conditions for unit cell analysis",
                "Mortar": "Mortar-based surface coupling (non-matching meshes)",
                "Penalty": "Penalty-based constraint enforcement",
            },
            "yaml_sections": [
                "DESIGN LINE COUPLING CONDITIONS",
                "DESIGN SURF COUPLING CONDITIONS",
                "DESIGN LINE MPC NORMAL COMPONENT CONDITIONS",
            ],
            "pitfalls": [
                (
                    "[Numerical] Constraint equations must be LINEARLY INDEPENDENT, and 4C "
                    "gives you no help finding out that they are not. Duplicating a multi- "
                    "point constraint makes the saddle-point system singular; the system "
                    "still reaches UMFPACK through "
                    "Constraints::ConstraintSolver::solve_direct and the back-substitution "
                    "divides by the zero pivot. Signal: the process is KILLED by SIGFPE "
                    "inside umfdi_usolve -- the MPI runtime names the floating point "
                    "exception, signal 8, and 4C names nothing -- with no "
                    "'PROC 0 ERROR' line and no mention of rank or singularity anywhere in "
                    "the log. Check rank(C) = n_constraints before the run; there is nothing "
                    "to diagnose after it. The wording 'zero pivot in Schur complement' does "
                    "not exist in 4C or in Trilinos. (Executed 2026-08-06.) "
                ),
                (
                    "[Numerical] For a penalty-enforced coupling constraint the penalty "
                    "parameter trades constraint accuracy against solvability, and the usable "
                    "window is narrow with a hard edge. Signal: too soft, and the run converges with "
                    "the constraint visibly drifting -- the only sign is a RESULT DESCRIPTION "
                    "that no longer matches. Too stiff and Newton simply stops. Signal at the "
                    "stiff end: 'The nonlinear solver did not converge!' from "
                    "4C_solver_nonlin_nox_problem.cpp. This is a Newton failure, not a "
                    "linear-solver one: it happens with a direct solver too, and no "
                    "condition-number warning is ever printed. Do not treat any fixed "
                    "multiple of the stiffness diagonal as a safe default -- bracket it by "
                    "bisection from the value a working deck uses, or switch to "
                    "CONSTRAINT_ENFORCEMENT lagrange_multiplier for exact enforcement. "
                    "(Executed 2026-08-06.) "
                ),
                (
                    "[Numerical] Mortar coupling needs its interface integration configured, "
                    "and the key is NUMGP_PER_DIM in the MORTAR COUPLING section -- there is "
                    "no INTPOINTS_MORTAR in 4C, and writing one is a parse error. "
                    "NUMGP_PER_DIM is only consulted when INTTYPE is Elements or Elements_BS; "
                    "its default of 0 is refused, and so is 1. Signal: 'Invalid Gauss point "
                    "number NUMGP_PER_DIM for element-based integration.' from "
                    "4C_contact_meshtying_strategy_factory.cpp. Switching INTTYPE to Segments "
                    "does not make the key inert -- it is reinterpreted as a triangle rule "
                    "and the run dies later with 'unknown tri gauss rule' from "
                    "4C_mortar_integrator.cpp, so change the two together. (Executed "
                    "2026-08-06.) "
                ),
                (
                    "[Input] Periodic BCs: master and slave surfaces must MATCH "
                    "GEOMETRICALLY. Signal: the failure splits into a silent regime and a loud "
                    "one. A small in-plane offset -- orders of magnitude above ABSTREETOL -- "
                    "is still paired: the pairing banner is unchanged and nothing is warned, "
                    "because ABSTREETOL guides the search tree rather than rejecting a bad "
                    "partner. Only when a slave node leaves the face footprint entirely is a "
                    "partner genuinely lost. Signal for that case: a throw from "
                    "4C_fem_condition_periodic.cpp reading have <n> 'masters in midtosid "
                    "list' <m> expected, both counts substituted at run time. A rotated "
                    "slave has its own limit: ANGLE on an xy pair gives 'Rotation of slave "
                    "plane only implemented for xz and yz planes'. Verify max|x_slave - "
                    "x_master + L*e_per| yourself; a clean pairing banner is not evidence "
                    "that the surfaces match. (Executed 2026-08-06.) "
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "constraint_3d", "description": "Multi-point constraint problem"}]

    def get_template(self, variant: str = "constraint_3d") -> str:
        if variant not in ("constraint_3d", "default"):
            raise ValueError(f"Unknown variant {variant!r}")
        from ..inline_mesh import matched_constraint_3d_input
        return matched_constraint_3d_input()

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
