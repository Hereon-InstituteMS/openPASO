"""ALE (Arbitrary Lagrangian-Eulerian) mesh motion generator for 4C.

Covers stand-alone ALE mesh motion problems where a computational mesh is
deformed according to prescribed boundary displacements.  This is typically
used as a sub-problem within FSI or other moving-boundary simulations, but
can also be run independently for mesh smoothing or mesh motion studies.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ALEGenerator(BaseGenerator):
    """Generator for stand-alone ALE mesh motion problems in 4C."""

    module_key = "ale"
    display_name = "ALE Mesh Motion"
    problem_type = "Ale"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The ALE (Arbitrary Lagrangian-Eulerian) module solves a "
                "pseudo-structural problem to deform a computational mesh "
                "while preserving element quality.  Given prescribed "
                "displacements on some boundaries, ALE computes smooth "
                "interior node displacements.  The PROBLEM TYPE is 'Ale' "
                "and the dynamics section is 'ALE DYNAMIC'.  Elements use "
                "the ALE2 (2-D) or ALE3 (3-D) category.  Materials are "
                "typically hyperelastic (MAT_ElastHyper with "
                "ELAST_CoupLogNeoHooke) to handle large mesh deformations."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "ALE DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_ElastHyper + ELAST_CoupLogNeoHooke": {
                    "description": (
                        "Logarithmic Neo-Hookean hyperelastic material for "
                        "ALE mesh motion.  Handles large rotations and "
                        "deformations of the mesh without element inversion.  "
                        "Material parameters are pseudo-mechanical (not "
                        "physical) and control mesh stiffness distribution."
                    ),
                    "parameters": {
                        "NUMMAT": {
                            "description": "Number of sub-materials (typically 1)",
                            "range": "1",
                        },
                        "MATIDS": {
                            "description": "List of sub-material IDs",
                            "range": "valid MAT IDs",
                        },
                        "DENS": {
                            "description": (
                                "Pseudo-density (not physically meaningful, "
                                "typically 1)"
                            ),
                            "range": "> 0",
                        },
                        "C1 (LogNeoHooke)": {
                            "description": (
                                "First material parameter of log Neo-Hookean "
                                "model (related to shear modulus)"
                            ),
                            "range": "> 0",
                        },
                        "C2 (LogNeoHooke)": {
                            "description": (
                                "Second material parameter (related to "
                                "Poisson's ratio; C2 < 0.5 required)"
                            ),
                            "range": "(0, 0.5)",
                        },
                        "MODE (LogNeoHooke)": {
                            "description": (
                                "Interpretation of C1/C2.  'YN' = Young's "
                                "modulus + Poisson's ratio; 'Lame' = Lame "
                                "parameters lambda + mu."
                            ),
                            "range": "YN | Lame",
                        },
                        "POLYCONVEX": {
                            "description": (
                                "MAT_ElastHyper wrapper flag: enable a runtime "
                                "polyconvexity check on the strain-energy "
                                "function (0 = off, 1 = on)."
                            ),
                            "range": "0 | 1",
                        },
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Simple linear-elastic pseudo-material for ALE.  "
                        "Sufficient for small mesh deformations.  For large "
                        "deformations, prefer the hyperelastic variant."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Pseudo Young's modulus",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Pseudo Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Pseudo density",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "small_problems": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "ALE systems are typically well-conditioned and "
                        "moderate in size.  Direct solvers work well."
                    ),
                },
                "large_problems": {
                    "type": "Belos with MueLu AMG",
                    "notes": (
                        "For very large 3-D meshes, iterative solvers with "
                        "AMG preconditioners are recommended."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": "Time step size for the ALE mesh update.",
                "NUMSTEP": "Total number of ALE time steps.",
                "MAXTIME": "Maximum simulation time.",
                "MAXITER": (
                    "Maximum nonlinear iterations per step.  For linear "
                    "ALE (spring-based), 1 is sufficient.  For nonlinear "
                    "ALE, use 10+."
                ),
            },
            "ale_types": {
                "solid_nln": (
                    "Nonlinear solid-like ALE.  Solves a full nonlinear "
                    "elasticity problem for mesh motion.  Best quality for "
                    "large deformations."
                ),
                "springs_spatial": (
                    "Spring-based ALE with spatial formulation.  Fast, "
                    "suitable for moderate deformations.  Default for 2-D "
                    "FSI."
                ),
                "springs_material": (
                    "Spring-based ALE with material formulation.  Good for "
                    "3-D problems."
                ),
            },
            "pitfalls": [
                (
                    "[API] ALE elements use the ALE2 (2D) or "
                    "ALE3 (3D) element category — NOT SOLID "
                    "or FLUID. In stand-alone ALE problems, "
                    "specify 'ALE ELEMENTS' (not STRUCTURE "
                    "ELEMENTS). Signal: writing 'SOLID QUAD4' "
                    "inside ALE ELEMENTS raises \"Element 'SOLID' does not "
                    "seem to know cell type 'quad4'.\" from "
                    "fem/general/element/4C_fem_general_element_definition.cpp "
                    "-- i.e. it fails on the CELL type, not on the element "
                    "category. Write 'ALE2 QUAD4 <nodes> MAT 1' (2D) or "
                    "'ALE3 HEX8 ... MAT 1' (3D); MAT is the only key either "
                    "takes. (Verified by execution 2026-08-03: the SOLID "
                    "line aborts, the ALE2 line runs to exit 0. An earlier "
                    "version quoted 'expected ALE element type' from a "
                    "'4C_ale_factory.cpp'; that string has zero hits under "
                    "`strings -F` and no such file exists.) "
                    "stand-alone ALE has its own element "
                    "section. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] ALE boundary conditions are "
                    "DIRICHLET only — prescribed mesh "
                    "displacements on boundaries. Interior "
                    "nodes are computed by the ALE solver. "
                    "Signal: applying a Neumann BC to ALE "
                    "is silently ignored (no error message) "
                    "— the mesh motion does not respond. "
                    "Use DESIGN POINT/LINE/SURF/VOL ALE "
                    "DIRICH CONDITIONS to drive the ALE "
                    "field. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] For LARGE rotations, use a "
                    "hyperelastic ALE material "
                    "(ELAST_CoupLogNeoHooke). A linear "
                    "StVenantKirchhoff material does not "
                    "abort — it STALLS AND CARRIES ON. "
                    "Signal: on a mesh rotated through 60° "
                    "4C prints 'ALE newton not converged in "
                    "10 iterations. Continue' for the steps "
                    "where the linear material fails, then "
                    "runs the rest of the schedule to the "
                    "end and reports success. The mesh is "
                    "left wrong and nothing but a RESULT "
                    "DESCRIPTION check notices. There is no "
                    "'negative Jacobian' message in 4C; an "
                    "earlier version of this entry claimed "
                    "one. Always give a rotating ALE mesh a "
                    "result check, or the stall is invisible."
                    " (Falsified and rewritten by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] ALE material parameters are "
                    "PSEUDO-mechanical — they do NOT "
                    "represent physical material properties. "
                    "What matters is how mesh stiffness is "
                    "DISTRIBUTED, not its absolute size: "
                    "stand-alone ALE mesh motion is driven "
                    "purely by boundary displacements with no "
                    "load, so a uniform scaling of YOUNG "
                    "cancels out of the linear system "
                    "entirely. Signal: scaling YOUNG by nine "
                    "orders of magnitude leaves the mesh "
                    "displacement unchanged to the tightest "
                    "tolerance a RESULT DESCRIPTION can ask "
                    "for, while changing NUE moves it "
                    "immediately. Tune NUE, or vary the "
                    "material per element/region, for mesh "
                    "quality — do not expect a bigger YOUNG "
                    "to stiffen anything. (Falsified and "
                    "rewritten by execution 2026-08-06: the "
                    "earlier text said realistic steel values "
                    "make the mesh too stiff to track "
                    "geometry.)"
                ),
                (
                    "[Input] In FSI simulations, the ALE "
                    "mesh is CLONED from the fluid mesh "
                    "via CLONING MATERIAL MAP. For stand-"
                    "alone ALE, define elements directly. "
                    "A stray CLONING MATERIAL MAP is INERT, "
                    "not fatal. Signal: adding one to a "
                    "single-field ALE input — even one naming "
                    "a source field that does not exist and "
                    "material IDs that were never defined — "
                    "produces no diagnostic, no mention of "
                    "cloning anywhere in the output, and a "
                    "bit-identical answer. So the presence of "
                    "the section is NOT evidence that field "
                    "cloning is wired up; check the target "
                    "discretisation was actually created "
                    "instead. There is no 'source field not "
                    "found' message in 4C. (Falsified and "
                    "rewritten by execution 2026-08-06.)"
                ),
                (
                    "[API] Result discretisation name is "
                    "'ale' (lowercase) in RESULT "
                    "DESCRIPTION, and 4C does not tell you "
                    "when you get it wrong. Signal: with "
                    "DIS: 'ALE' the run completes normally, "
                    "writes its output, matches NONE of the "
                    "result tests without comment, and only "
                    "then aborts on a count mismatch — "
                    "'tests but performed' — the two counts are "
                    "fmt-interpolated around that clause, so the line "
                    "reads expected N tests but performed 0 — from "
                    "core/utils/src/result_test/"
                    "4C_utils_result_test.cpp. The mis-spelled "
                    "name is never echoed back, so the log "
                    "looks like a successful run until the "
                    "last line. There is no 'discretisation "
                    "ALE not found' message; an earlier "
                    "version of this entry claimed one. "
                    "(Falsified and rewritten by execution "
                    "2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "rotating_mesh_2d",
                    "description": (
                        "A 2-D mesh with an inner boundary rotating while "
                        "the outer boundary is held fixed.  Tests mesh "
                        "quality preservation under large rotations.  Uses "
                        "nonlinear ALE with ELAST_CoupLogNeoHooke material."
                    ),
                    "template_variant": "ale_2d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "ale_2d",
                "description": (
                    "2-D ALE mesh motion with prescribed boundary "
                    "displacements.  Uses nonlinear hyperelastic material "
                    "(LogNeoHooke) for mesh smoothing, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "ale_2d") -> str:
        templates = {
            "ale_2d": self._template_ale_2d,
        }
        if variant == "default":
            variant = "ale_2d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_ale_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D ALE Mesh Motion
            #
            # A computational mesh is deformed by prescribed boundary
            # displacements.  Interior nodes are smoothly relocated by
            # the ALE solver using a nonlinear hyperelastic formulation.
            #
            # Mesh: exodus file with:
            #   element_block 1 = ALE domain (QUAD4)
            #   node_set 1 = moving boundary (prescribed displacement)
            #   node_set 2 = fixed boundary (zero displacement x)
            #   node_set 3 = fixed boundary (zero displacement y)
            #   point_set 1 = fixed corner point
            # ---------------------------------------------------------------
            TITLE:
              - "2-D ALE mesh motion -- generated template"
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Ale"

            ALE DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              MAXITER: <max_nonlinear_iterations>
              RESULTSEVERY: <results_output_interval>
              LINEAR_SOLVER: 1

            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "ale_solver"

            MATERIALS:
              - MAT: 1
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [<sub_material_id>]
                  DENS: <pseudo_density>
              - MAT: <sub_material_id>
                ELAST_CoupLogNeoHooke:
                  MODE: "YN"
                  C1: <mesh_stiffness_C1>
                  C2: <mesh_stiffness_C2>

            # Prescribed displacement functions
            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<displacement_x_expression>"
            FUNCT2:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<displacement_y_expression>"

            # == Boundary Conditions ===========================================

            # Fixed corner point
            DESIGN POINT DIRICH CONDITIONS:
              - E: <fixed_point_id>
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]

            # Moving boundary (prescribed displacement via functions)
            DESIGN LINE DIRICH CONDITIONS:
              - E: <moving_boundary_id>
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [<displacement_scale_x>, <displacement_scale_y>]
                FUNCT: [1, 2]
              # Fixed boundaries
              - E: <fixed_boundary_x_id>
                NUMDOF: 2
                ONOFF: [1, 0]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]
              - E: <fixed_boundary_y_id>
                NUMDOF: 2
                ONOFF: [0, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]

            # == Geometry (for exodus-based mesh) ==============================
            # Note: Stand-alone ALE uses ALE ELEMENTS section or an exodus file.
            # With exodus, define ALE element blocks:
            # ALE GEOMETRY:
            #   FILE: "<mesh_file>"
            #   ELEMENT_BLOCKS:
            #     - ID: 1
            #       ALE2:
            #         QUAD4:
            #           MAT: 1

            RESULT DESCRIPTION:
              - ALE:
                  DIS: "ale"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement_x>
                  TOLERANCE: <result_tolerance>
              - ALE:
                  DIS: "ale"
                  NODE: <result_node_id>
                  QUANTITY: "dispy"
                  VALUE: <expected_displacement_y>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check C1 (mesh stiffness)
        c1 = params.get("C1")
        if c1 is not None:
            try:
                val = float(c1)
                if val <= 0:
                    issues.append(f"C1 (mesh stiffness) must be > 0, got {val}.")
            except (TypeError, ValueError):
                issues.append(
                    f"C1 must be a positive number, got {c1!r}."
                )

        # Check C2 (related to Poisson's ratio)
        c2 = params.get("C2")
        if c2 is not None:
            try:
                val = float(c2)
                if val <= 0 or val >= 0.5:
                    issues.append(
                        f"C2 must be in (0, 0.5), got {val}.  "
                        f"C2 controls the mesh Poisson's ratio."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"C2 must be a number in (0, 0.5), got {c2!r}."
                )

        # Check MAXITER
        maxiter = params.get("MAXITER")
        if maxiter is not None:
            try:
                mi = int(maxiter)
                if mi < 1:
                    issues.append(
                        f"MAXITER must be >= 1, got {mi}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"MAXITER must be a positive integer, got {maxiter!r}."
                )

        # Check TIMESTEP
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
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        return issues
