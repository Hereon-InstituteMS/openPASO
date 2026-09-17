"""Elastohydrodynamic Lubrication (EHL) generator for 4C.

Covers coupled lubrication + structure problems where the lubricant
pressure deforms the bounding surfaces and the deformation changes the
film geometry (two-way coupling).  The lubrication field solves the
Reynolds equation for pressure, the structural field solves for elastic
deformation, and the two are coupled through the film height (structure
-> lubrication) and pressure loads (lubrication -> structure).
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class EHLGenerator(BaseGenerator):
    """Generator for Elastohydrodynamic Lubrication problems in 4C."""

    module_key = "ehl"
    display_name = "Elastohydrodynamic Lubrication (EHL)"
    problem_type = "Elastohydrodynamic_Lubrication"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Elastohydrodynamic Lubrication (EHL) couples the "
                "Reynolds equation (thin-film lubrication) with "
                "structural mechanics.  The lubricant pressure from "
                "the Reynolds equation is applied as a surface load "
                "on the structural bodies, and the resulting elastic "
                "deformation changes the film thickness, which feeds "
                "back into the Reynolds equation.  This two-way "
                "coupling is essential when the lubricant pressure is "
                "comparable to the elastic modulus of the contact "
                "surfaces (e.g. rolling element bearings, gear tooth "
                "contacts, bio-tribology).  The PROBLEM TYPE is "
                "'Elastohydrodynamic_Lubrication'.  The dynamics "
                "sections are LUBRICATION DYNAMIC, STRUCTURAL DYNAMIC "
                "and ELASTO HYDRO DYNAMIC -- there is no section called "
                "'EHL DYNAMIC'.  The coupling solver settings live one "
                "level down, in the slash-joined top-level sections "
                "ELASTO HYDRO DYNAMIC/MONOLITHIC and ELASTO HYDRO "
                "DYNAMIC/PARTITIONED.  The two fields are tied together "
                "by mortar contact: CONTACT DYNAMIC/STRATEGY must be "
                "'Ehl', a MORTAR COUPLING section must be present, and "
                "the film interface is declared with a Slave/Master pair "
                "in DESIGN SURF EHL MORTAR COUPLING CONDITIONS 3D (or "
                "DESIGN LINE EHL MORTAR COUPLING CONDITIONS 2D).  The "
                "lubrication mesh represents the 2-D film surface and "
                "the structural mesh represents the 3-D elastic bodies; "
                "both live in a 3-D discretisation.  Materials include "
                "MAT_lubrication for the lubricant -- which carries only "
                "DENSITY and LUBRICATIONLAWID, the viscosity coming from "
                "the separate lubrication-law material it points to -- "
                "and a structural material (e.g. "
                "MAT_Struct_StVenantKirchhoff) for the elastic bodies."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "LUBRICATION DYNAMIC",
                "ELASTO HYDRO DYNAMIC",
                "ELASTO HYDRO DYNAMIC/MONOLITHIC",
                "CONTACT DYNAMIC",
                "MORTAR COUPLING",
                "DESIGN SURF EHL MORTAR COUPLING CONDITIONS 3D",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "ELASTO HYDRO DYNAMIC/PARTITIONED",
                "MORTAR COUPLING/PARALLEL REDISTRIBUTION",
                "CLONING MATERIAL MAP",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_lubrication": {
                    "description": (
                        "Lubricant material for the Reynolds equation.  It "
                        "holds ONLY the density and a pointer to a "
                        "lubrication-law material; the viscosity (constant "
                        "or piezoviscous) lives in that law material."
                    ),
                    "parameters": {
                        "LUBRICATIONLAWID": {
                            "description": (
                                "MAT id of the lubrication-law material: "
                                "MAT_lubrication_law_constant for constant "
                                "viscosity, MAT_lubrication_law_barus or "
                                "MAT_lubrication_law_roeland for the "
                                "piezoviscous EHL cases"
                            ),
                            "range": "existing MAT id",
                        },
                        "DENSITY": {
                            "description": "Lubricant density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_lubrication_law_constant": {
                    "description": (
                        "Constant-viscosity lubrication law.  Start here to "
                        "verify an EHL setup before turning on piezoviscosity."
                    ),
                    "parameters": {
                        "VISCOSITY": {
                            "description": "Dynamic viscosity [Pa s]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_lubrication_law_barus": {
                    "description": (
                        "Barus piezoviscous law, mu = mu_0 * exp(alpha*p)."
                    ),
                    "parameters": {
                        "ABSViscosity": {
                            "description": "Reference viscosity mu_0 [Pa s]",
                            "range": "> 0",
                        },
                        "PreVisCoeff": {
                            "description": (
                                "Pressure-viscosity coefficient alpha [1/Pa]"
                            ),
                            "range": ">= 0",
                        },
                    },
                },
                "MAT_lubrication_law_roeland": {
                    "description": "Roelands piezoviscous law.",
                    "parameters": {
                        "ABSViscosity": {
                            "description": "Reference viscosity mu_0 [Pa s]",
                            "range": "> 0",
                        },
                        "PreVisCoeff": {
                            "description": (
                                "Pressure-viscosity coefficient alpha [1/Pa]"
                            ),
                            "range": ">= 0",
                        },
                        "RefPress": {
                            "description": "Roelands reference pressure [Pa]",
                            "range": "> 0",
                        },
                        "RefVisc": {
                            "description": (
                                "Roelands reference viscosity [Pa s]"
                            ),
                            "range": "> 0",
                        },
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Linear elastic material for the structural "
                        "bodies in contact."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus [Pa]",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Structural density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "lubrication_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "The Reynolds equation system is small (2-D) "
                        "and well suited for direct solvers."
                    ),
                },
                "structure_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Structural solver for the 3-D elastic bodies.  "
                        "Direct solver for small problems; iterative "
                        "with AMG for large meshes."
                    ),
                },
            },
            "coupling_parameters": {
                "ELASTO HYDRO DYNAMIC/COUPALGO": (
                    "EHL coupling algorithm.  Exactly two spellings are "
                    "accepted, and they are case sensitive: "
                    "'ehl_Monolithic' (the default) for simultaneous "
                    "solution of lubrication + structure, or "
                    "'ehl_IterStagg' for staggered iteration.  Anything "
                    "else, including the lower-case 'ehl_monolithic', is "
                    "rejected at parse time."
                ),
                "ELASTO HYDRO DYNAMIC/ITEMAX, ITEMIN": (
                    "Maximum / minimum number of coupling iterations over "
                    "the two fields."
                ),
                "ELASTO HYDRO DYNAMIC/UNPROJ_ZERO_DBC": (
                    "Pin film nodes that do not project onto the structure "
                    "to zero pressure with a Dirichlet condition.  Every "
                    "upstream EHL test deck sets this true."
                ),
                "ELASTO HYDRO DYNAMIC/DIFFTIMESTEPSIZE": (
                    "Allow a different step size for the lubrication and "
                    "the solid field."
                ),
                "ELASTO HYDRO DYNAMIC/MONOLITHIC": (
                    "CONVTOL, TOLINC, NORM_RESF, NORM_INC, "
                    "NORMCOMBI_RESFINC, ITERNORM, PTCDT, LINEAR_SOLVER, "
                    "INFNORMSCALING live HERE, not at ELASTO HYDRO "
                    "DYNAMIC.  Putting CONVTOL one level up is rejected."
                ),
                "ELASTO HYDRO DYNAMIC/PARTITIONED": (
                    "CONVTOL, MAXOMEGA, MINOMEGA, STARTOMEGA; only "
                    "consulted when COUPALGO is ehl_IterStagg."
                ),
                "film_height": (
                    "There is no FILM_HEIGHT_FROM key.  In an EHL run the "
                    "film height is the mortar contact gap, delivered "
                    "through the DESIGN ... EHL MORTAR COUPLING CONDITIONS "
                    "Slave/Master pair; LUBRICATION DYNAMIC/PURE_LUB stays "
                    "false and HEIGHTFEILD/HFUNCNO are not used.  Only a "
                    "stand-alone Lubrication problem prescribes the height "
                    "from a FUNCT."
                ),
            },
            "pitfalls": [
                (
                    "[Mesh] The lubrication mesh (2D) and "
                    "the structural mesh (3D) must be "
                    "geometrically compatible at the "
                    "contact surface, and the DESIGN SURF "
                    "EHL MORTAR COUPLING CONDITIONS 3D "
                    "Slave/Master pair must both be there. "
                    "Signal: none, in either direction — "
                    "this is a silent-wrong failure. Giving "
                    "the Master side a different InterfaceID "
                    "from the Slave is accepted and changes "
                    "nothing (the id is not used to pair "
                    "them), and deleting the Master "
                    "condition outright still runs to "
                    "completion with the contact "
                    "displacement collapsing to ~1e-15. "
                    "The quoted 'no matching lubrication "
                    "interface' does not exist in 4C, and "
                    "neither does the file it was attributed "
                    "to, 4C_ehl_factory.cpp; check the "
                    "condition pair by hand. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] The minimum-film-height "
                    "regularisation for h -> 0 is a real key: "
                    "LUBRICATION DYNAMIC/GAP_OFFSET, whose "
                    "default is 0. Signal: none — with it at "
                    "0 the run does NOT produce NaN and does "
                    "NOT report a singular matrix; it "
                    "completes every time step and hands back "
                    "finite displacements and a finite "
                    "pressure that are simply wrong. Set "
                    "GAP_OFFSET to a physical h_min and "
                    "compare the two runs; near-contact will "
                    "not announce itself. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] EHL is HIGHLY nonlinear "
                    "due to the pressure-viscosity "
                    "(piezoviscous) effect: mu = "
                    "mu_0 * exp(alpha * p). Signal: too "
                    "large an alpha does not oscillate "
                    "between two states — the Newton "
                    "residual explodes in a single step and "
                    "the PROCESS IS KILLED by SIGFPE "
                    "(floating-point divide-by-zero inside "
                    "LubricationEleCalc::calc_mat_psl). The "
                    "shell reports 128+8 = 136; 4C prints no "
                    "error line, no 'did not converge' and "
                    "no result-test verdict, because "
                    "MPI_Abort is never reached, so a caller "
                    "that only greps for 4C diagnostics sees "
                    "nothing. Under-relaxation cannot help "
                    "by default: MAXOMEGA / MINOMEGA / "
                    "STARTOMEGA live in ELASTO HYDRO "
                    "DYNAMIC/PARTITIONED and are only "
                    "consulted when ELASTO HYDRO "
                    "DYNAMIC/COUPALGO is ehl_IterStagg, "
                    "while the default is ehl_Monolithic — "
                    "adding them to a monolithic run is "
                    "accepted and inert. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Pressure-dependent "
                    "viscosity (Barus exp or Roelands) "
                    "dramatically affects convergence, so "
                    "start with constant viscosity "
                    "(alpha = 0) to verify the setup and "
                    "then ramp alpha over several "
                    "pseudo-load-steps. Signal: the alpha = 0 "
                    "and moderately-ramped runs complete "
                    "every time step and reach the result-"
                    "test manager; the run that has gone too "
                    "far is killed by SIGFPE part-way "
                    "through the first step, so the usable "
                    "marker is the ABSENCE of a 'Checking "
                    "results of' line rather than any 'Newton "
                    "diverged' message. Bisect alpha on that "
                    "marker. (Corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] The structural load driving the "
                    "contact MUST sit on the CORRECT "
                    "structural face; a coupling surface is "
                    "not a load surface. Signal: none — "
                    "moving the condition onto the mortar "
                    "Slave face is accepted without comment, "
                    "runs every time step, and gives a "
                    "deformation that LOOKS reasonable "
                    "(the driven nodes take exactly the "
                    "prescribed value, so a plot shows a "
                    "deflected body) while the tangential "
                    "displacement collapses to exactly zero "
                    "and the contact-pressure profile is "
                    "wrong. Compare against Hertzian to "
                    "verify face selection. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] For TRANSIENT EHL the "
                    "squeeze-film (dh/dt) term must be "
                    "switched on, and the key is LUBRICATION "
                    "DYNAMIC/ADD_SQUEEZE_TERM (bool, default "
                    "false). There is NO 'TRANSIENT' key. "
                    "Signal: writing TRANSIENT: true aborts "
                    "at parse time in "
                    "core/io/src/4C_io_input_spec_builders.cpp "
                    "with 'Could not match this input', the "
                    "section echoed back and the offending "
                    "key listed under '[!] The following "
                    "data remains unused:' — before any "
                    "field is built. ADD_SQUEEZE_TERM needs "
                    "the height rate, so it requires the EHL "
                    "height field; a function-prescribed "
                    "height cannot supply it. (Corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] Units must be consistent: "
                    "viscosity in Pa s, pressure in Pa, "
                    "lengths in m, Young's modulus in Pa. "
                    "Signal: on a monolithic EHL a viscosity "
                    "unit slip is not a silent scaling "
                    "error — it destroys the linear algebra "
                    "and the process is KILLED by SIGFPE "
                    "(shell status 136) with no 4C error "
                    "line, no MPI_ABORT block and no result-"
                    "test verdict. Where it dies depends on "
                    "the very scaling that is supposed to "
                    "help: with 4C's default INFNORMSCALING "
                    "it is Epetra_CrsMatrix::InvRowSums, "
                    "called from EHL::Monolithic::"
                    "scale_system, that overflows; with "
                    "INFNORMSCALING off the same deck dies "
                    "inside UMFPACK's umfdi_kernel_init. "
                    "Non-dimensionalising pressure by "
                    "p_Hertz and length by the contact "
                    "half-width is the real remedy. "
                    "(Corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] Do NOT put IO/RUNTIME VTK "
                    "OUTPUT/STRUCTURE: OUTPUT_STRUCTURE: true "
                    "in an EHL deck. Signal: it is not a "
                    "silent no-op — the run aborts during "
                    "setup with 'Runtime output is not "
                    "available in the old structure time "
                    "integration! You need to take the new "
                    "one, i.e. set `INT_STRATEGY: Standard`!' "
                    "from structure/4C_structure_timint.cpp, "
                    "raised out of EHL::Base::Base. The advice "
                    "in that message does NOT work here: EHL "
                    "constructs Solid::TimIntStatics directly, "
                    "so adding INT_STRATEGY: Standard to "
                    "STRUCTURAL DYNAMIC changes nothing and "
                    "the identical abort repeats. Drop the "
                    "IO/RUNTIME VTK OUTPUT/STRUCTURE section; "
                    "the plain IO/RUNTIME VTK OUTPUT section "
                    "is fine. (Verified by execution "
                    "2026-08-07.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "line_contact_ehl",
                    "description": (
                        "EHL line contact: a cylinder rolling on a "
                        "flat surface with a lubricant film.  The "
                        "classical Hertzian-EHL benchmark.  Tests "
                        "pressure-film-height coupling and elastic "
                        "flattening of the contact zone."
                    ),
                    "template_variant": "ehl_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "ehl_3d",
                "description": (
                    "3-D EHL: lubrication + elastic structure coupling.  "
                    "Reynolds equation on 2-D film mesh, linear elastic "
                    "3-D structure.  UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "ehl_3d") -> str:
        templates = {
            "ehl_3d": self._template_ehl_3d,
        }
        if variant == "default":
            variant = "ehl_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_ehl_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Elastohydrodynamic Lubrication (EHL)
            #
            # Coupled lubrication (Reynolds equation) + structural mechanics.
            # The lubricant pressure deforms the elastic bodies, which
            # changes the film geometry and feeds back into the Reynolds
            # equation.
            #
            # Mesh: requires:
            #   Lubrication mesh: "lub.e" with
            #     element_block 1 = lubrication film, a QUAD4 SURFACE in
            #                       3-D space (the discretisation is 3-D)
            #     node_set 1 = inlet boundary (pressure Dirichlet)
            #     node_set 2 = outlet boundary (pressure Dirichlet)
            #   Structure mesh: "structure.e" with
            #     element_block 1 = elastic body (HEX8, 3-D)
            #     node_set 1 = bottom face (fixed)
            #     node_set 2 = contact surface (receives lubricant pressure)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D elastohydrodynamic lubrication -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Elastohydrodynamic_Lubrication"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            # Do NOT add IO/RUNTIME VTK OUTPUT/STRUCTURE here: EHL runs on
            # the old structure time integrator and OUTPUT_STRUCTURE: true
            # aborts the run during setup (INT_STRATEGY does not help).

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <structure_timestep>
              NUMSTEP: <structure_num_steps>
              MAXTIME: <structure_max_time>
              LINEAR_SOLVER: 1
              TOLRES: <structure_residual_tolerance>
              TOLDISP: <structure_displacement_tolerance>
              DIVERCONT: "continue"

            # == Contact / mortar coupling of the two fields ===================
            # EHL is driven through mortar contact.  Without STRATEGY "Ehl"
            # and a MORTAR COUPLING section the two fields are not tied.
            CONTACT DYNAMIC:
              LINEAR_SOLVER: 1
              STRATEGY: "Ehl"
            MORTAR COUPLING:
              SEARCH_PARAM: <mortar_search_parameter>
              INTTYPE: "Elements"
              NUMGP_PER_DIM: <mortar_gauss_points_per_dim>
              TRIANGULATION: "Center"
            MORTAR COUPLING/PARALLEL REDISTRIBUTION:
              PARALLEL_REDIST: "None"

            # == Lubrication ===================================================
            # No SOLVERTYPE key exists here, and the sliding velocity is NOT
            # a key either: in EHL the height and the velocity both come
            # from the mortar coupling (PURE_LUB stays at its default false).
            LUBRICATION DYNAMIC:
              TIMESTEP: <lubrication_timestep>
              NUMSTEP: <lubrication_num_steps>
              MAXTIME: <lubrication_max_time>
              LINEAR_SOLVER: 2
              RESULTSEVERY: <results_output_interval>
              CONVTOL: <lubrication_newton_tolerance>
              PENALTY_CAVITATION: <cavitation_penalty>
              GAP_OFFSET: <minimum_film_height>
              ADD_SQUEEZE_TERM: true

            # == EHL coupling ==================================================
            # The section is "ELASTO HYDRO DYNAMIC"; "EHL DYNAMIC" does not
            # exist.  COUPALGO takes exactly ehl_Monolithic or ehl_IterStagg.
            ELASTO HYDRO DYNAMIC:
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              COUPALGO: "ehl_Monolithic"
              ITEMAX: <ehl_max_coupling_iterations>
              ITEMIN: <ehl_min_coupling_iterations>
              RESULTSEVERY: <results_output_interval>
              UNPROJ_ZERO_DBC: true
            # Convergence control of the monolithic EHL solve lives in the
            # sub-section, not in ELASTO HYDRO DYNAMIC itself.
            ELASTO HYDRO DYNAMIC/MONOLITHIC:
              CONVTOL: <ehl_residual_tolerance>
              TOLINC: <ehl_increment_tolerance>
              NORMCOMBI_RESFINC: "And"
              LINEAR_SOLVER: 1

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "lubrication_solver"

            # == Materials =====================================================
            MATERIALS:
              # Structural material
              - MAT: 1
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <Young_modulus>
                  NUE: <Poisson_ratio>
                  DENS: <density>
              # Lubricant material.  MAT_lubrication carries only DENSITY
              # and LUBRICATIONLAWID; the viscosity is in the law material.
              # Swap MAT 3 for MAT_lubrication_law_barus (ABSViscosity,
              # PreVisCoeff) or ..._roeland for a piezoviscous EHL run.
              - MAT: 2
                MAT_lubrication:
                  LUBRICATIONLAWID: 3
                  DENSITY: <lubricant_density>
              - MAT: 3
                MAT_lubrication_law_constant:
                  VISCOSITY: <lubricant_dynamic_viscosity>

            # == Boundary Conditions ===========================================

            # Structure: fixed bottom
            DESIGN SURF DIRICH CONDITIONS:
              - E: <structure_fixed_face_id>
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Lubrication: pressure BCs.  There is no lubrication-specific
            # Dirichlet section -- the film uses the generic DESIGN ...
            # DIRICH CONDITIONS with NUMDOF 1 (the single pressure dof).
            DESIGN LINE DIRICH CONDITIONS:
              - E: <lub_inlet_boundary_id>
                ENTITY_TYPE: node_set_id
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<inlet_pressure>]
                FUNCT: [0]
              - E: <lub_outlet_boundary_id>
                ENTITY_TYPE: node_set_id
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<outlet_pressure>]
                FUNCT: [0]

            # Film interface: the Slave/Master pair is what actually ties
            # the lubrication field to the structure.  BOTH entries must be
            # present; a missing Master runs to completion and silently
            # produces ~0 coupling.
            DESIGN SURF EHL MORTAR COUPLING CONDITIONS 3D:
              - E: <lubrication_contact_surface_id>
                ENTITY_TYPE: node_set_id
                InterfaceID: 1
                Side: "Slave"
              - E: <structure_contact_surface_id>
                ENTITY_TYPE: node_set_id
                InterfaceID: 1
                Side: "Master"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<structure_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 1
                      KINEM: <kinematics>

            LUBRICATION GEOMETRY:
              FILE: "<lubrication_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  LUBRICATION:
                    QUAD4:
                      MAT: 2

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_structure_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
              - LUBRICATION:
                  DIS: "lubrication"
                  NODE: <result_lubrication_node_id>
                  QUANTITY: "pre"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check lubricant viscosity.  It belongs to the lubrication-law
        # material (VISCOSITY / ABSViscosity), NOT to MAT_lubrication.
        if params.get("DYNVISCOSITY") is not None:
            issues.append(
                "DYNVISCOSITY is not a lubrication parameter. Put the "
                "viscosity on MAT_lubrication_law_constant as VISCOSITY "
                "(or on MAT_lubrication_law_barus/_roeland as "
                "ABSViscosity) and point MAT_lubrication/LUBRICATIONLAWID "
                "at it."
            )
        for key in ("VISCOSITY", "ABSViscosity"):
            viscosity = params.get(key)
            if viscosity is not None:
                try:
                    mu = float(viscosity)
                    if mu <= 0:
                        issues.append(f"{key} must be > 0, got {mu}.")
                except (TypeError, ValueError):
                    issues.append(
                        f"{key} must be a positive number, "
                        f"got {viscosity!r}."
                    )

        # Check lubricant density
        density = params.get("DENSITY") or params.get("lubricant_density")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(
                        f"Lubricant DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu < 0 or nu >= 0.5:
                    issues.append(
                        f"NUE must be in [0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in [0, 0.5), got {nue!r}."
                )

        # Keys that do not exist in LUBRICATION DYNAMIC
        if params.get("SURFACE_VELOCITY") is not None:
            issues.append(
                "SURFACE_VELOCITY is not a LUBRICATION DYNAMIC key. In an "
                "EHL run the sliding velocity comes from the mortar "
                "coupling, not from an input key."
            )
        if params.get("SOLVERTYPE") is not None:
            issues.append(
                "SOLVERTYPE is not a LUBRICATION DYNAMIC key. The Reynolds "
                "problem always uses the implicit Newton loop; tune it with "
                "CONVTOL / ITEMAX."
            )
        if params.get("FILM_HEIGHT_FROM") is not None:
            issues.append(
                "FILM_HEIGHT_FROM does not exist. The EHL film height is "
                "the mortar contact gap, declared with the Slave/Master "
                "pair in DESIGN SURF EHL MORTAR COUPLING CONDITIONS 3D."
            )

        # Check coupling algorithm (case sensitive, exactly two values)
        coupalgo = params.get("COUPALGO")
        if coupalgo is not None and coupalgo not in (
            "ehl_Monolithic", "ehl_IterStagg",
        ):
            issues.append(
                f"EHL COUPALGO must be 'ehl_Monolithic' or "
                f"'ehl_IterStagg' (case sensitive), got {coupalgo!r}."
            )

        return issues
