"""Runnable 4C deck templates — every one executed on the installed binary.

WHY THIS MODULE EXISTS
----------------------
`prepare_simulation(fourc, <physics>)` used to answer 27 of 4C's 49 catalog
physics with a comment block that said, in as many words, "Not a runnable
input — the user must supply the case-specific mesh + material parameters."
This module is what it answers with instead: 29 decks over 24 physics, every
one of them executed.
Every other backend in this project ships a fillable skeleton for every physics
it advertises (deal.II 27/27, scikit-fem 22/22, FEBio 17/17, SPARTA 10/10). 4C,
the backend the project is named around, was the exception — and it is the one
whose input format a model is least able to reconstruct from prose: 478
sections, 7383 paths, 2728 distinct keys.

WHAT IS IN HERE, AND WHAT "RUNNABLE" MEANS
------------------------------------------
Each entry is a complete deck, not a skeleton with `<...>` holes. Every one was
executed with `{FOURC_BINARY}` at the rank count recorded in
`np` and exited 0. A deck that 4C rejects is worse than no deck at all,
because it looks like help; so the rule for adding an entry here is that the
exact bytes shipped are the exact bytes that ran.

Each deck is also checked for *physics*, not only for exit status — a run that
completes without the coupling doing anything is a silent failure. The evidence
per deck lives in the `evidence` field and was measured from that deck's own
output (VTU fields, reaction forces, interface fluxes), never assumed.

HOW THE DECKS WERE BUILT
------------------------
From the upstream corpus, not from the grammar. `{FOURC_ROOT}/tests/
input_files` holds 1978 parseable decks that 4C's own CI runs; each template
starts from a named one of those (`upstream`) and is reduced — mesh shrunk,
regression `RESULT DESCRIPTION` deleted, Belos+XML solvers replaced by direct
UMFPACK — until it is self-contained and small enough to render whole. Two
(`particle_pd_*`) had no upstream deck to start from and were built from the
grammar dump plus the deck shape used by the author of 4C's PD module.

THE TWO DEPENDENCIES THAT COULD NOT BE REMOVED
-----------------------------------------------
`fs3i` and `multiscale` name a file outside the deck. Both were established by
execution, not assumed:

  * FS3I refuses any direct solver — `4C_fs3i_partitioned.cpp:604` throws
    "Iterative solver expected" unless COUPLED_LINEAR_SOLVER is Belos, `:610`
    demands AZPREC Teko, and `TekoPreconditioner::setup` then demands
    TEKO_XML_FILE. The deck therefore names 4C's own recommended block
    preconditioner by absolute path.
  * FE2 multiscale reads the RVE from a second, standalone `InputFile` through
    `Global::read_micro_fields`, one per macro multiscale material, so
    `MICROFILE` cannot be inlined. The deck names 4C's own RVE by path.

Both resolve through `FOURC_ROOT`, the same environment variable this backend
already uses to resolve Exodus meshes. When it is unset the deck still renders
and still teaches the layout, but it will not run — `requires_fourc_root` marks
those two so callers can say so instead of letting a user discover it at
MPI_Abort time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DECK_DIR = Path(__file__).resolve().parent / "decks"

# Placeholder that the loader rewrites to an absolute path under FOURC_ROOT.
FOURC_ROOT_TOKEN = "@FOURC_ROOT@"


@dataclass(frozen=True)
class Deck:
    """One executed template."""

    physics: str
    variant: str
    filename: str
    np: int                      # rank count the deck was verified at
    upstream: str                # the tests/input_files deck it derives from
    summary: str                 # one line, what it simulates
    evidence: str                # what was measured to show the physics is live
    requires_fourc_root: bool = False
    pitfalls: tuple[str, ...] = field(default_factory=tuple)

    def path(self) -> Path:
        return _DECK_DIR / self.filename

    def text(self) -> str:
        raw = self.path().read_text()
        return _resolve_root(raw)


def _resolve_root(text: str) -> str:
    """Rewrite @FOURC_ROOT@ to the configured 4C source tree.

    Left verbatim when FOURC_ROOT is unset: a wrong absolute path is a worse
    failure than a visible placeholder, because 4C reports the missing file
    and not the missing configuration.

    EVERY SUBSTITUTED LINE SAYS WHERE ITS PATH CAME FROM. The resolved deck is
    handed to an agent by `generate_input`, so an unannotated
    `/home/<someone>/4C/tests/...` in it reads as a fact about 4C rather than
    as a fact about this machine, and is wrong on every other one. The deck
    header already says to set FOURC_ROOT, but the substituted lines sit
    thousands of characters below it — far outside the window
    test_no_served_payload_hard_codes_a_host_path looks in, and far outside
    what a reader skimming the middle of a deck would connect it to.
    A trailing YAML comment is ignored by 4C, so the deck still runs verbatim.
    """
    if FOURC_ROOT_TOKEN not in text:
        return text
    root = os.environ.get("FOURC_ROOT")
    if not root:
        return text
    resolved = str(Path(root).resolve())
    out = []
    for line in text.split("\n"):
        if FOURC_ROOT_TOKEN not in line:
            out.append(line)
            continue
        line = line.replace(FOURC_ROOT_TOKEN, resolved)
        # Only where a comment cannot change the value: a `#` inside a quoted
        # scalar would be part of the string, and a line that already carries
        # one needs nothing.
        if "#" not in line:
            line = f"{line}    # path from $FOURC_ROOT on this machine"
        out.append(line)
    return "\n".join(out)


# ── the catalog ────────────────────────────────────────────────────────────
# Ordered by physics name. `np` is the rank count the deck was RUN at, not a
# guess: two decks genuinely need more than one rank (see their pitfalls).

DECKS: tuple[Deck, ...] = (
    Deck(
        physics="arterial_network", variant="single_artery_1d",
        filename="arterial_network.4C.yaml", np=1,
        upstream="one_d_3_artery_network.4C.yaml",
        summary="One-dimensional blood flow through a small arterial tree: a "
                "parent vessel bifurcating into two compliant daughter "
                "vessels, with a periodic cardiac inflow waveform.",
        evidence="Explicit Taylor-Galerkin integration of the area/flow "
                 "equations completes with the junction and both reflective "
                 "outlets active; area and flow-rate results written.",
        pitfalls=(
            "The reference area comes from the element line's DIAM, not from "
            "an AREA0 material parameter — there is no AREA0 key.",
            "The junction is closed by DESIGN NODE 1D ARTERY JUNCTION "
            "CONDITIONS entries sharing one ConditionID; a missing partner is "
            "not diagnosed, the tree just leaks.",
        ),
    ),
    Deck(
        physics="beam_interaction", variant="beam_contact_3d",
        filename="beam_interaction_beam_contact.4C.yaml", np=2,
        upstream="beam3eb_static_contact_penalty_linposquadpen_"
                 "beamslidingoverarc.4C.yaml",
        summary="Beam-to-beam penalty contact: a straight Kirchhoff beam is "
                "pressed onto a bent one and slid across it over 18 "
                "quasi-static steps.",
        evidence="4C reports 'currently monitors 3 beam contact pairs' in "
                 "every step, so pairs are found and the penalty law is "
                 "evaluated.",
        pitfalls=(
            "SEARCH_STRATEGY: bounding_volume_hierarchy needs ArborX. On a "
            "build without it every beam-interaction deck aborts in "
            "4C_geometric_search_bounding_volume.hpp:79. The default "
            "bruteforce_with_binning has no such dependency.",
            "BINNING STRATEGY DOMAINBOUNDINGBOX must enclose the DEFORMED "
            "geometry — pairs outside it are silently never searched.",
        ),
    ),
    Deck(
        physics="beam_interaction", variant="beam_solid_meshtying_3d",
        filename="beam_interaction_beam_to_solid.4C.yaml", np=1,
        upstream="beam3r_herm2line3_static_beam_to_solid_volume_meshtying_"
                 "beam_along_solid_boundary_segmentation.4C.yaml",
        summary="Beam-to-solid volume meshtying: a Simo-Reissner beam embedded "
                "in a solid column, tied to it by a Gauss-point-to-segment "
                "penalty constraint, is pulled at its overhanging tip.",
        evidence="32 meshtying pairs monitored every step, and a solid corner "
                 "node on the far side of the column is dragged 0.117 — load "
                 "really crosses the tie.",
        pitfalls=(
            "Both condition sections (…VOLUME MESHTYING VOLUME and …LINE) and "
            "both topology sections are mandatory. Drop either and beam and "
            "solid pass through each other with no diagnostic.",
            "See the beam_contact_3d note on ArborX.",
        ),
    ),
    Deck(
        physics="brownian_dynamics", variant="brownian_3d",
        filename="brownian_dynamics.4C.yaml", np=1,
        upstream="beam3r_herm2line3_backweuler_browndyn_crosslinking_"
                 "beam3rline2_additional_fixed_crosslink.4C.yaml",
        summary="Brownian dynamics of semiflexible filaments: crosslinked "
                "beams in a periodic box under thermal forcing, integrated "
                "with overdamped backward Euler.",
        evidence="An otherwise identical KT: 0.0 control differs at t = 0.01 "
                 "by max|du| = 6.6e-2 and max|d(curvature)| = 0.36 at the "
                 "Gauss points, 22% of the deterministic curvature — the "
                 "stochastic forcing drives the result.",
        pitfalls=(
            "KT defaults to 0. BROWNDYNPROB: true with KT unset is a "
            "deterministic run wearing a Brownian label — the upstream deck "
            "this derives from had exactly that defect.",
            "Turning KT on with the default Cylinder_geometry_approx drag "
            "diverges in step 2 (residual 2.67e2 -> 8.74e6, abort in "
            "4C_solver_nonlin_nox_problem.cpp:165). Specifying "
            "BEAMS_DAMPING_COEFF_PER_UNITLENGTH explicitly is what makes the "
            "stochastic step size survivable.",
            "np 1 only: at 2 ranks this aborts in 4C_binstrategy.cpp:1008 "
            "('Node … resides outside the binning domain'). The unmodified "
            "upstream deck fails identically, so it is inherited.",
        ),
    ),
    Deck(
        physics="cardiac_monodomain", variant="monodomain_3d",
        filename="cardiac_monodomain.4C.yaml", np=1,
        upstream="scatra_myocard_MV_material.4C.yaml",
        summary="Cardiac monodomain: the reaction-diffusion equation for the "
                "transmembrane potential coupled to the Bueno-Orovio minimal "
                "ventricular ionic model, stimulated twice.",
        evidence="phi at node 1 reads 0.7136 at t = 400 ms, i.e. mid-"
                 "repolarisation of the second action potential — the AP "
                 "actually fires rather than the field sitting at rest.",
        pitfalls=(
            "The element line needs TYPE CardMono and a FIBER1 direction; "
            "DIFF1/DIFF2/DIFF3 are along fibre and the two cross-fibre "
            "directions, so an isotropic-looking material is still "
            "orientation-dependent.",
        ),
    ),
    Deck(
        physics="contact", variant="penalty_3d",
        filename="contact_penalty_3d.4C.yaml", np=1,
        upstream="contact3D_lin_penalty.4C.yaml + "
                 "contact3D_symmetry_penalty_new_struct.4C.yaml",
        summary="Mortar penalty contact between two separate bodies: a stiff "
                "punch descends across an initial gap onto a clamped soft "
                "foundation and indents it.",
        evidence="4C's per-step 'Total ACTIVE nodes' reads 0 for the first "
                 "nine steps and 5 then 9 from step 10 — exactly where the "
                 "prescribed descent closes the 0.05 gap. The foundation node "
                 "under the punch moves -4.19e-02 while a corner outside the "
                 "patch reaches only -1.46e-02, so this is local indentation "
                 "and not rigid translation.",
        pitfalls=(
            "LM_SHAPEFCN: Dual needs LM_DUAL_CONSISTENT: none, or "
            "contact_strategy_factory.cpp:263 throws 'Consistent dual shape "
            "functions in boundary elements only for Lagrange multiplier "
            "strategy.'",
            "This is the two-body mortar route. The other contact variant, "
            "inline_penalty_3d, is a single inline block and teaches the "
            "self-contact/simple case instead.",
        ),
    ),
    Deck(
        physics="ehl", variant="ehl_3d",
        filename="ehl.4C.yaml", np=2,
        upstream="ehl3d_mixed.4C.yaml",
        summary="Elastohydrodynamic lubrication: a soft neo-Hookean pad "
                "pressed onto a sliding rigid plate with the oil film "
                "resolved by a Reynolds equation on the pad underside, solved "
                "monolithically with mortar contact.",
        evidence="Newton converges quadratically with 6 to 8 active contact "
                 "nodes in all 20 steps — the mixed lubricated/dry patch is "
                 "genuinely resolved.",
        pitfalls=(
            "CONTACT DYNAMIC STRATEGY must be 'Ehl'; the ordinary contact "
            "strategies do not carry the film coupling.",
            "A viscosity unit slip is not a silent scaling error here — the "
            "monolithic Newton diverges.",
        ),
    ),
    Deck(
        physics="fbi", variant="penalty_3d",
        filename="fbi.4C.yaml", np=2,
        upstream="fbi_mortar_solidcoupling.4C.yaml",
        summary="Fluid-beam interaction: a slender beam immersed in a 3-D "
                "flow, tied to the fluid by a penalty-regularised mortar "
                "constraint, on a fluid mesh that does not conform to it.",
        evidence="Beam displacement grows monotonically 1.8e-5 -> 2.8e-3 over "
                 "five steps as the free stream ramps up — the fluid-to-beam "
                 "force transfer is live.",
        pitfalls=(
            "SEARCH_RADIUS must cover a fluid element diagonal; too small and "
            "the beam couples to nothing, with no error — a beam lying "
            "entirely outside the fluid mesh also raises none.",
            "A Dirichlet FUNCT must be SYMBOLIC_FUNCTION_OF_SPACE_TIME; the "
            "time-only form aborts in 4C_utils_function_manager.hpp:143.",
        ),
    ),
    Deck(
        physics="fpsi", variant="monolithic_3d",
        filename="fpsi.4C.yaml", np=1,
        upstream="fpsi_ofsiinterface.4C.yaml",
        summary="Monolithic fluid-porous-structure interaction: free ALE flow "
                "next to a Darcy-saturated neo-Hookean poroelastic block and "
                "an elastic solid, with all three interface families active.",
        evidence="All four fields (fluid, poro structure, porofluid, ALE) "
                 "assemble into one Newton system that converges over the "
                 "driven ramp.",
        pitfalls=(
            "Three distinct condition families are needed and are easy to "
            "confuse: DESIGN FPSI COUPLING SURF (free fluid to porous), "
            "DESIGN FSI COUPLING SURF (free fluid to solid), and DESIGN VOLUME "
            "POROCOUPLING CONDITION (skeleton to pore fluid).",
            "The upstream deck declares a DSURFACE built from nodes that do "
            "not exist; 4C accepts it because no condition references it.",
        ),
    ),
    Deck(
        physics="fsi_xfem", variant="xfem_fsi_3d",
        filename="fsi_xfem.4C.yaml", np=2,
        upstream="xfsi_3D_boxes.4C.yaml",
        summary="Monolithic fixed-grid FSI: the structure surface cuts a fixed "
                "Eulerian fluid mesh and Nitsche coupling enforces the "
                "interface conditions, with both meshes generated inline.",
        evidence="Traction-driven flow past the immersed rotated box completes "
                 "with the XFEM monolithic coupling active on all six "
                 "structure faces.",
        pitfalls=(
            "COUPALGO must be iter_xfem_monolithic — the ALE-based FSI "
            "algorithms do not apply to a cut mesh.",
            "STRUCTURE DOMAIN and FLUID DOMAIN generate both meshes inline; "
            "the structure must lie inside the fluid box or it cuts nothing.",
        ),
    ),
    Deck(
        physics="multiscale", variant="fe2_3d",
        filename="multiscale.4C.yaml", np=2,
        upstream="sohex8_multiscale_macro.4C.yaml",
        summary="FE2 computational homogenisation: a macro block with no "
                "constitutive law of its own, whose stress at every Gauss "
                "point comes from solving a boundary value problem on a micro "
                "RVE driven by the local deformation gradient.",
        evidence="64 independent micro problems are opened and written "
                 "(out_microdis{1,2,3}_el*_gp*), one per macro Gauss point per "
                 "micro discretisation — the homogenisation really runs rather "
                 "than the macro material falling back to something local.",
        requires_fourc_root=True,
        pitfalls=(
            "FE2 is intrinsically a TWO-file problem type. "
            "Global::read_micro_fields opens a second, standalone InputFile "
            "per macro multiscale material, so the RVE cannot be inlined; this "
            "deck names 4C's own RVE by absolute path. Omitting MICROFILE does "
            "not help — its default is the literal placeholder 'filename.dat', "
            "so you get the same 'Input file does not exist' abort with a "
            "different name.",
            "Pointing MICROFILE at the deck itself segfaults: the recursive "
            "read corrupts the problem registry and the backtrace surfaces in "
            "an unrelated field's setup.",
            "MICRODIS_NUM numbers the independent micro discretisations; two "
            "macro materials sharing a number share one RVE.",
        ),
    ),
    Deck(
        physics="particle_dem", variant="settling_3d",
        filename="particle_dem.4C.yaml", np=2,
        upstream="particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml",
        summary="Discrete element method: 48 rigid spheres of two sizes fall "
                "under gravity, collide with each other and the six walls of "
                "the bounding box, and settle into a static pack.",
        evidence="Potential energy falls 0.1911 -> 0.0599, kinetic energy "
                 "decays to 5.6e-6 and contact energy stays small but "
                 "non-zero — the pack lands and comes to rest.",
        pitfalls=(
            "PARTICLE_WALL_SOURCE: BoundingBox turns the six faces of "
            "DOMAINBOUNDINGBOX into rigid walls, so a container needs no mesh "
            "at all — but it also needs PARTICLE_WALL_MAT.",
            "The normal stiffness is derived from MAX_VELOCITY and "
            "REL_PENETRATION, not given directly; raising the stiffness "
            "without lowering TIMESTEP silently violates the stability limit.",
        ),
    ),
    Deck(
        physics="particle_sph", variant="hydrostatic_2d",
        filename="particle_sph_hydrostatic.4C.yaml", np=2,
        upstream="particle_sph_1d_hydrostatic_freesurface_densityintegration_"
                 "cubicspline_adami.4C.yaml",
        summary="Weakly compressible SPH: a fluid column on three boundary "
                "layers settles under gravity to the hydrostatic density and "
                "pressure profile.",
        evidence="At the final time max|v| = 1.7e-12 and the pressure profile "
                 "matches rho0*g*(H-x) to under 1% in the bulk — it really "
                 "converges to the hydrostatic solution.",
        pitfalls=(
            "There is no SOUNDSPEED key. The artificial speed of sound is "
            "sqrt(BULK_MODULUS/rho0) from MAT_ParticleSPHFluid.",
            "INITRADIUS is the kernel support radius and must match the "
            "kernel: 2x the spacing for a cubic spline, 3x for a quintic.",
            "The upstream deck this derives from stops while its gravity ramp "
            "is still at 35% of g, so it never reaches equilibrium; the ramp "
            "was shortened here.",
        ),
    ),
    Deck(
        physics="particle_sph", variant="dam_break_2d",
        filename="particle_sph_dambreak.4C.yaml", np=2,
        upstream="particle_sph_2d_dambreak_freesurface_"
                 "densitynormalizedreinit.4C.yaml",
        summary="Two-dimensional dam break: a water column released onto a dry "
                "bed inside a closed tank, the standard free-surface SPH "
                "benchmark.",
        evidence="Surge front runs 0.375 -> 1.041 and column height drops "
                 "0.375 -> 0.241 while density stays within 0.3% of rho0 — "
                 "the collapse is resolved and nothing is blowing up.",
        pitfalls=(
            "TIMESTEP must stay below roughly 0.2*spacing/c with "
            "c = sqrt(BULK_MODULUS/rho0).",
            "Changing the kernel changes how many boundary layers the wall "
            "needs: two for a cubic spline, three for a quintic.",
        ),
    ),
    Deck(
        physics="pasi", variant="dem_impact_3d",
        filename="pasi.4C.yaml", np=2,
        upstream="pasi_twoway_norelax_particle_dem_1d_normalcontact_linspring_"
                 "walldiscretcond.4C.yaml",
        summary="Particle-structure interaction, partitioned two-way: a DEM "
                "sphere presses into an elastic plate, the plate deflects, and "
                "the deformed wall is fed back to the particle solver each "
                "coupling iteration.",
        evidence="The outer loop converges in all 150 steps without hitting "
                 "ITEMAX; the particle sinks 0.500000 -> 0.498574 while the "
                 "plate centre deflects uz = -1.0e-3 — force goes one way and "
                 "displacement the other.",
        pitfalls=(
            "PARTICLE_WALL_MOVING and PARTICLE_WALL_LOADED are what make the "
            "coupling two-way; with them false the structure is a rigid "
            "obstacle and PASI silently degenerates to one-way.",
            "Rayleigh K_DAMP scales with the stiffness: the upstream K_DAMP 1 "
            "gives the plate a relaxation time 13x the simulated time, so it "
            "never responds and the run still exits 0.",
        ),
    ),
    Deck(
        physics="plasticity", variant="linear_2d",
        filename="plasticity_linear_2d.4C.yaml", np=1,
        upstream="plastic_pressurisedcylinder.4C.yaml",
        summary="Small-strain J2 (von Mises) elastoplasticity with linear "
                "isotropic hardening, plane strain, displacement controlled "
                "past yield.",
        evidence="Reaction force 591 N against 4360 N for an identical deck "
                 "with the yield stress raised out of reach — a factor 7.4, "
                 "so the specimen is genuinely flowing plastically.",
        pitfalls=(
            "There is no 2-D plasticity element on this build. WALL QUAD4 with "
            "a plasticity material aborts in 4C_w1_mat.cpp:179 ('Invalid type "
            "of material law for wall element'); plane strain is done the way "
            "the upstream deck does it, one layer of SOLID HEX8 with u_z "
            "locked by a volume Dirichlet.",
            "Removing that u_z condition turns the deck into a 3-D bar without "
            "any diagnostic.",
        ),
    ),
    Deck(
        physics="plasticity", variant="nonlinear_3d",
        filename="plasticity_nonlinear_3d.4C.yaml", np=1,
        upstream="plastic_necking_eas.4C.yaml",
        summary="Finite-strain J2 elastoplasticity: the classic necking "
                "tensile bar, one eighth modelled with symmetry planes and a "
                "2% taper so localisation picks a plane deterministically.",
        evidence="Reaction force 5.417 against 221.8 for the raised-yield "
                 "control (factor 41); the load passes a maximum at step 18 "
                 "and the section contracts 12.8% against 7.7% elastic — "
                 "necking and isochoric plastic flow.",
        pitfalls=(
            "TECH eas_mild and TECH fbar abort with SIGFPE inside "
            "evaluate_eas_kinematics for this material (the same EAS with "
            "StVenantKirchhoff runs), so this ships TECH none and accepts HEX8 "
            "volumetric locking.",
            "The hardening law is Voce plus linear: "
            "sigma_y = YIELD + ISOHARD*e_p + (SATHARDENING - YIELD)*"
            "(1 - exp(-HARDEXPO*e_p)).",
        ),
    ),
    Deck(
        physics="porous_media", variant="terzaghi_2d",
        filename="porous_media_terzaghi_2d.4C.yaml", np=1,
        upstream="poro_2D_quad4_br_stsplit_nbc.4C.yaml + "
                 "poro_2D_quad4_linporo.4C.yaml",
        summary="Terzaghi one-dimensional consolidation: a saturated soil "
                "column loaded at the drained top surface, with the pore "
                "pressure carrying the load initially and dissipating over "
                "time as the skeleton takes it up.",
        evidence="Base pore pressure 9.903e-01 at the end of the load ramp "
                 "(99.0% of the applied q, the undrained Terzaghi limit) "
                 "falling monotonically to 6.322e-03; settlement -1.836e-04 -> "
                 "-8.957e-04, converging on the drained oedometric value "
                 "q*H/E_oed = 9.0e-04.",
        pitfalls=(
            "Poroelasticity requires the SAME THETA in STRUCTURAL "
            "DYNAMIC/ONESTEPTHETA and in FLUID DYNAMIC, or "
            "poroelast_base.cpp:182 throws 'porous media problem is limited in "
            "functionality'.",
            "Runtime VTK output needs STRUCTURAL DYNAMIC INT_STRATEGY: "
            "Standard; the old integrator throws 'Runtime output is not "
            "available in the old structure time integration!'",
            "In 2-D the porofluid VTU 'pressure' array is all NaN and the pore "
            "pressure lands in the THIRD component of 'velocity' — "
            "FluidImplicitTimeInt::write_runtime_output hardcodes three "
            "velocity components. The 3-D output is fine.",
        ),
    ),
    Deck(
        physics="porous_media", variant="consolidation_3d",
        filename="porous_media_consolidation_3d.4C.yaml", np=1,
        upstream="poro_3D_hex8_stat.4C.yaml + poro_2D_quad4_linporo.4C.yaml",
        summary="Three-dimensional consolidation under a surface load, the "
                "HEX8 counterpart of the Terzaghi column.",
        evidence="Base pressure 9.910e-01 -> 6.278e-03 and settlement "
                 "-1.833e-04 -> -8.957e-04, agreeing with the independent 2-D "
                 "deck to three digits — the right cross-check for a problem "
                 "that is one-dimensional in the physics.",
        pitfalls=(
            "There is no SOLIDH8PORO element. It appears in zero files of the "
            "4C source, zero upstream decks and is absent from the grammar "
            "index; earlier openPASO knowledge named it as the 3-D poro element. "
            "The real ones are SOLIDPORO_PRESSURE_VELOCITY_BASED (used here, "
            "and the only one the 25 upstream Poroelasticity decks use), "
            "SOLIDPORO_PRESSURE_VELOCITY_BASED_P1 (porosity as a 4th nodal "
            "unknown, needs PHYSICAL_TYPE: Poro_P1) and "
            "SOLIDPORO_PRESSURE_BASED (no fluid-velocity field; belongs to the "
            "pressure-based multiphase module, not to Poroelasticity).",
            "See the terzaghi_2d note on matching THETA.",
        ),
    ),
    Deck(
        physics="reduced_lung", variant="lung_1d",
        filename="reduced_lung.4C.yaml", np=2,
        upstream="reduced_lung_1d_pipe_flow_continuous.4C.yaml",
        summary="One-dimensional compliant-tube airway flow: a pressure pulse "
                "propagating along a pipe whose diameter halves midway, so it "
                "partially reflects at the area change.",
        evidence="Wave propagation and partial reflection resolved over 10000 "
                 "steps with the flow inlet and reflecting outlet active.",
        pitfalls=(
            "This is PROBLEMTYPE Reduced_Lung_1D_Pipe_Flow. The other lung "
            "problem type, Reduced_Lung (the lung-tree model), CANNOT be "
            "written as a single file on this build: its topology comes "
            "through from_file / from_mesh / field_reference, and the "
            "top-level `fields:` section only offers separate_file or "
            "from_mesh, so there is no way to give node coordinates inline. "
            "The `constant:` alternative assigns one value to every index, "
            "which collapses the tree — measured: 'Multiple pressure boundary "
            "conditions assigned to node 1', then SIGFPE from a zero reference "
            "volume once all nodes coincide.",
            "Nearly all the physics is in one lowercase top-level section, "
            "`reduced_lung:`, not in the usual upper-case DYNAMIC sections.",
        ),
    ),
    Deck(
        physics="ssi", variant="monolithic_elch_3d",
        filename="ssi.4C.yaml", np=1,
        upstream="ssi_2D_quad4.4C.yaml",
        summary="Structure-scalar interaction: a scalar transported on a "
                "deforming mesh, with the scatra discretisation cloned from "
                "the structure so both fields share nodes.",
        evidence="Staggered solid-to-scatra loop runs to completion with the "
                 "scalar transported in conservative form on the stretched "
                 "element.",
        pitfalls=(
            "The structure elements must be a *SCATRA type (WALLSCATRA, "
            "SOLIDSCATRA, …) with a meaningful TYPE. A plain WALL/SOLID aborts "
            "in 4C_ssi_clonestrategy.cpp:97, naming ImplType 'Undefined'.",
            "COUPALGO chooses one-way, staggered or monolithic; the one-way "
            "variants run happily and simply do not feed the scalar back.",
        ),
    ),
    Deck(
        physics="ssti", variant="monolithic_3d",
        filename="ssti.4C.yaml", np=1,
        upstream="ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_"
                 "growthlaw.4C.yaml",
        summary="Structure-scalar-thermo interaction: a 1-D lithium-ion cell "
                "(anode / electrolyte / cathode) solved monolithically for "
                "displacement, lithium concentration, potential and "
                "temperature at once.",
        evidence="Four-field monolithic Newton converges with Butler-Volmer "
                 "kinetics plus thermal contact resistance active on both "
                 "electrode-electrolyte interfaces.",
        pitfalls=(
            "S2I kinetics and SSTI interface meshtying are separate condition "
            "families and both are needed at each interface.",
            "Electrode swelling comes from the inelastic factors of "
            "MAT_MultiplicativeSplitDefgradElastHyper, not from a thermal "
            "expansion coefficient on the elastic material.",
        ),
    ),
    Deck(
        physics="sti", variant="monolithic_3d",
        filename="sti.4C.yaml", np=1,
        upstream="sti_mono_3D_hex8_elch_s2i_butlervolmerpeltier_adiabatic_"
                 "mortar_standard.4C.yaml",
        summary="Monolithic scatra-thermo interaction: electrochemistry "
                "(lithium concentration plus potential) and temperature solved "
                "in one Newton system, coupled by Butler-Volmer-Peltier "
                "kinetics across non-conforming mortar interfaces.",
        evidence="Cell voltage 3.888 -> 3.728 V over 20 s at C rate 10, SOC "
                 "100% -> 94.4/95.8%, interface current density exactly the "
                 "applied -2.4586e-05, and non-zero Peltier and Joule heat "
                 "fluxes across both interfaces.",
        pitfalls=(
            "Without ELCH CONTROL the run stops with 'Invalid type of closing "
            "equation for electric potential'.",
            "Set SORET to 0 and the species field still moves, not just the "
            "temperature — Soret is a cross-coupling, not the whole coupling.",
        ),
    ),
    Deck(
        physics="xfem_fluid", variant="xfem_3d",
        filename="xfem_fluid.4C.yaml", np=2,
        upstream="xfluid_ls_neumann_inflow_stab.4C.yaml",
        summary="XFEM fluid: transient incompressible Navier-Stokes on a fixed "
                "Eulerian mesh cut by a level-set circle, with a Neumann "
                "traction imposed on the embedded interface.",
        evidence="Cut elements are enriched and the level-set Neumann "
                 "condition is integrated on the embedded circle over the "
                 "whole time loop.",
        pitfalls=(
            "The interface is a FUNCT level set, so refining the interface "
            "means refining FLUID DOMAIN subdivisions — there is no interface "
            "mesh to refine.",
            "Ghost-penalty and mass-conservation parameters live in XFLUID "
            "DYNAMIC/STABILIZATION, separately from the ordinary fluid "
            "stabilisation.",
        ),
    ),
    Deck(
        physics="fluid_turbulence", variant="les_channel_3d",
        filename="fluid_turbulence.4C.yaml", np=2,
        upstream="f3_cha_8x8x8_recongradl2.4C.yaml + "
                 "f3_stokes_residualbased_rotboxgeom.4C.yaml",
        summary="Large-eddy simulation of turbulent channel flow of height 2: "
                "Smagorinsky subgrid model, periodic in x and z, driven by a "
                "constant streamwise body force.",
        evidence="4C reports 'Turbulence model : Smagorinsky with Smagorinsky "
                 "constant Cs= 0.1' and opens plane-and-time averaged channel "
                 "statistics over the sampling window.",
        pitfalls=(
            "On ONE MPI rank this deck aborts inside "
            "Core::Conditions::PeriodicBoundaryConditions::balance_load with "
            "'terminate called after throwing an instance of int' (SIGABRT, "
            "rc 134). It runs clean on 2 and 4 ranks; reproduced three times. "
            "Periodic boundary conditions need np > 1 on this build.",
            "Every boundary here is periodic or Dirichlet, so the pressure has "
            "a null space. Without DESIGN VOL MODE FOR KRYLOV SPACE PROJECTION "
            "the run stops with 'Nullspace check for sysmat_ failed'.",
            "8x8x8 is a demonstration resolution. An LES on an under-resolved "
            "mesh runs and produces wrong statistics silently.",
        ),
    ),
    Deck(
        physics="fs3i", variant="fs3i_3d",
        filename="fs3i.4C.yaml", np=1,
        upstream="fsi_fp_mono_fs_ga_ga.4C.yaml + fs3i_part_1wc_finperm.4C.yaml",
        summary="Fluid-structure-scalar interaction: a scalar transported in "
                "the fluid and in the solid, exchanging mass across the FSI "
                "interface through a permeability condition.",
        evidence="Fluid scalar 1.000 -> 0.934/0.983 and solid scalar 0.000 -> "
                 "0.054/0.089 over 10 steps, measured from the scatra1/scatra2 "
                 "VTU output; ALE displacement reaches 1.0.",
        requires_fourc_root=True,
        pitfalls=(
            "FS3I rejects direct solvers. 4C_fs3i_partitioned.cpp:604 throws "
            "'Iterative solver expected' unless COUPLED_LINEAR_SOLVER names a "
            "Belos solver, :610 demands AZPREC Teko, and "
            "4C_linear_solver_preconditioner_teko.cpp:48 then throws "
            "'TEKO_XML_FILE parameter not set!'. This is the one deck here "
            "that cannot use UMFPACK and cannot be a single file.",
            "A plain SOLID element aborts in 4C_ssi_clonestrategy.cpp:97 — the "
            "structure elements must be SOLIDSCATRA / WALLSCATRA / SHELLSCATRA "
            "/ TRUSS3SCATRA carrying a meaningful TYPE.",
            "Np_Gen_Alpha for the fluid aborts in 4C_fs3i.cpp:204; BDF2 and "
            "Stationary are rejected as well. One_Step_Theta in all three "
            "fields works.",
        ),
    ),
    Deck(
        physics="fsi", variant="fsi_2d",
        filename="fsi_2d.4C.yaml", np=1,
        upstream="volmortar2D_fsi.4C.yaml + fsi_fp_mono_fs_ga_ga.4C.yaml",
        summary="Partitioned (Dirichlet-Neumann) 2-D fluid-structure "
                "interaction: an elastic block pushed on its far edge drives "
                "an incompressible channel flow on a deforming ALE mesh.",
        evidence="10 coupled steps, FSI outer loop converging with a non-zero "
                 "interface increment (dx 7.7e-05); structure, fluid and ALE "
                 "result files all written.",
        pitfalls=(
            "A Dirichlet FUNCT must be a SYMBOLIC_FUNCTION_OF_SPACE_TIME. "
            "Giving it a SYMBOLIC_FUNCTION_OF_TIME aborts in "
            "4C_utils_function_manager.hpp:143 with 'You tried to query "
            "function 1 as a function of type FunctionOfSpaceTime. Actually, "
            "it has type FunctionOfTime.'",
            "With COUPALGO iter_monolithicfluidsplit the FLUID is the slave "
            "field and may carry no Dirichlet condition on interface dofs — "
            "4C_fsi_monolithicfluidsplit.cpp:135 prints a boxed diagnostic "
            "naming master and slave.",
        ),
    ),
    Deck(
        physics="particle_pd", variant="plate_2d",
        filename="particle_pd_plate.4C.yaml", np=1,
        upstream="none — no bond-based PD deck exists upstream; built from the "
                 "grammar dump and 4C's own PD generator script",
        summary="Bond-based peridynamics: a pre-cracked plate pulled in "
                "tension until the crack runs from the notch tip.",
        evidence="pd_damage_phi mean 0.1203 at step 0 (the pre-crack alone) "
                 "rising to 0.2102 at step 200 over 144 particles — bonds "
                 "break beyond the notch.",
        pitfalls=(
            "PD is not a separate interaction mode. INTERACTION must be SPH "
            "and PD_BODY_INTERACTION true; the PD parameters then live in "
            "PARTICLE DYNAMIC/PD.",
            "A 2-D PD_DIMENSION requires PARTICLE DYNAMIC/INITIAL AND BOUNDARY "
            "CONDITIONS CONSTRAINT: Projection2D. Without it 4C aborts in "
            "4C_particle_interaction_sph_peridynamic.cpp:92 with 'Plane stress "
            "or plane strain for peridynamic requested. CONSTRAINT must be set "
            "to Projection2D!'",
            "PDFIXED 1 pins a particle at its reference position; PDFIXED 2 "
            "makes it part of a rigid body moved at IMPACTOR_VELOCITY.",
        ),
    ),
    Deck(
        physics="particle_pd", variant="impact_2d",
        filename="particle_pd_impact.4C.yaml", np=1,
        upstream="none — see plate_2d",
        summary="Kalthoff-Winkler edge impact: a rigid impactor strikes a "
                "doubly-notched plate between the notches.",
        evidence="Two peridynamic bodies present; damage mean 0.0825 -> 0.2070 "
                 "over 300 steps and particle speeds reaching 4.6e4 mm/s.",
        pitfalls=(
            "The impactor is a second PDBODYID whose particles carry PDFIXED 2; "
            "contact between bodies is the NORMALCONTACTLAW / NORMAL_STIFF "
            "penalty pair in PARTICLE DYNAMIC/PD.",
            "dx = 12.5 mm here is a teaching resolution. The published "
            "Kalthoff-Winkler study resolves the same specimen at dx = 0.5 mm; "
            "crack paths at this spacing are indicative only.",
        ),
    ),
)


_BY_KEY = {(d.physics, d.variant): d for d in DECKS}


def get(physics: str, variant: str) -> Deck | None:
    return _BY_KEY.get((physics, variant))


def render(physics: str, variant: str) -> str | None:
    """Return the deck text with a short provenance header, or None."""
    d = get(physics, variant)
    if d is None:
        return None
    head = [
        f"# 4C {d.physics} / {d.variant} — runnable template",
        f"# {d.summary}",
        f"# Verified: executed on the installed 4C binary with "
        f"{d.np} MPI rank{'s' if d.np > 1 else ''}, exit 0.",
        f"# Evidence the physics is live: {d.evidence}",
        f"# Derived from upstream deck(s): {d.upstream}",
    ]
    if d.requires_fourc_root:
        head.append(
            "# NOTE: this deck names a file from the 4C source tree; set "
            "FOURC_ROOT so the path resolves.")
    for p in d.pitfalls:
        head.append(f"# Pitfall: {p}")
    return "\n".join(head) + "\n" + d.text()


def variants_for(physics: str) -> list[str]:
    return [d.variant for d in DECKS if d.physics == physics]


def physics_covered() -> list[str]:
    seen: list[str] = []
    for d in DECKS:
        if d.physics not in seen:
            seen.append(d.physics)
    return seen
