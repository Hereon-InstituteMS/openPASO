"""Kratos GeoMechanics generators and knowledge.

Covers soil mechanics, consolidation, groundwater flow, slope stability.
Application: GeoMechanicsApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _geomechanics_2d generator
# was an availability-probe stub (import-check + {"note": ...}, no solver
# run), so 'geomechanics' was removed from the generator registry and from
# KratosBackend.supported_physics(). That removal was right and stands.
#
# CORRECTION (2026-09-19): the second half of that note -- "GeoMechanics-
# Application is NOT importable in the installed Kratos stack" -- IS NO
# LONGER TRUE, and the claim outlived the condition it described. Measured
# on this machine through the interpreter openPASO itself resolves for
# Kratos (open-fem-agent/.venv, Kratos 10.3.0):
#
#     import KratosMultiphysics.GeoMechanicsApplication   -> imports fine
#
# and so do FluidDynamics, FSI, CoSimulation and Meshing. Seventeen
# applications are installed. So five capabilities are currently WITHHELD on
# the strength of a statement that has gone stale -- the mirror image of
# advertising one we do not have, and just as wrong.
#
# The KNOWLEDGE below was re-verified against the installed binary the same
# day and is ACCURATE: GeoLinearElasticPlaneStrain2DLaw,
# GeoIncrementalLinearElastic3DLaw, GeoIncrementalLinearElasticInterfaceLaw
# and LinearElastic2DInterfaceLaw all return True from
# KratosGlobals.Kernel.HasConstitutiveLaw, and UPwSmallStrainElement2D3N /
# 2D4N both construct without "is not registered!".
#
# WHAT IS STILL MISSING IS A REAL SOLVE, and the physics row stays out until
# one exists and has been executed. Re-registering it now would put back
# exactly the kind of row the June audit removed: a capability advertised
# with nothing behind it. The knowledge is ready; the generator is the work.
#
# MEASURED WHILE ATTEMPTING ONE, and worth having before the next attempt.
#
# The obvious property names do not exist at all: KM.DENSITY_SOLID,
# KM.BULK_MODULUS_SOLID, KM.BULK_MODULUS_FLUID and KM.PERMEABILITY_XX/YY/XY
# are absent from BOTH modules -- AttributeError, not a silent default, which
# is the good case.
#
# CORRECTION (adversarial audit, same day): an earlier version of this note
# said the variables live "on the APPLICATION module, not the core one",
# copying the shape of the StructuralMechanics note. That is true for exactly
# two of the ten names. Measured on Kratos 10.3.0:
#     on KM (core): DENSITY, BULK_MODULUS, PERMEABILITY, PERMEABILITY_WATER,
#                   DENSITY_WATER, POROSITY, DYNAMIC_VISCOSITY, WATER_PRESSURE
#     on Geo only:  PERMEABILITY_MATRIX, DT_WATER_PRESSURE
# So the rule is NOT "geomechanics variables are on the application module".
# There is no rule: check each name against both, which costs one hasattr.
# Generalising from one backend's habit to another is how the wrong half of
# this note got written.
#
# ATTEMPTED AND STOPPED, 2026-09-19, with what was measured, so the next
# attempt does not repeat it. A hand-built U-Pw model part -- nodes,
# UPwSmallStrainElement2D4N, GeoLinearElasticPlaneStrain2DLaw, the three
# DOFs, drained top and clamped base -- RUNS AND RETURNS NaN WATER PRESSURE
# AT EVERY NODE. It does not raise, it does not warn, and the displacement
# field looks plausible while the pressure is entirely NaN.
#
# The pressure block is singular for a STRUCTURAL reason, not a scaling one.
# Ruled out by execution rather than by argument, holding everything else
# fixed: permeability 1e-12, 1e-6 and 1e-3, time step 1 and 100, a static
# scheme and the application's own BackwardEulerQuasistaticUPwScheme. All
# four combinations return NaN. So it is not a stiff-system or a
# vanishing-diffusion problem, and raising the permeability does not help.
#
# THE CONCLUSION FOR ANYONE WRITING THIS NEXT: do not drive GeoMechanics by
# assembling a model part in Python. The application is designed around
# GeoMechanicsAnalysis (an AnalysisStage subclass) reading ProjectParameters
# and a .mdpa mesh, and that is the route that gets the ProcessInfo flags,
# the variable set and the solver wiring right. The hand-built route is how
# StructuralMechanics and FluidDynamics work in this repo, and it does NOT
# generalise here.
#
# Two further facts from the attempt, both general to Kratos:
#   * A MODEL PART NAME CONTAINING A DOT IS PARSED AS A PATH. Naming a part
#     "soil1e-06_1.0" gives "Error: Calling the method of the sub model part
#     0", which names neither the part nor the dot. Kratos uses "." as the
#     sub-model-part separator.
#   * Kratos reports a NaN solution as a successful solve. Nothing in the
#     return value or the echo distinguishes it. Read a field back and test
#     it is finite; that check costs one line and it is the only thing
#     between you and a confident wrong answer.
#
# (A genuine saturated-porous-media consolidation solve exists separately as
# the 'poromechanics' physics in specialized.py.)


KNOWLEDGE = {
    "geomechanics": {
        "description": "Geomechanics: soil mechanics, consolidation, groundwater flow, slope stability",
        "application": "GeoMechanicsApplication (pip install KratosGeoMechanicsApplication)",
        "elements": {
            "2D": ["UPwSmallStrainElement2D3N", "UPwSmallStrainElement2D4N",
                   "UPwSmallStrainElement2D6N", "UPwSmallStrainElement2D8N",
                   "UPwSmallStrainElement2D9N", "UPwSmallStrainElement2D10N",
                   "UPwSmallStrainElement2D15N"],
            "3D": ["UPwSmallStrainElement3D4N", "UPwSmallStrainElement3D8N",
                   "UPwSmallStrainElement3D10N", "UPwSmallStrainElement3D20N",
                   "UPwSmallStrainElement3D27N"],
            "interface": ["UPwSmallStrainInterfaceElement2D4N", "UPwSmallStrainInterfaceElement3D6N",
                          "UPwSmallStrainInterfaceElement3D8N"],
        },
        # Real registered names from KratosGeoMechanicsApplication
        # binary scan (libKratosGeoMechanicsCore.so).
        # CAVEAT: ModifiedCamClay and DruckerPrager were in the
        # prior catalog but DO NOT exist as registered laws in
        # GeoMechanicsApplication at all — see pitfall #0.
        "constitutive_laws": [
            "GeoLinearElasticPlaneStrain2DLaw",
            "GeoIncrementalLinearElastic3DLaw",
            "GeoIncrementalLinearElasticInterfaceLaw",
            "LinearElastic2DInterfaceLaw",
            "LinearElastic3DInterfaceLaw",
            "GeoMohrCoulombWithTensionCutOff2D",
            "GeoMohrCoulombWithTensionCutOff3D",
            "SmallStrainUDSM2DPlaneStrainLaw",
            "SmallStrainUDSM3DLaw",
            "SmallStrainUDSM2DInterfaceLaw",
            "SmallStrainUDSM3DInterfaceLaw",
            "TrussBackboneConstitutiveLaw",
        ],
        "solver_types": ["U-Pw (displacement-water pressure coupled)",
                         "Pw (groundwater flow only)", "U (structural only)"],
        "analysis_types": ["consolidation", "groundwater_flow", "slope_stability",
                           "excavation_staged", "dam_safety"],
        "pitfalls": [
            "[API] Kratos GeoMechanicsApplication 10.4.2 has the following CL families (verified via binary scan of libKratosGeoMechanicsCore.so): Geo-prefixed LinearElastic + Mohr-Coulomb-with-tension-cutoff variants, UDSM (user-defined soil model) variants, plus 2D/3D Interface laws and TrussBackboneConstitutiveLaw. NOTABLY ABSENT: no ModifiedCamClay anywhere; no DruckerPrager anywhere. The prior catalog listed both as available \u2014 they were never registered in this Application. Real Mohr-Coulomb is \"GeoMohrCoulombWithTensionCutOff2D\" (or 3D), NOT plain \"MohrCoulomb\". Linear elastic is \"GeoLinearElasticPlaneStrain2DLaw\" / \"GeoIncrementalLinearElastic3DLaw\", NOT \"LinearElastic2DPlaneStrain\". Signal: constitutive_law.name = \"ModifiedCamClay\" in the materials file raises, at ReadMaterialsUtility (so at materials-read time, BEFORE AnalysisStage.Initialize), RuntimeError 'Error: Kratos components missing \"ModifiedCamClay\"' \u2014 the name of the law is interpolated into the message, and the same text comes back for DruckerPrager, MohrCoulomb and LinearElastic2DPlaneStrain. TWO CORRECTIONS, both by execution. First, the previously quoted 'Trying to add a non registered ConstitutiveLaw' is emitted by nothing in this build and does not reproduce; a guard matching that text never fires. Second, LinearElastic3DLaw was listed here as rejected and it is NOT \u2014 with GeoMechanicsApplication loaded and StructuralMechanics absent it is ACCEPTED, as are GeoLinearElasticPlaneStrain2DLaw, GeoIncrementalLinearElastic3DLaw and GeoMohrCoulombWithTensionCutOff2D. (Verified by execution 2026-08-07 on Kratos 10.3.0 as installed here, one ReadMaterialsUtility call per name; supersedes the 2026-06-01 binary-scan note.)",
            "[Physics] GeoMechanicsApplication's U-Pw elements carry DISPLACEMENT together with a pressure DOF. Mind the APPLICATION split, which is easy to read as a version change and is not one: GeoMechanicsApplication registers the UPw* element stem with WATER_PRESSURE, while PoromechanicsApplication registers the UPl* stem with LIQUID_PRESSURE. Which spelling resolves depends on which application is imported. Signal: with only PoromechanicsApplication loaded, CreateNewElement on a UPw* name raises 'is not registered'; importing GeoMechanicsApplication makes the same call succeed, and the reverse holds for the UPl* stem.",
        ],
        "guidance": [
            "[Numerical] Gravity loading via body_force_per_unit_mass: [0, -9.81, 0]",
            "[Numerical] Initial stress state often needed via K0 procedure",
            "[Numerical] Time stepping critical for consolidation (geometric progression recommended)",
            "[Integration] Material parameters: use effective stress parameters, not total stress",
        ]
    },
}

# Empty because there is no REAL SOLVE here yet, not because the application
# is missing: it imports fine (see the correction at the top of this file).
# The prior generator was a no-solve probe stub and was rightly removed.
GENERATORS = {}
