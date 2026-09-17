"""Kratos GeoMechanics generators and knowledge.

Covers soil mechanics, consolidation, groundwater flow, slope stability.
Application: GeoMechanicsApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _geomechanics_2d generator
# was an availability-probe stub (import-check + {"note": ...}, no solver
# run). GeoMechanicsApplication is NOT importable in the installed Kratos
# stack, so 'geomechanics' has been removed from the generator registry and
# from KratosBackend.supported_physics(). KNOWLEDGE retained for reference.
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
        # 10.4.2 binary scan (libKratosGeoMechanicsCore.so).
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
            "[API] Kratos GeoMechanicsApplication 10.4.2 has the following CL families (verified via binary scan of libKratosGeoMechanicsCore.so): Geo-prefixed LinearElastic + Mohr-Coulomb-with-tension-cutoff variants, UDSM (user-defined soil model) variants, plus 2D/3D Interface laws and TrussBackboneConstitutiveLaw. NOTABLY ABSENT: no ModifiedCamClay anywhere; no DruckerPrager anywhere. The prior catalog listed both as available \u2014 they were never registered in this Application. Real Mohr-Coulomb is \"GeoMohrCoulombWithTensionCutOff2D\" (or 3D), NOT plain \"MohrCoulomb\". Linear elastic is \"GeoLinearElasticPlaneStrain2DLaw\" / \"GeoIncrementalLinearElastic3DLaw\", NOT \"LinearElastic2DPlaneStrain\". Signal: constitutive_law.name = \"ModifiedCamClay\" in the materials file raises, at ReadMaterialsUtility (so at materials-read time, BEFORE AnalysisStage.Initialize), RuntimeError 'Error: Kratos components missing \"ModifiedCamClay\"' \u2014 the name of the law is interpolated into the message, and the same text comes back for DruckerPrager, MohrCoulomb and LinearElastic2DPlaneStrain. TWO CORRECTIONS, both by execution. First, the previously quoted 'Trying to add a non registered ConstitutiveLaw' is emitted by nothing in this 28-application build and does not reproduce; a guard matching that text never fires. Second, LinearElastic3DLaw was listed here as rejected and it is NOT \u2014 with GeoMechanicsApplication loaded and StructuralMechanics absent it is ACCEPTED, as are GeoLinearElasticPlaneStrain2DLaw, GeoIncrementalLinearElastic3DLaw and GeoMohrCoulombWithTensionCutOff2D. (Verified by execution 2026-08-07 on Kratos 10.4.3, one ReadMaterialsUtility call per name; supersedes the 2026-06-01 binary-scan note.)",
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

# Empty: GeoMechanicsApplication not installable in this Kratos stack; the
# prior generator was a no-solve probe stub (removed).
GENERATORS = {}
