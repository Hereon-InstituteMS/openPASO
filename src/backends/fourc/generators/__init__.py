"""Generator registry for 4C physics modules.

Provides :func:`get_generator` and :func:`list_generators` for discovering
and retrieving physics-specific generator instances by key or alias.

Usage::

    from src.generators import get_generator, list_generators

    gen = get_generator("scalar_transport")
    gen = get_generator("heat")           # alias works too
    all_gens = list_generators()
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseGenerator

logger = logging.getLogger(__name__)

# ── Internal registry ──────────────────────────────────────────────────
#
# Maps a module name (str) to a tuple of (dotted-module-path, class-name).
# Each module is lazily imported on first access so that missing generator
# files don't prevent the rest of the system from working.

_GENERATOR_SPECS: dict[str, tuple[str, str]] = {
    "scalar_transport":    (".scalar_transport",    "ScalarTransportGenerator"),
    "solid_mechanics":     (".solid_mechanics",     "SolidMechanicsGenerator"),
    "structural_dynamics": (".structural_dynamics", "StructuralDynamicsGenerator"),
    "fluid":               (".fluid",               "FluidGenerator"),
    "fsi":                 (".fsi",                  "FSIGenerator"),
    "beams":               (".beams",               "BeamsGenerator"),
    "contact":             (".contact",             "ContactGenerator"),
    "particle_pd":         (".particle_pd",         "ParticlePDGenerator"),
    "particle_sph":        (".particle_sph",        "ParticleSPHGenerator"),
    "particle_dem":        (".particle_dem",        "ParticleDEMGenerator"),
    "porous_media":        (".porous_media",        "PorousMediaGenerator"),
    "tsi":                 (".tsi",                  "TSIGenerator"),
    "ssi":                 (".ssi",                  "SSIGenerator"),
    "ale":                 (".ale",                  "ALEGenerator"),
    "electrochemistry":    (".electrochemistry",    "ElectrochemistryGenerator"),
    "level_set":           (".level_set",           "LevelSetGenerator"),
    "low_mach":            (".low_mach",            "LowMachGenerator"),
    "ssti":                (".ssti",                 "SSTIGenerator"),
    "sti":                 (".sti",                  "STIGenerator"),
    "fbi":                 (".fbi",                  "FBIGenerator"),
    "fpsi":                (".fpsi",                 "FPSIGenerator"),
    "pasi":                (".pasi",                 "PASIGenerator"),
    "lubrication":         (".lubrication",          "LubricationGenerator"),
    "cardiac_monodomain":  (".cardiac_monodomain",   "CardiacMonodomainGenerator"),
    "arterial_network":    (".arterial_network",     "ArterialNetworkGenerator"),
    "xfem_fluid":          (".xfem_fluid",           "XFEMFluidGenerator"),
    "fsi_xfem":            (".fsi_xfem",             "FSIXFEMGenerator"),
    "fs3i":                (".fs3i",                  "FS3IGenerator"),
    "ehl":                 (".ehl",                   "EHLGenerator"),
    "reduced_airways":     (".reduced_airways",       "ReducedAirwaysGenerator"),
    "beam_interaction":    (".beam_interaction",      "BeamInteractionGenerator"),
    "multiscale":          (".multiscale",            "MultiscaleGenerator"),
    # New generators
    "membrane":            (".membrane",              "MembraneGenerator"),
    "shell":               (".shell",                 "ShellGenerator"),
    "thermo":              (".thermo",                "ThermoGenerator"),
    "thermo_transient_mms": (".thermo_transient_mms",
                             "ThermoTransientMMSGenerator"),
    "mixture":             (".mixture",               "MixtureGenerator"),
    "constraint":          (".constraint",            "ConstraintGenerator"),
    "brownian_dynamics":   (".brownian_dynamics",     "BrownianDynamicsGenerator"),
    "cardiovascular0d":    (".cardiovascular0d",      "Cardiovascular0DGenerator"),
    "reduced_lung":        (".reduced_lung",           "ReducedLungGenerator"),
    "fluid_turbulence":    (".fluid_turbulence",       "FluidTurbulenceGenerator"),
}

# Aliases map user-friendly names to the canonical module key.
_ALIASES: dict[str, str] = {
    # Scalar transport aliases
    "poisson":          "scalar_transport",
    "heat":             "scalar_transport",
    "heat_conduction":  "scalar_transport",
    "diffusion":        "scalar_transport",
    "scatra":           "scalar_transport",
    # Solid mechanics aliases
    "elasticity":       "solid_mechanics",
    "linear_elasticity": "solid_mechanics",
    "solid":            "solid_mechanics",
    "statics":          "solid_mechanics",
    "nonlinear_solid":  "solid_mechanics",
    # Plasticity is folded into solid_mechanics in 4C — the
    # plasticity_models / plasticity_pitfalls keys are on
    # the solid_mechanics generator's get_knowledge() output.
    # Without this alias, fourc.get_knowledge('plasticity')
    # returns {'error': "no knowledge for 'plasticity' in
    # fourc"} (the catalog advertises the physics in
    # supported_physics() so the orphan-audit test catches it).
    "plasticity":       "solid_mechanics",
    # Structural dynamics aliases
    "dynamics":         "structural_dynamics",
    "structural":       "structural_dynamics",
    "transient_solid":  "structural_dynamics",
    # Fluid aliases
    "cfd":              "fluid",
    "navier_stokes":    "fluid",
    "incompressible":   "fluid",
    # FSI aliases
    "fluid_structure":  "fsi",
    # Beam aliases
    "beam":             "beams",
    "kirchhoff":        "beams",
    # Contact aliases
    "mortar":           "contact",
    "penalty_contact":  "contact",
    # Peridynamics aliases
    "peridynamics":     "particle_pd",
    "pd":               "particle_pd",
    "bond_based":       "particle_pd",
    # SPH aliases
    "sph":              "particle_sph",
    "smoothed_particle": "particle_sph",
    # DEM aliases
    "dem":              "particle_dem",
    "discrete_element": "particle_dem",
    "granular":         "particle_dem",
    # Porous media aliases
    "porous":           "porous_media",
    "darcy":            "porous_media",
    "poro":             "porous_media",
    # TSI aliases
    "thermo_structure":          "tsi",
    "thermo_structure_interaction": "tsi",
    "thermal_expansion":         "tsi",
    "thermomechanical":          "tsi",
    # SSI aliases
    "structure_scalar":          "ssi",
    "structure_scalar_interaction": "ssi",
    "electrode_mechanics":       "ssi",
    "battery_mechanics":         "ssi",
    # ALE aliases
    "mesh_motion":               "ale",
    "mesh_movement":             "ale",
    "arbitrary_lagrangian_eulerian": "ale",
    # Electrochemistry aliases
    "elch":                      "electrochemistry",
    "nernst_planck":             "electrochemistry",
    "battery_electrolyte":       "electrochemistry",
    "ionic_transport":           "electrochemistry",
    # Level-set aliases
    "levelset":                  "level_set",
    "interface_tracking":        "level_set",
    "two_phase":                 "level_set",
    # Low Mach aliases
    "loma":                      "low_mach",
    "low_mach_number":           "low_mach",
    "variable_density":          "low_mach",
    "buoyancy":                  "low_mach",
    "natural_convection":        "low_mach",
    # SSTI aliases
    "structure_scalar_thermo":           "ssti",
    "structure_scalar_thermo_interaction": "ssti",
    "three_field":                       "ssti",
    "battery_cell":                      "ssti",
    # STI aliases
    "scalar_thermo":                     "sti",
    "scalar_thermo_interaction":         "sti",
    "thermodiffusion":                   "sti",
    "soret":                             "sti",
    # FBI aliases
    "fluid_beam":                        "fbi",
    "fluid_beam_interaction":            "fbi",
    "immersed_beam":                     "fbi",
    "fiber_flow":                        "fbi",
    # FPSI aliases
    "fluid_porous_structure":            "fpsi",
    "fluid_porous_structure_interaction": "fpsi",
    "fpsi_monolithic":                   "fpsi",
    "porous_flow_structure":             "fpsi",
    # PASI aliases
    "particle_structure":                "pasi",
    "particle_structure_interaction":    "pasi",
    "dem_structure":                     "pasi",
    "dem_impact":                        "pasi",
    # Lubrication aliases
    "reynolds":                          "lubrication",
    "reynolds_equation":                 "lubrication",
    "thin_film":                         "lubrication",
    "bearing":                           "lubrication",
    # Cardiac monodomain aliases
    "cardiac":                           "cardiac_monodomain",
    "monodomain":                        "cardiac_monodomain",
    "electrophysiology":                 "cardiac_monodomain",
    "cardiac_ep":                        "cardiac_monodomain",
    "action_potential":                   "cardiac_monodomain",
    # Arterial network aliases
    "arterial":                          "arterial_network",
    "artery":                            "arterial_network",
    "blood_flow_1d":                     "arterial_network",
    "pulse_wave":                        "arterial_network",
    "hemodynamics_1d":                   "arterial_network",
    # XFEM fluid aliases
    "xfem":                              "xfem_fluid",
    "fluid_xfem":                        "xfem_fluid",
    "embedded_boundary":                 "xfem_fluid",
    "cut_fem_fluid":                     "xfem_fluid",
    # FSI XFEM aliases
    "fsi_xfem_coupling":                 "fsi_xfem",
    "xfem_fsi":                          "fsi_xfem",
    "immersed_fsi":                      "fsi_xfem",
    "fixed_grid_fsi":                    "fsi_xfem",
    # FS3I aliases
    "fs3i_coupling":                     "fs3i",
    "fluid_structure_scalar":            "fs3i",
    "five_field":                        "fs3i",
    "drug_delivery":                     "fs3i",
    "mass_transfer_fsi":                 "fs3i",
    # EHL aliases
    "elastohydrodynamic":                "ehl",
    "ehl_coupling":                      "ehl",
    "lubrication_structure":             "ehl",
    "bearing_deformation":               "ehl",
    # Reduced airways aliases
    "airways":                           "reduced_airways",
    "lung":                              "reduced_airways",
    "lung_airways":                      "reduced_airways",
    "respiratory":                       "reduced_airways",
    "airway_tree":                       "reduced_airways",
    "redairway":                         "reduced_airways",
    # Beam interaction aliases
    "beam_contact":                      "beam_interaction",
    "beam_meshtying":                    "beam_interaction",
    "beam_to_beam":                      "beam_interaction",
    "beam_to_solid":                     "beam_interaction",
    "fiber_contact":                     "beam_interaction",
    # Multiscale aliases
    "fe2":                               "multiscale",
    "fe_squared":                        "multiscale",
    "computational_homogenisation":      "multiscale",
    "computational_homogenization":      "multiscale",
    "rve":                               "multiscale",
    "stru_multi":                        "multiscale",
    # Membrane aliases
    "inflatable":                        "membrane",
    "fabric":                            "membrane",
    "membrane_structure":                "membrane",
    # Shell aliases
    "plate":                             "shell",
    "thin_shell":                        "shell",
    "kirchhoff_love":                    "shell",
    "reissner_mindlin":                  "shell",
    # Thermo aliases (standalone, not TSI)
    "heat_standalone":                   "thermo",
    "thermal":                           "thermo",
    "heat_conduction_standalone":        "thermo",
    # Transient-thermo MMS aliases (temporal-convergence deck family)
    "transient_thermo_mms":              "thermo_transient_mms",
    "thermo_mms":                        "thermo_transient_mms",
    "unsteady_heat_mms":                 "thermo_transient_mms",
    # Mixture aliases
    "composite":                         "mixture",
    "fiber_reinforced":                  "mixture",
    "biological_tissue":                 "mixture",
    "growth_remodeling":                 "mixture",
    # Constraint aliases
    "mpc":                               "constraint",
    "rigid_body":                        "constraint",
    "periodic_bc":                       "constraint",
    "coupling_condition":                "constraint",
    # Brownian dynamics aliases
    "brownian":                          "brownian_dynamics",
    "fiber_network":                     "brownian_dynamics",
    "biopolymer":                        "brownian_dynamics",
    "actin":                             "brownian_dynamics",
    # Cardiovascular0D aliases
    "windkessel":                        "cardiovascular0d",
    "cardiovascular":                    "cardiovascular0d",
    "lumped_circulation":                "cardiovascular0d",
    "heart_model":                       "cardiovascular0d",
    # Reduced lung aliases
    "lung_model":                        "reduced_lung",
    "ventilation":                       "reduced_lung",
    "lung_ventilation":                  "reduced_lung",
    # Fluid turbulence aliases
    "les":                               "fluid_turbulence",
    "dns":                               "fluid_turbulence",
    "turbulence":                        "fluid_turbulence",
    "smagorinsky":                       "fluid_turbulence",
    "large_eddy":                        "fluid_turbulence",
}

# Cache of instantiated generators (populated lazily).
_GENERATORS: dict[str, "BaseGenerator"] = {}


def _resolve_key(key: str) -> str:
    """Resolve an alias to its canonical module key.

    Parameters
    ----------
    key : str
        A module key or alias (case-insensitive).

    Returns
    -------
    str
        The canonical module key.

    Raises
    ------
    KeyError
        If *key* is not a known module key or alias.
    """
    normalised = key.strip().lower()
    if normalised in _GENERATOR_SPECS:
        return normalised
    if normalised in _ALIASES:
        return _ALIASES[normalised]
    raise KeyError(
        f"Unknown generator key or alias: {key!r}. "
        f"Available keys: {sorted(_GENERATOR_SPECS)}. "
        f"Available aliases: {sorted(_ALIASES)}."
    )


def _load_generator(canonical_key: str) -> "BaseGenerator":
    """Lazily import and instantiate a generator by its canonical key."""
    if canonical_key in _GENERATORS:
        return _GENERATORS[canonical_key]

    module_path, class_name = _GENERATOR_SPECS[canonical_key]
    try:
        mod = importlib.import_module(module_path, package=__name__)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import generator module {module_path} for key "
            f"{canonical_key!r}: {exc}"
        ) from exc

    cls = getattr(mod, class_name)
    instance = cls()
    _GENERATORS[canonical_key] = instance
    return instance


# ── Public API ─────────────────────────────────────────────────────────


def get_generator(key: str) -> "BaseGenerator":
    """Retrieve a generator instance by module key or alias.

    Parameters
    ----------
    key : str
        Canonical key (e.g. ``"scalar_transport"``) or alias
        (e.g. ``"heat"``, ``"poisson"``).  Case-insensitive.

    Returns
    -------
    BaseGenerator
        The singleton generator instance for the requested module.

    Raises
    ------
    KeyError
        If *key* does not match any known generator or alias.
    ImportError
        If the generator module file has not been created yet.

    Examples
    --------
    >>> gen = get_generator("heat")
    >>> gen.module_key
    'scalar_transport'
    """
    canonical = _resolve_key(key)
    return _load_generator(canonical)


def list_generators() -> dict[str, "BaseGenerator"]:
    """Return a dict of all available generators (key -> instance).

    Only generators whose modules can be successfully imported are included.
    Generators that fail to import are logged as warnings and skipped.

    Returns
    -------
    dict[str, BaseGenerator]
        Mapping of canonical module key to generator instance.
    """
    result: dict[str, "BaseGenerator"] = {}
    for key in _GENERATOR_SPECS:
        try:
            result[key] = _load_generator(key)
        except ImportError:
            logger.debug(
                "Generator %r not yet available (module not implemented).", key
            )
    return result


def list_aliases() -> dict[str, str]:
    """Return a copy of the alias-to-canonical-key mapping.

    Returns
    -------
    dict[str, str]
        Mapping of alias string to canonical module key.
    """
    return dict(_ALIASES)


def available_keys() -> list[str]:
    """Return sorted list of all accepted keys (canonical + aliases).

    Returns
    -------
    list[str]
        Every string that :func:`get_generator` will accept.
    """
    return sorted(set(_GENERATOR_SPECS) | set(_ALIASES))
