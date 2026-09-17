"""Generator registry for deal.II physics templates.

Each generator module exposes:
  - A template function  ``generate(params) -> str``
  - A knowledge dict     ``KNOWLEDGE: dict``
  - A template key list  ``TEMPLATE_KEYS: list[str]``  (physics_variant keys it provides)

The registry is populated lazily so missing modules don't prevent startup.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Maps template key ("physics_variant") to (module_path, function_name).
_TEMPLATE_SPECS: dict[str, tuple[str, str]] = {
    # poisson
    "poisson_2d":                (".poisson",              "_poisson_2d"),
    "poisson_3d":                (".poisson",              "_poisson_3d"),
    "poisson_l_domain":          (".poisson",              "_poisson_l_domain"),
    "poisson_rectangle":         (".poisson",              "_poisson_rectangle"),
    "poisson_2d_adaptive":       (".poisson",              "_poisson_adaptive_2d"),
    # 3D mixed Dirichlet-Neumann MMS convergence study (step-7 pattern)
    "poisson_3d_mixed_bc":       (".poisson_mixed_bc",     "_poisson_3d_mixed_bc"),
    # elasticity
    "linear_elasticity_2d":      (".elasticity",           "_elasticity_2d"),
    "linear_elasticity_thick_beam": (".elasticity",        "_elasticity_thick_beam"),
    # heat
    "heat_2d_transient":         (".heat",                 "_heat_2d_transient"),
    "heat_2d_steady":            (".heat",                 "_heat_2d_steady"),
    "heat_rectangle":            (".heat",                 "_heat_rectangle"),
    # stokes
    "stokes_2d":                 (".stokes",               "_stokes_2d"),
    # convection diffusion
    "convection_diffusion_2d":   (".convection_diffusion", "_convdiff_2d"),
    # nonlinear
    "nonlinear_2d_minimal_surface": (".nonlinear",         "_nonlinear_minimal_surface_2d"),
    # helmholtz
    "helmholtz_2d":              (".helmholtz",            "_helmholtz_2d"),
    # eigenvalue
    "eigenvalue_2d":             (".eigenvalue",           "_eigenvalue_2d"),
    # wave
    "wave_2d":                   (".wave",                 "_wave_2d"),
    # hp adaptive
    "hp_adaptive_2d":            (".hp_adaptive",          "_hp_adaptive_2d"),
    # dg transport
    "dg_transport_2d":           (".dg_transport",         "_dg_transport_2d"),
    # hyperelasticity
    "hyperelasticity_3d":        (".hyperelasticity",      "_hyperelasticity_3d"),
    # parallel
    "parallel_poisson_2d":       (".parallel",             "_parallel_poisson_2d"),
    # Navier-Stokes
    "navier_stokes_2d":          (".navier_stokes",        "_navier_stokes_2d"),
    # Advanced physics — every entry below is a REAL, verified generator
    # (compiled + run + .vtu confirmed on deal.II 9.1.1, overhaul 2026-06-26).
    # The print-and-exit placeholders for compressible_euler, multiphysics,
    # topology_opt, cg_dg_coupled and optimal_control were removed: they could
    # not be made runnable with reasonable effort on the supported deal.II.
    "mixed_laplacian_2d":        (".advanced",             "_mixed_laplacian_2d"),
    "time_dependent_heat_2d":    (".advanced",             "_time_dependent_heat_2d"),
    "time_dependent_wave_2d":    (".advanced",             "_time_dependent_wave_2d"),
    "time_dependent_ns_2d":      (".advanced",             "_time_dependent_ns_2d"),
    "matrix_free_2d":            (".advanced",             "_matrix_free_2d"),
    "multigrid_2d":              (".advanced",             "_multigrid_2d"),
    "obstacle_problem_2d":       (".advanced",             "_obstacle_2d"),
    "error_estimation_2d":       (".advanced",             "_error_estimation_2d"),
    "phase_field_2d":            (".advanced",             "_phase_field_2d"),
    "dg_advection_reaction_2d":  (".advanced",             "_dg_advection_2d"),
    # ── 2026-06-01: Catalog physics that the backend's
    #    supported_physics() advertised but had no template
    #    dispatch. They alias to the primary template-bearing
    #    physics that covers the same underlying tutorial:
    #
    #      advection_dg ↔ dg_advection_reaction (step-9/12)
    #      contact      ↔ obstacle_problem      (step-41/42)
    #      nonlinear_elasticity ↔ hyperelasticity (step-44)
    #
    #    The aliasing avoids duplicating ~500 lines of C++ per
    #    template. Each aliased physics still has its own
    #    PhysicsCapability entry (so it appears in discover()
    #    and the docs distinguish the framing) and its own
    #    knowledge dict (_DEALII_KNOWLEDGE fallback), but
    #    generate_input() returns the same template. Reflects
    #    the comments in backend.py L319-342 ("Distinct from
    #    ... but related to ...").
    "advection_dg_2d":           (".advanced",             "_dg_advection_2d"),
    "contact_2d":                (".advanced",             "_obstacle_2d"),
    "nonlinear_elasticity_3d":   (".hyperelasticity",      "_hyperelasticity_3d"),
}

# Maps physics name to (module_path, dict_name) for knowledge.
_KNOWLEDGE_SPECS: dict[str, tuple[str, str]] = {
    "poisson":              (".poisson",              "KNOWLEDGE"),
    # Mixed-BC MMS study — knowledge lives in its own module; the
    # template key poisson_3d_mixed_bc is a variant of physics
    # "poisson" (capability row in backend.py), so per-physics
    # knowledge lookups for "poisson" still resolve to .poisson.
    "poisson_mixed_bc":     (".poisson_mixed_bc",     "KNOWLEDGE"),
    "linear_elasticity":    (".elasticity",           "KNOWLEDGE"),
    "heat":                 (".heat",                 "KNOWLEDGE"),
    "stokes":               (".stokes",               "KNOWLEDGE"),
    "convection_diffusion": (".convection_diffusion", "KNOWLEDGE"),
    "nonlinear":            (".nonlinear",            "KNOWLEDGE"),
    "helmholtz":            (".helmholtz",            "KNOWLEDGE"),
    "eigenvalue":           (".eigenvalue",           "KNOWLEDGE"),
    "wave":                 (".wave",                 "KNOWLEDGE"),
    "hp_adaptive":          (".hp_adaptive",          "KNOWLEDGE"),
    "dg_transport":         (".dg_transport",         "KNOWLEDGE"),
    "hyperelasticity":      (".hyperelasticity",      "KNOWLEDGE"),
    "parallel_poisson":     (".parallel",             "KNOWLEDGE"),
    "navier_stokes":        (".navier_stokes",        "KNOWLEDGE"),
    "mixed_laplacian":      (".advanced",             "KNOWLEDGE"),
    "time_dependent_heat":  (".advanced",             "KNOWLEDGE"),
    "time_dependent_wave":  (".advanced",             "KNOWLEDGE"),
    "time_dependent_ns":    (".advanced",             "KNOWLEDGE"),
    "matrix_free":          (".advanced",             "KNOWLEDGE"),
    "multigrid":            (".advanced",             "KNOWLEDGE"),
    "obstacle_problem":     (".advanced",             "KNOWLEDGE"),
    "error_estimation":     (".advanced",             "KNOWLEDGE"),
    "phase_field":          (".advanced",             "KNOWLEDGE"),
    "dg_advection_reaction": (".advanced",            "KNOWLEDGE"),
    "_general":             (".poisson",              "GENERAL_KNOWLEDGE"),
}

# Caches
_TEMPLATE_CACHE: dict[str, Callable] = {}
_KNOWLEDGE_CACHE: dict[str, dict] = {}


def get_template(key: str) -> Callable:
    """Return the template generator function for a given key."""
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]

    if key not in _TEMPLATE_SPECS:
        raise ValueError(
            f"No deal.II template for {key}. "
            f"Available: {sorted(_TEMPLATE_SPECS)}"
        )

    module_path, func_name = _TEMPLATE_SPECS[key]
    mod = importlib.import_module(module_path, package=__name__)
    func = getattr(mod, func_name)
    _TEMPLATE_CACHE[key] = func
    return func


def get_knowledge(physics: str) -> dict:
    """Return knowledge dict for a physics type.

    If the physics's KNOWLEDGE['elements'] is a dict (post-refactor
    shape, ``{class_name: applicability_note}``), the entries are
    joined with the canonical ``element_catalog.ELEMENTS`` records
    and returned as a list of rich dicts. Legacy list-form entries
    are returned verbatim — both shapes coexist during migration
    (senior-AI-scientist critic 2026-05-31: canonical-element
    refactor).
    """
    if physics in _KNOWLEDGE_CACHE:
        return _KNOWLEDGE_CACHE[physics]

    if physics not in _KNOWLEDGE_SPECS:
        return {}

    module_path, dict_name = _KNOWLEDGE_SPECS[physics]
    try:
        mod = importlib.import_module(module_path, package=__name__)
        knowledge = getattr(mod, dict_name)
    except (ImportError, AttributeError) as exc:
        logger.debug("Cannot load knowledge for %r: %s", physics, exc)
        return {}

    # Several specs (notably in .advanced) point at an UMBRELLA
    # KNOWLEDGE dict whose keys are the physics names and whose
    # values are the per-physics blocks. Without this unwrap step
    # get_knowledge('mixed_laplacian') returns the umbrella dict
    # (top-level keys = other physics names, no 'pitfalls'/
    # 'description' at the top level) — so prepare_simulation's
    # pitfall extraction silently sees zero pitfalls and the
    # LLM client gets nothing actionable. (Audit 2026-06-01:
    # 14 deal.II physics affected — mixed_laplacian,
    # time_dependent_*, matrix_free, multigrid, ...)
    if (isinstance(knowledge, dict)
            and physics in knowledge
            and isinstance(knowledge[physics], dict)
            and ("pitfalls" in knowledge[physics]
                 or "description" in knowledge[physics])):
        knowledge = knowledge[physics]

    # Resolve elements / mesh_generators if in the structured (dict)
    # form; pass through if legacy (list) form. Done at retrieval
    # time, not import time, so each backend's catalog stays simple.
    if isinstance(knowledge, dict):
        from backends.dealii.element_catalog import (
            resolve_elements_section,
            resolve_mesh_generators_section,
        )
        patched = False
        new_knowledge = dict(knowledge)
        if "elements" in knowledge:
            resolved = resolve_elements_section(knowledge["elements"])
            if resolved is not None:
                new_knowledge["elements"] = resolved
                patched = True
        if "mesh_generators" in knowledge:
            resolved = resolve_mesh_generators_section(
                knowledge["mesh_generators"])
            if resolved is not None:
                new_knowledge["mesh_generators"] = resolved
                patched = True
        if patched:
            knowledge = new_knowledge

    # Prepend the install/build/API facts that decide whether ANY
    # deal.II program works. They are identical for every physics and
    # are NOT discoverable from a physics section, so serving them only
    # under some special topic= string means the consumer never sees
    # them. Inserted FIRST so it survives truncation, and only into
    # dicts that already carry pitfalls — injecting into an empty dict
    # would make an unknown physics look covered.
    if isinstance(knowledge, dict) and knowledge.get("pitfalls"):
        from backends.dealii.generators._critical import CRITICAL_KNOWLEDGE
        knowledge = {"deal_II_essentials": CRITICAL_KNOWLEDGE, **knowledge}

    _KNOWLEDGE_CACHE[physics] = knowledge
    return knowledge


def list_template_keys() -> list[str]:
    """Return sorted list of all available template keys."""
    return sorted(_TEMPLATE_SPECS)
