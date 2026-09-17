"""SPARTA generator registry — maps physics_variant -> generator function.

Same layout as the FEM backends (skfem, fenics, ngsolve, febio, ...): one file
per SPARTA physics exposes a per-module ``GENERATORS`` + ``KNOWLEDGE`` dict and
this aggregator merges them into the top-level dicts ``backend.py`` consumes.

Why SPARTA needed this shape at all: the knowledge used to be a single flat
JSON dump — a 121-entry verbatim copy of the SPARTA doc index plus 37 upstream
example decks — served whole for every physics. One knowledge() call returned
tens of kilobytes of manual, an order of magnitude more than any other backend,
and the pitfalls sat at the very end, behind the client-side truncation point.
The entries below are attached to the physics they concern, ordered so the
load-bearing material arrives first, and every pitfall carries an observable
``Signal:`` clause. The verbatim command index stays in sparta_knowledge.json
as a lookup table for validate_input and for on-demand syntax queries; it is no
longer pushed at the agent on every call.
"""

from ._common import (
    BUILD_FACTS,
    DECK_SKELETON,
    HARD_ORDERING_ERRORS,
    READING_OUTPUT,
    SILENTLY_ACCEPTED,
    UNIVERSAL_PITFALLS,
)
from .adaptive_grid import GENERATORS as _adapt_gen, KNOWLEDGE as _adapt_kn
from .ambipolar_plasma import GENERATORS as _ambi_gen, KNOWLEDGE as _ambi_kn
from .axisymmetric import GENERATORS as _axi_gen, KNOWLEDGE as _axi_kn
from .chemistry import GENERATORS as _chem_gen, KNOWLEDGE as _chem_kn
from .collision_relaxation import (
    GENERATORS as _coll_gen,
    KNOWLEDGE as _coll_kn,
)
from .conjugate_heat_transfer import GENERATORS as _cht_gen, KNOWLEDGE as _cht_kn
from .hypersonic_flow import GENERATORS as _hyp_gen, KNOWLEDGE as _hyp_kn
from .particle_emission import GENERATORS as _emit_gen, KNOWLEDGE as _emit_kn
from .rarefied_flow import GENERATORS as _rare_gen, KNOWLEDGE as _rare_kn
from .surface_interaction import GENERATORS as _surf_gen, KNOWLEDGE as _surf_kn

GENERATORS: dict[str, callable] = {}
for _g in (_rare_gen, _coll_gen, _hyp_gen, _surf_gen, _chem_gen,
           _axi_gen, _emit_gen, _adapt_gen, _ambi_gen, _cht_gen):
    GENERATORS.update(_g)

KNOWLEDGE: dict[str, dict] = {}
for _k in (_rare_kn, _coll_kn, _hyp_kn, _surf_kn, _chem_kn,
           _axi_kn, _emit_kn, _adapt_kn, _ambi_kn, _cht_kn):
    KNOWLEDGE.update(_k)

# Attach the cross-cutting pitfalls to every physics. They are properties of a
# SPARTA deck rather than of a flow regime, so an agent that pulls one physics
# still gets them; they are appended AFTER the physics-specific ones so the
# most relevant material is read first.
for _phys in KNOWLEDGE.values():
    _phys["pitfalls"] = list(_phys.get("pitfalls", [])) + list(UNIVERSAL_PITFALLS)
    _phys.setdefault("deck_skeleton", DECK_SKELETON)

# Backend-level reference catalog, surfaced by knowledge(topic='overview').
KNOWLEDGE["_general"] = {
    "description": "SPARTA — Stochastic PArallel Rarefied-gas Time-accurate "
                   "Analyzer. Direct Simulation Monte Carlo: the Boltzmann "
                   "equation solved with simulator particles and probabilistic "
                   "collisions on a hierarchical Cartesian grid. Not a "
                   "continuum solver and not a finite element code.",
    "when_to_use": "Knudsen number above roughly 0.01: rarefied and "
                   "transitional flows, vacuum and micro-scale devices, "
                   "re-entry aerothermodynamics, and any case where the "
                   "velocity distribution is not close to Maxwellian. Below "
                   "that a Navier-Stokes solver is cheaper and more accurate.",
    "deck_skeleton": DECK_SKELETON,
    "build": BUILD_FACTS,
    "hard_ordering_errors": HARD_ORDERING_ERRORS,
    "silently_accepted": SILENTLY_ACCEPTED,
    "reading_output": READING_OUTPUT,
    "physics_rows": sorted(k for k in KNOWLEDGE if not k.startswith("_")),
}

__all__ = ["GENERATORS", "KNOWLEDGE", "UNIVERSAL_PITFALLS", "DECK_SKELETON",
           "READING_OUTPUT", "BUILD_FACTS", "HARD_ORDERING_ERRORS",
           "SILENTLY_ACCEPTED"]
