"""Coupling failure modes, in the indexed pitfall-corpus format.

THIS IS NOT A BACKEND. Nothing here registers with `core.registry`, and
`get_backend('coupling')` returns None as it always did. The directory sits
under `src/backends/` for one reason, stated plainly so nobody has to guess:
that is the tree `scripts/convergence_ledger.py::measured_pitfall_coverage()`
walks to build the pitfall-coverage denominator. Coupling counted zero pitfall
claims against its tier-2 fixtures — not because it was verified, but because
its knowledge carried no Signal clause anywhere, so nothing could count it.
The ledger printed UNMEASURABLE for exactly that, and a capability with an
unmeasurable denominator is not passing or failing the 80% bar, it is exempt
from it. Putting the claims where the counter already looks is what ends the
exemption, and it needs no edit to the ledger.

See `pitfalls.py` for the entries and the reasoning behind their shape.
"""
from .pitfalls import (                                     # noqa: F401
    COUPLING_PITFALLS,
    coupling_failure_index,
    get_coupling_pitfalls,
)
