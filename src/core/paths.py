"""Where openPASO's shipped data lives: ``<checkout>/data`` or ``openpaso/data``.

In a checkout the code sits in ``src/`` and the data beside it in ``data/``; in an
installed wheel the same code is ``openpaso/`` and the data ``openpaso/data/``
(setup.py maps it there). The four readers of shipped data -- the coupling
participants, the post-mortems and the deal.II and 4C knowledge modules -- ask
here instead of counting ``parents[...]`` from their own file, which gave the
checkout answer only.
"""
from __future__ import annotations

from pathlib import Path


def data_dir() -> Path:
    here = Path(__file__).resolve()
    installed = here.parents[1] / "data"     # openpaso/data
    checkout = here.parents[2] / "data"      # <repo>/data
    if installed.is_dir():
        return installed
    return checkout
