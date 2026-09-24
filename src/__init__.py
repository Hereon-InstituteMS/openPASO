"""openPASO -- open Platform for Agentic Simulation and Optimization.

Installed from PyPI this directory is the package ``openpaso``; in a checkout it
is ``src/`` and is put on sys.path by the editable install. The code base uses
flat imports (``core``, ``backends``, ``tools``, ``reporting``, ``blind_eval``,
``server``), so the package puts its own directory on sys.path once, at import,
and the same modules resolve either way. Nothing imports ``openpaso.core``: the
flat names are the only ones used, so no module is loaded twice.

``openpaso`` (the console script) starts the MCP server on stdio.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

__version__ = "1.3.0"


def main() -> None:
    """The ``openpaso`` command: the server by default, ``doctor`` and ``install`` beside it."""
    from cli import main as _cli  # noqa: E402  (flat import, see above)
    raise SystemExit(_cli())
