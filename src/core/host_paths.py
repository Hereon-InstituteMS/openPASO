"""Paths in served text belong to the reader's machine, not to ours.

The installed-version API reference is written by running each solver and
recording what worked. That recording necessarily contains absolute paths, and
for a long time those were the paths of the one machine the reference was made
on. A caveat was added telling the reader to substitute their own -- but a
caveat does not help a model, which reads
``/home/user/miniconda3/envs/fenics/bin/python`` and uses it verbatim. On
anyone else's computer that command cannot run, and the reader has been handed
a stranger's directory layout as though it were fact.

So the reference stores TOKENS, and this module fills them in at the moment the
text is served, in this order:

1. the environment variable the project already documents for that solver,
2. what autodiscovery actually found on this machine,
3. a readable placeholder that names the variable to set.

The third case is the important one: an honest ``<path to your dolfinx python
-- set FENICS_PYTHON>`` is better than a confident wrong path, because it tells
the reader what to do instead of sending them somewhere that does not exist.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable

# token -> (environment variable, discovery key, what to call it in a placeholder)
_TOKENS: dict[str, tuple[str | None, str | None, str]] = {
    "{PYTHON}":         (None,             None,      "the Python running openPASO"),
    "{FENICS_PYTHON}":  ("FENICS_PYTHON",  "fenics",  "your dolfinx Python"),
    "{DUNE_PYTHON}":    ("DUNE_PYTHON",    "dune",    "your DUNE-fem Python"),
    "{FOURC_BINARY}":   ("FOURC_BINARY",   "fourc",   "your 4C binary"),
    "{FOURC_ROOT}":     ("FOURC_ROOT",     None,      "your 4C source tree"),
    "{FEBIO_BINARY}":   ("FEBIO_BINARY",   "febio",   "your FEBio binary"),
    "{DEALII_BUILD}":   ("DEALII_DIR",     "dealii",  "your deal.II build tree"),
    "{DEALII_ROOT}":    ("DEALII_ROOT",    None,      "your deal.II source tree"),
    "{KRATOS_ROOT}":    ("KRATOS_ROOT",    None,      "your Kratos source tree"),
    "{SPARTA_BINARY}":  ("SPARTA_BINARY",  "sparta",  "your SPARTA binary"),
    "{SPARTA_ROOT}":    ("SPARTA_ROOT",    None,      "your SPARTA source tree"),
}


def _from_discovery(key: str) -> str | None:
    """What autodiscovery recorded for this backend on this machine, if anything."""
    try:
        from core.autodiscovery import load_discovered_config
    except ImportError:
        return None
    config = load_discovered_config()
    if not isinstance(config, dict):
        return None
    backends = config.get("backends")
    if not isinstance(backends, dict):
        return None
    entry = backends.get(key)
    if not isinstance(entry, dict):
        return None
    found = entry.get("location") or entry.get("source_root")
    return found if isinstance(found, str) and found else None


def _resolve_one(token: str) -> str:
    env_var, discovery_key, human = _TOKENS[token]
    if token == "{PYTHON}":
        return sys.executable or "python3"
    if env_var:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    if discovery_key:
        found = _from_discovery(discovery_key)
        if found:
            return found
    if env_var:                                  # last chance: on PATH?
        guess = shutil.which(env_var.split("_")[0].lower())
        if guess:
            return guess
        return f"<{human} -- set {env_var}>"
    return f"<{human}>"


def resolve(text: str) -> str:
    """Replace every host-path token in `text` with this machine's own value."""
    for token in _TOKENS:
        if token in text:
            text = text.replace(token, _resolve_one(token))
    return text


def tokens() -> tuple[str, ...]:
    """The tokens this module understands. Used by the test that guards the rule."""
    return tuple(_TOKENS)
