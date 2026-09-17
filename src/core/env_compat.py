"""One variable, two spellings.

The project was called Open-FEM-agent, then OASiS, and its environment variables
were named ``OFA_*`` and ``OASIS_*``. It is now called openPASO and they are named
``OPENPASO_*``.
Both spellings have to keep working, because they are read across a process
boundary: the server writes the mesh keys into a participant's environment,
and the participant that reads them may be a script somebody wrote, saved and
still runs from a year ago. A rename that only touched this repository would
break those quietly, at the far end, with an empty value.

So at start-up every ``OASIS_X`` and ``OFA_X`` gains a twin ``OPENPASO_X``, and every
``OPENPASO_X`` gains twins under both older names, unless a twin is already set --
an explicitly set value always wins over one inferred from another name.

This file is deliberately excluded from ``scripts/rename_to_openpaso.py``:
it is the one place that must go on saying both names.
"""
from __future__ import annotations

import os
from typing import MutableMapping

LEGACY_PREFIX = "OASIS_"
LEGACY_PREFIXES = ("OASIS_", "OFA_")
CURRENT_PREFIX = "OPENPASO_"


def _twins(name: str) -> list[str]:
    """Every other spelling of `name`; empty if it is not one of ours."""
    for prefix in (CURRENT_PREFIX, *LEGACY_PREFIXES):
        if name.startswith(prefix):
            stem = name[len(prefix):]
            return [p + stem for p in (CURRENT_PREFIX, *LEGACY_PREFIXES) if p != prefix]
    return []


def _twin(name: str) -> str | None:
    """The first other spelling of `name`, or None if it is not one of ours."""
    twins = _twins(name)
    return twins[0] if twins else None


def install_aliases(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Give every one of our variables both spellings. Returns the names added.

    Idempotent, and safe to call again after the environment has changed.
    """
    env = os.environ if environ is None else environ
    added: list[str] = []
    for name, value in list(env.items()):        # snapshot: we mutate as we go
        for twin in _twins(name):
            if twin not in env:
                env[twin] = value
                added.append(twin)
    return added


def set_env(name: str, value: str,
            environ: MutableMapping[str, str] | None = None) -> None:
    """Set one of our variables under both spellings at once."""
    env = os.environ if environ is None else environ
    env[name] = value
    for twin in _twins(name):
        env[twin] = value


def get_env(name: str, default: str | None = None,
            environ: MutableMapping[str, str] | None = None) -> str | None:
    """Read one of our variables under either spelling, current name first."""
    env = os.environ if environ is None else environ
    if name in env:
        return env[name]
    for twin in _twins(name):
        if twin in env:
            return env[twin]
    return default
