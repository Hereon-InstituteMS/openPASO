"""The user's own desktop folder, in their own language.

Solver checkouts are often kept on the desktop, and the desktop is not always
called Desktop: its name follows the system language (~/Bureau on a French
system, for example). Discovery used to spell out one developer's localised
desktop by name, which found that machine's installs and nobody else's. This
asks the system instead: XDG_DESKTOP_DIR from ~/.config/user-dirs.dirs (what
`xdg-user-dir DESKTOP` reads), then ~/Desktop.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def desktop_dirs() -> list[Path]:
    """Candidate desktop folders, most specific first, without duplicates."""
    home = Path.home()
    found: list[Path] = []
    env = os.environ.get("XDG_DESKTOP_DIR")
    if env:
        found.append(Path(os.path.expandvars(env)).expanduser())
    cfg = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "user-dirs.dirs"
    try:
        m = re.search(r'^\s*XDG_DESKTOP_DIR\s*=\s*"([^"]+)"', cfg.read_text(errors="ignore"), re.M)
    except OSError:
        m = None
    if m:
        found.append(Path(m.group(1).replace("$HOME", str(home))).expanduser())
    found.append(home / "Desktop")
    out: list[Path] = []
    for p in found:
        if p != home and p not in out:
            out.append(p)
    return out
