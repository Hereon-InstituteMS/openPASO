"""Nothing that identifies the person or the machine leaves the server.

Transcripts, file views and run records are things people screenshot, share
and attach to papers. Tool output routinely contains the home directory and the
user name (`pwd`, `ls -l`, tracebacks). Everything sent to a browser or
downloaded passes through `scrub`; the files on disk stay exactly as written.
"""
from __future__ import annotations

import getpass
import re
from pathlib import Path

_HOME = str(Path.home())
try:
    _USER = getpass.getuser()
except Exception:
    _USER = ""

_HOME_RE = re.compile(re.escape(_HOME) + r"(?=[/\s'\"\\:,)\]}]|$)")
_ANY_HOME_RE = re.compile(r"/(?:home|Users|media)/[A-Za-z0-9._-]+")
_USER_RE = re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(_USER) + r"(?![A-Za-z0-9_-])") if len(_USER) >= 3 else None


def scrub_text(s: str) -> str:
    if not s:
        return s
    s = _HOME_RE.sub("~", s)
    s = _ANY_HOME_RE.sub(lambda m: "~" if m.group(0).startswith(("/home/", "/Users/")) else "/media/…", s)
    if _USER_RE is not None:
        s = _USER_RE.sub("user", s)
    return s


def scrub(obj):
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    return obj
