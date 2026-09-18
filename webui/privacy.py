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

# A path is recognised only where one can begin. Base64 (a field file's frames
# are megabytes of it) uses A-Z a-z 0-9 + / =, so "/home/<name>" occurs inside
# it by chance; replacing that changed the numbers a run had computed. Requiring
# a boundary in front leaves encoded data alone and still catches every path in
# prose, JSON, logs and tracebacks.
_BOUNDARY = r"(?<![A-Za-z0-9+/])"   # "=" stays a boundary: --prefix=/home/... is a path
_HOME_RE = re.compile(_BOUNDARY + re.escape(_HOME) + r"(?=[/\s'\"\\:,)\]}]|$)")
_ANY_HOME_RE = re.compile(_BOUNDARY + r"/(?:home|Users|media)/[A-Za-z0-9._-]+")
_USER_RE = re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(_USER) + r"(?![A-Za-z0-9_-])") if len(_USER) >= 3 else None


# A long unbroken run of base64 is a field file's frames, not prose. The paths
# above are guarded by their boundary, but a bare name needs the same care: in
# "...+alexander/..." the characters around it are exactly the boundary this
# pattern wants, so encoded data was being rewritten and decoded to a different
# field than the solver produced.
_ENCODED = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def _scrub_prose(t: str) -> str:
    """Every substitution, on a span that is not encoded data."""
    t = _HOME_RE.sub("~", t)
    t = _ANY_HOME_RE.sub(lambda m: "~" if m.group(0).startswith(("/home/", "/Users/")) else "/media/…", t)
    if _USER_RE is not None:
        t = _USER_RE.sub("user", t)
    return t


def scrub_text(s: str) -> str:
    """Scrub the prose and leave encoded data exactly as the run wrote it.

    The split comes first and covers all three substitutions, rather than
    guarding whichever one was fixed last. The boundary in the patterns above
    is not enough on its own: "=" has to stay a boundary so that
    --prefix=/home/... is caught, and "=" is also base64's padding, so
    an equals sign followed by a home path inside concatenated frames matched
    and was rewritten."""
    if not s:
        return s
    out, last = [], 0
    for m in _ENCODED.finditer(s):
        out.append(_scrub_prose(s[last:m.start()]))
        out.append(m.group(0))              # untouched
        last = m.end()
    out.append(_scrub_prose(s[last:]))
    return "".join(out)


def scrub(obj):
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, dict):
        # keys as well: a run's own JSON can be keyed by a path it wrote
        return {(scrub_text(k) if isinstance(k, str) else k): scrub(v)
                for k, v in obj.items()}
    return obj
