"""File → visualization payload.

Given a path inside the sandbox, return a JSON-friendly payload the
frontend can render directly: Plotly figure JSON for tabular/CSV data,
parsed mesh metadata for VTK files, raw text for input files, etc.

Heavyweight VTK rendering happens in the browser via vtk.js; this
module only inspects the file and tells the frontend how to wire it up.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

from . import config, files


def visualize(rel: str) -> dict:
    p = files._safe(rel)
    if not p.is_file():
        return {"kind": "error", "error": f"not a file: {p}"}
    kind = files.classify(p)
    if kind == "vtk":
        return _vtk(p)
    if kind == "hdf":
        return _hdf(p)
    if kind == "image":
        return {"kind": "image", "path": str(p),
                "rel": str(p.relative_to(config.SANDBOX_ROOT))}
    if kind == "table":
        return _csv(p)
    if kind == "json":
        return _json(p)
    if kind in ("yaml", "mesh", "text"):
        return {"kind": "text", "text": _read_text(p),
                "syntax": _syntax_for(p)}
    return {"kind": "unknown", "path": str(p)}


def _read_text(p: Path, limit: int = 200_000) -> str:
    try:
        return p.read_bytes()[:limit].decode("utf-8", errors="replace")
    except OSError as e:
        return f"[read error: {e}]"


def _syntax_for(p: Path) -> str:
    return {".py": "python", ".cc": "cpp", ".cpp": "cpp", ".c": "c",
            ".yaml": "yaml", ".yml": "yaml", ".4c": "yaml",
            ".json": "json", ".md": "markdown",
            ".sh": "bash"}.get(p.suffix.lower(), "text")


_PREVIEW_ROWS = 999


def _csv(p: Path) -> dict:
    """The first rows of a table, for the preview. A solver log can be hundreds
    of megabytes, so only what is shown is read (plus one row, to know whether
    to say the table is cut off)."""
    from itertools import islice
    delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        with p.open(newline="") as f:
            rows = list(islice(csv.reader(f, delimiter=delimiter), _PREVIEW_ROWS + 2))
    except Exception as e:
        return {"kind": "error", "error": f"csv read failed: {e}"}
    if not rows:
        return {"kind": "table", "header": [], "rows": []}
    header = rows[0]
    body = rows[1:_PREVIEW_ROWS + 1]
    # No figure is built here: the interface draws the chart itself, and the
    # Plotly figure and its editable-toolbar config this used to return were
    # read by nothing.
    return {"kind": "table", "header": header, "rows": body,
            "truncated": len(rows) > _PREVIEW_ROWS + 1,
            "rel": str(p.relative_to(config.SANDBOX_ROOT))}


def _json(p: Path) -> dict:
    try:
        obj = json.loads(p.read_text())
    except Exception as e:
        return {"kind": "error", "error": f"json: {e}"}

    # A field series is a solver's own output sampled onto a grid, one frame
    # per stored timestep. Hand back a descriptor and let the browser fetch the
    # file once; re-serialising several megabytes through this endpoint would
    # buy nothing.
    if isinstance(obj, dict) and obj.get("kind") == "field_series":
        return {
            "kind": "field_series",
            "url": f"/sandbox-file/{p.relative_to(config.SANDBOX_ROOT)}",
            "name": p.name,
            "field": obj.get("field", "field"),
            "unit": obj.get("unit", ""),
            "nx": obj.get("nx"), "ny": obj.get("ny"),
            "vmin": obj.get("vmin"), "vmax": obj.get("vmax"),
            "n_frames": len(obj.get("times") or []),
            "x0": obj.get("x0"), "y0": obj.get("y0"),
            "dx": obj.get("dx"), "dy": obj.get("dy"),
            # What the picture does not show on its own: the true range behind
            # the clip, how much is saturated, and where it came from.
            "provenance": obj.get("provenance") or {},
        }

    return {"kind": "json", "obj": obj, "path": str(p)}


def _vtk(p: Path) -> dict:
    """Return a descriptor for vtk.js to load from /sandbox/<rel>.

    We keep the actual parsing to the browser to avoid pulling vtk
    server-side; we just expose the URL and a few hints.
    """
    return {
        "kind": "vtk",
        "url": f"/sandbox-file/{p.relative_to(config.SANDBOX_ROOT)}",
        "format": p.suffix.lower().lstrip("."),
        "name": p.name,
    }


def _hdf(p: Path) -> dict:
    """Light h5/xdmf descriptor. Pair-detection: if a .xdmf exists next
    to a .h5 we show the .xdmf and link to the .h5."""
    try:
        import h5py  # optional
        with h5py.File(p, "r") as h:
            keys = list(h.keys())
    except Exception:
        keys = []
    return {"kind": "hdf", "name": p.name, "keys": keys,
            "rel": str(p.relative_to(config.SANDBOX_ROOT))}
