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


def _csv(p: Path) -> dict:
    try:
        with p.open(newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:
        return {"kind": "error", "error": f"csv read failed: {e}"}
    if not rows:
        return {"kind": "table", "header": [], "rows": []}
    header = rows[0]
    body = rows[1:1000]
    cols = list(zip(*body)) if body else []
    plot = None
    if len(header) >= 2:
        try:
            x = [float(v) for v in cols[0]]
            traces = []
            for i, name in enumerate(header[1:], start=1):
                ys = []
                for v in cols[i]:
                    try:
                        ys.append(float(v))
                    except ValueError:
                        ys.append(None)
                traces.append({"x": x, "y": ys, "name": name,
                               "mode": "lines+markers", "type": "scatter"})
            plot = {
                "data": traces,
                "layout": {
                    "title": p.name,
                    "xaxis": {"title": header[0]},
                    "yaxis": {"title": "value"},
                    "margin": {"t": 40, "l": 60, "r": 20, "b": 50},
                },
                # Plotly config — the frontend forwards these flags so
                # axes, titles and legend entries become click-editable
                # in place, and the toolbar exposes PNG/SVG download.
                "config": {
                    "editable": True,
                    "edits": {"titleText": True, "axisTitleText": True,
                              "legendText": True,
                              "annotationText": True,
                              "shapePosition": True},
                    "responsive": True,
                    "displayModeBar": True,
                    "toImageButtonOptions": {
                        "format": "png", "filename": p.stem,
                        "scale": 2,
                    },
                },
            }
        except Exception:
            plot = None
    return {"kind": "table", "header": header, "rows": body,
            "truncated": len(rows) > 1001, "plot": plot,
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


def extract_params(script_text: str) -> list[dict]:
    """Heuristic: pull numeric scalar assignments from a script so the
    UI can offer sliders to re-run with modified values.

    Returns a list of ``{name, value, min, max}`` descriptors. We pick
    up only top-level assignments where the right-hand side is a single
    integer or float literal — definitely-safe to round-trip.
    """
    import re
    out, seen = [], set()
    pat = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:#.*)?$")
    for line in script_text.splitlines():
        m = pat.match(line.rstrip())
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        if name in seen:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        seen.add(name)
        lo = v * 0.5 if v != 0 else -1.0
        hi = v * 1.5 if v != 0 else 1.0
        if v < 0:
            lo, hi = hi, lo
        out.append({"name": name, "value": v,
                    "min": lo, "max": hi,
                    "step": (hi - lo) / 100 if hi != lo else 1.0})
        if len(out) >= 12:
            break
    return out
