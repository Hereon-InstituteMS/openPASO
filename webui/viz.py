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


_JSON_FULL_MAX = 4 * 1024 * 1024     # above this, only the head is read


def _json(p: Path) -> dict:
    """A JSON file, or — for a field series, which holds one frame of numbers
    per timestep — its description only. A run's field file is routinely tens of
    megabytes, and the interface asks about every JSON a run writes, so parsing
    all of it here loaded the whole thing into the server to read a dozen
    fields."""
    try:
        if p.stat().st_size > _JSON_FULL_MAX:
            head = _head_fields(p)
            if head is not None:
                return _field_series(p, head)
            return {"kind": "text", "text": _read_text(p),
                    "syntax": "json", "truncated": True,
                    "rel": str(p.relative_to(config.SANDBOX_ROOT))}
        obj = json.loads(p.read_text())
    except Exception as e:
        return {"kind": "error", "error": f"json: {e}"}

    # A field series is a solver's own output sampled onto a grid, one frame
    # per stored timestep. Hand back a description and let the browser fetch the
    # file once; re-serialising several megabytes through this endpoint would
    # buy nothing.
    if isinstance(obj, dict) and obj.get("kind") == "field_series":
        return _field_series(p, obj)

    return {"kind": "json", "obj": obj, "path": str(p)}


def _url_for(p: Path) -> str:
    """The address the browser fetches a run's file from. A name may hold ? or
    #, which the browser would read as a query or a fragment."""
    from urllib.parse import quote
    rel = p.relative_to(config.SANDBOX_ROOT)
    return "/sandbox-file/" + "/".join(quote(part, safe="") for part in rel.parts)


def _field_series(p: Path, meta: dict) -> dict:
    """What the interface needs to draw a field: where the file is, the grid it
    sits on, and the range behind the picture."""
    times = meta.get("times")
    return {
        "kind": "field_series",
        "url": _url_for(p),
        "name": p.name,
        "field": meta.get("field", "field"),
        "unit": meta.get("unit", ""),
        "nx": meta.get("nx"), "ny": meta.get("ny"),
        "vmin": meta.get("vmin"), "vmax": meta.get("vmax"),
        "n_frames": len(times) if isinstance(times, list) else meta.get("n_frames"),
        "x0": meta.get("x0"), "y0": meta.get("y0"),
        "dx": meta.get("dx"), "dy": meta.get("dy"),
        # What the picture does not show on its own: the true range behind
        # the clip, how much is saturated, and where it came from.
        "provenance": meta.get("provenance") or {},
    }


_HEAD_BYTES = 1024 * 1024


def _head_fields(p: Path) -> dict | None:
    """The description at the start of a large field series, read without
    parsing the frames behind it. None when the file is not one.

    These files are written description first, then the frames, so the fields
    the interface needs are in the first kilobytes of a file that may be tens
    of megabytes."""
    import re
    with p.open("rb") as f:
        head = f.read(_HEAD_BYTES).decode("utf-8", "replace")
    if '"field_series"' not in head[:200]:
        return None
    out: dict = {}
    for key, value in re.findall(r'"([a-z_]+)"\s*:\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?|"[^"]*")', head):
        if key in ("field", "unit"):
            out[key] = value.strip('"')
        elif key in ("nx", "ny", "vmin", "vmax", "x0", "y0", "dx", "dy", "fps"):
            try:
                out[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                pass
    times = re.search(r'"times"\s*:\s*\[([^\]]*)\]', head)
    if times:
        out["n_frames"] = len([x for x in times.group(1).split(",") if x.strip()])
    prov = re.search(r'"provenance"\s*:\s*(\{[^{}]*\})', head)
    if prov:
        try:
            out["provenance"] = json.loads(prov.group(1))
        except ValueError:
            pass
    return out


def _vtk(p: Path) -> dict:
    """Return a descriptor for vtk.js to load from /sandbox/<rel>.

    We keep the actual parsing to the browser to avoid pulling vtk
    server-side; we just expose the URL and a few hints.
    """
    return {
        "kind": "vtk",
        "url": _url_for(p),
        "format": p.suffix.lower().lstrip("."),
        "name": p.name,
    }


def _hdf(p: Path) -> dict:
    """An HDF5 file's top-level contents, or — for the XDMF file that describes
    one — the XML itself and the data files it points at. h5py cannot read XDMF,
    so routing both here labelled an XDMF file as HDF5 with no contents and
    never showed the .h5 beside it that holds the numbers."""
    if p.suffix.lower() == ".xdmf":
        import re
        text = _read_text(p)
        seen, refs = set(), []
        for hit in re.findall(r"<DataItem[^>]*>\s*([^<\s]+)", text):
            name = hit.split(":")[0].strip()
            if name and name not in seen:
                seen.add(name)
                refs.append(name)
        return {"kind": "xdmf", "name": p.name, "text": text, "syntax": "xml",
                "data_files": refs,
                "rel": str(p.relative_to(config.SANDBOX_ROOT))}
    try:
        import h5py  # optional
        with h5py.File(p, "r") as h:
            keys = list(h.keys())
    except Exception:
        keys = []
    return {"kind": "hdf", "name": p.name, "keys": keys,
            "rel": str(p.relative_to(config.SANDBOX_ROOT))}
