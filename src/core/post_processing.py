"""
Unified post-processing for all FEM solver outputs.

Handles VTU (deal.ii, 4C, converted FEniCS) and XDMF transparently.
Extracts field statistics, generates QA-checked PNG visualizations.
"""

import json
import logging
from dataclasses import dataclass, field as datafield
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("openpaso.postprocess")


@dataclass
class FieldStats:
    """Statistics for a single field."""
    name: str
    min: float
    max: float
    mean: float
    std: float
    n_components: int  # 1=scalar, 2/3=vector
    location: str  # "point" or "cell"
    min_location: Optional[list[float]] = None  # xyz of min value
    max_location: Optional[list[float]] = None  # xyz of max value

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "n_components": self.n_components,
            "location": self.location,
        }
        if self.min_location:
            d["min_at"] = [round(x, 6) for x in self.min_location]
        if self.max_location:
            d["max_at"] = [round(x, 6) for x in self.max_location]
        return d


@dataclass
class MeshInfo:
    """Basic mesh information."""
    n_points: int
    n_cells: int
    bounds: list[float]  # [xmin, xmax, ymin, ymax, zmin, zmax]
    spatial_dim: int  # 2 or 3 (based on z-range)
    cell_types: list[str]

    def to_dict(self) -> dict:
        return {
            "n_points": self.n_points,
            "n_cells": self.n_cells,
            "bounds": [round(b, 6) for b in self.bounds],
            "spatial_dim": self.spatial_dim,
        }


@dataclass
class PostProcessResult:
    """Complete post-processing result for one output file."""
    file: str
    mesh: MeshInfo
    fields: list[FieldStats]
    qa_warnings: list[str] = datafield(default_factory=list)
    plots: list[str] = datafield(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "mesh": self.mesh.to_dict(),
            "fields": [f.to_dict() for f in self.fields],
            "qa_warnings": self.qa_warnings,
            "plots": self.plots,
        }


def _detect_spatial_dim(bounds: list[float]) -> int:
    """Detect 2D vs 3D from mesh bounds."""
    xy_range = max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1e-30)
    z_range = bounds[5] - bounds[4]
    return 2 if z_range < 1e-10 * xy_range else 3


def read_mesh(file_path: Path):
    """Read a VTU, VTK, or XDMF file into a PyVista mesh."""
    import pyvista as pv
    pv.OFF_SCREEN = True

    suffix = file_path.suffix.lower()
    if suffix in (".vtu", ".vtk", ".pvd", ".pvtu"):
        return pv.read(str(file_path))
    elif suffix == ".xdmf":
        # Try direct read first, fall back to conversion
        try:
            reader = pv.get_reader(str(file_path))
            return reader.read()
        except Exception:
            # Try meshio fallback
            import meshio
            m = meshio.read(str(file_path))
            return pv.wrap(m)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def extract_field_stats(mesh, points: np.ndarray = None) -> list[FieldStats]:
    """Extract statistics for all fields in a PyVista mesh."""
    stats = []

    for name in mesh.point_data:
        arr = np.asarray(mesh.point_data[name])
        if arr.ndim == 1:
            n_comp = 1
            magnitude = arr
        else:
            n_comp = arr.shape[1]
            magnitude = np.linalg.norm(arr, axis=1) if n_comp > 1 else arr.flatten()

        fs = FieldStats(
            name=name,
            min=float(magnitude.min()),
            max=float(magnitude.max()),
            mean=float(magnitude.mean()),
            std=float(magnitude.std()),
            n_components=n_comp,
            location="point",
        )

        # Find locations of min/max
        if points is not None or hasattr(mesh, 'points'):
            pts = points if points is not None else np.asarray(mesh.points)
            fs.min_location = pts[np.argmin(magnitude)].tolist()
            fs.max_location = pts[np.argmax(magnitude)].tolist()

        stats.append(fs)

    for name in mesh.cell_data:
        arr = np.asarray(mesh.cell_data[name])
        if arr.ndim == 1:
            n_comp = 1
            magnitude = arr
        else:
            n_comp = arr.shape[1]
            magnitude = np.linalg.norm(arr, axis=1) if n_comp > 1 else arr.flatten()

        stats.append(FieldStats(
            name=name,
            min=float(magnitude.min()),
            max=float(magnitude.max()),
            mean=float(magnitude.mean()),
            std=float(magnitude.std()),
            n_components=n_comp,
            location="cell",
        ))

    return stats


def qa_check(mesh, fields: list[FieldStats]) -> list[str]:
    """Run quality assurance checks on mesh and field data."""
    warnings = []

    # Check for NaN/Inf in fields
    for name in mesh.point_data:
        arr = np.asarray(mesh.point_data[name])
        if np.any(np.isnan(arr)):
            warnings.append(f"NaN detected in field '{name}'")
        if np.any(np.isinf(arr)):
            warnings.append(f"Inf detected in field '{name}'")

    for name in mesh.cell_data:
        arr = np.asarray(mesh.cell_data[name])
        if np.any(np.isnan(arr)):
            warnings.append(f"NaN detected in cell field '{name}'")
        if np.any(np.isinf(arr)):
            warnings.append(f"Inf detected in cell field '{name}'")

    # Check mesh quality
    if mesh.n_points == 0:
        warnings.append("Mesh has zero points")
    if mesh.n_cells == 0:
        warnings.append("Mesh has zero cells")

    # Check for degenerate bounds
    bounds = mesh.bounds
    for i in range(3):
        rng = bounds[2*i+1] - bounds[2*i]
        if rng < 0:
            warnings.append(f"Negative range in dimension {i}")

    return warnings


def _sbar_right(title: str, fmt: str = "%.3f", n_labels: int = 5) -> dict:
    """Vertical color bar in right margin — for 2D plots with shifted camera."""
    return {
        "title": title,
        "title_font_size": 16,
        "label_font_size": 14,
        "shadow": False,
        "n_labels": n_labels,
        "fmt": fmt,
        "position_x": 0.86,
        "position_y": 0.15,
        "width": 0.04,
        "height": 0.55,
        "vertical": True,
    }


def _sbar_bottom(title: str, fmt: str = "%.3f", n_labels: int = 5) -> dict:
    """Horizontal color bar at bottom — for 3D views and subplots."""
    return {
        "title": title,
        "title_font_size": 16,
        "label_font_size": 14,
        "shadow": False,
        "n_labels": n_labels,
        "fmt": fmt,
        "position_x": 0.15,
        "position_y": 0.05,
        "width": 0.7,
        "height": 0.05,
        "vertical": False,
    }


def _auto_fmt(arr) -> str:
    """Choose a number format based on value range."""
    val_range = float(np.abs(arr).max())
    if val_range == 0:
        return "%.1f"
    elif val_range > 100:
        return "%.1f"
    elif val_range > 1:
        return "%.3f"
    elif val_range > 0.001:
        return "%.5f"
    else:
        return "%.2e"


def _setup_2d_camera(plotter, mesh):
    """Set up 2D camera — domain in left ~75%, right margin for color bar.

    Handles meshes at any z-offset (e.g. z=999 in 4C fluid meshes).
    """
    plotter.view_xy()
    cx = (mesh.bounds[0] + mesh.bounds[1]) / 2
    cy = (mesh.bounds[2] + mesh.bounds[3]) / 2
    zval = mesh.bounds[4]
    dx = mesh.bounds[1] - mesh.bounds[0]
    dy = mesh.bounds[3] - mesh.bounds[2]

    cx_shifted = cx + dx * 0.12
    plotter.camera.position = (cx_shifted, cy, zval + 1.0)
    plotter.camera.focal_point = (cx_shifted, cy, zval)
    plotter.camera.up = (0, 1, 0)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = max(dx, dy) * 0.55


def _init_pyvista():
    """Initialize PyVista with clean, publication-ready settings."""
    import pyvista as pv
    import os
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    pv.OFF_SCREEN = True
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"
    return pv


def plot_field(mesh, field_name: str, output_path: Path,
               title: str = "", spatial_dim: int = None,
               show_edges: bool = False, cmap: str = "viridis",
               window_size: tuple = (1400, 1000)) -> Path:
    """Generate a publication-quality PNG plot of a field.

    PLOT RULES (non-negotiable):
    - Color bars NEVER overlap the mesh domain
    - 2D: domain in left ~75%, vertical cbar in right margin
    - 3D: isometric camera, horizontal cbar below mesh
    - White background, black text
    - Mesh wireframe as faint overlay (not thick edges on top of data)
    - Screenshot at scale=2 for publication resolution

    Args:
        mesh: PyVista mesh object
        field_name: Name of the field to plot
        output_path: Path to save PNG
        title: Plot title
        spatial_dim: Override 2D/3D detection
        show_edges: Show solid mesh edges (default: False for clean look)
        cmap: Colormap (viridis for scalars, turbo for displacement, jet for stress)
        window_size: Image size in pixels (before scale=2)

    Returns:
        Path to saved PNG.
    """
    pv = _init_pyvista()

    if spatial_dim is None:
        spatial_dim = _detect_spatial_dim(list(mesh.bounds))

    # Resolve field and handle vectors
    if field_name in mesh.point_data:
        arr = np.asarray(mesh.point_data[field_name])
    elif field_name in mesh.cell_data:
        arr = np.asarray(mesh.cell_data[field_name])
    else:
        raise ValueError(f"Field '{field_name}' not found")

    is_vector = arr.ndim > 1 and arr.shape[1] > 1
    if is_vector:
        magnitude = np.linalg.norm(arr, axis=1)
        mesh[f"|{field_name}|"] = magnitude
        scalars = f"|{field_name}|"
        bar_title = f"|{field_name}|"
    else:
        scalars = field_name
        bar_title = field_name

    fmt = _auto_fmt(arr if not is_vector else magnitude)

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))

    # Main mesh with data (no edges — clean)
    plotter.add_mesh(
        mesh, scalars=scalars, cmap=cmap,
        show_edges=False,
        scalar_bar_args=_sbar_right(bar_title, fmt=fmt) if spatial_dim == 2
                        else _sbar_bottom(bar_title, fmt=fmt),
    )

    # Faint wireframe overlay for mesh structure
    if show_edges or mesh.n_cells < 3000:
        plotter.add_mesh(mesh, style="wireframe", color="gray",
                         opacity=0.08, line_width=0.5)

    # Camera
    if spatial_dim == 2:
        _setup_2d_camera(plotter, mesh)
    else:
        plotter.view_isometric()

    # Title — top left, not overlapping
    if title:
        plotter.add_title(title, font_size=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path), transparent_background=False, scale=2)
    plotter.close()

    # QA
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        if size_kb < 5:
            logger.warning(f"Plot suspiciously small: {size_kb:.1f} KB")
    else:
        logger.warning(f"Plot not created: {output_path}")

    return output_path


def plot_comparison(result_files: list[Path], field_name: str,
                    output_path: Path, solver_names: list[str] = None,
                    cmap: str = "viridis") -> Path:
    """Side-by-side comparison of the same field from multiple solvers.

    Key paper figure — shows cross-solver agreement visually with shared color scale.
    """
    pv = _init_pyvista()

    n = len(result_files)
    if n == 0:
        raise ValueError("No result files to compare")

    plotter = pv.Plotter(off_screen=True, shape=(1, n),
                         window_size=[550 * n, 550])

    global_min, global_max = float('inf'), float('-inf')
    meshes = []

    # First pass: read all and find global range
    for fp in result_files:
        m = read_mesh(fp)
        meshes.append(m)
        if field_name in m.point_data:
            arr = np.asarray(m.point_data[field_name])
        elif field_name in m.cell_data:
            arr = np.asarray(m.cell_data[field_name])
        else:
            continue
        if arr.ndim > 1:
            arr = np.linalg.norm(arr, axis=1)
        global_min = min(global_min, float(arr.min()))
        global_max = max(global_max, float(arr.max()))

    fmt = _auto_fmt(np.array([global_min, global_max]))

    # Second pass: plot with shared color scale
    for i, (m, fp) in enumerate(zip(meshes, result_files)):
        plotter.subplot(0, i)

        plotter.add_mesh(
            m, scalars=field_name,
            show_edges=False, cmap=cmap,
            clim=[global_min, global_max],
            scalar_bar_args=_sbar_bottom(field_name, fmt=fmt) if i == n // 2 else
                            {"vertical": False, "position_x": 99, "width": 0},  # hide non-center bars
        )
        plotter.add_mesh(m, style="wireframe", color="gray", opacity=0.06, line_width=0.5)

        label = solver_names[i] if solver_names and i < len(solver_names) else fp.parent.name
        plotter.add_text(label, position="upper_left", font_size=14, color="black")

        dim = _detect_spatial_dim(list(m.bounds))
        if dim == 2:
            plotter.view_xy()
            plotter.camera.tight(padding=0.1)
        else:
            plotter.view_isometric()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path), transparent_background=False, scale=2)
    plotter.close()
    return output_path


def post_process_file(file_path: Path, plot_dir: Path = None,
                      plot_fields: bool = True) -> PostProcessResult:
    """Full post-processing pipeline for a single result file.

    Args:
        file_path: Path to VTU/XDMF file
        plot_dir: Directory for PNG outputs (default: same as file)
        plot_fields: Whether to generate PNG plots

    Returns:
        PostProcessResult with mesh info, field stats, QA warnings, and plot paths.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    mesh = read_mesh(file_path)
    if plot_dir is None:
        plot_dir = file_path.parent

    bounds = list(mesh.bounds)
    spatial_dim = _detect_spatial_dim(bounds)

    mesh_info = MeshInfo(
        n_points=mesh.n_points,
        n_cells=mesh.n_cells,
        bounds=bounds,
        spatial_dim=spatial_dim,
        cell_types=[str(ct) for ct in set(mesh.celltypes)] if hasattr(mesh, 'celltypes') else [],
    )

    fields = extract_field_stats(mesh)
    warnings = qa_check(mesh, fields)

    plots = []
    if plot_fields and len(fields) > 0:
        for fs in fields:
            try:
                plot_path = plot_dir / f"plot_{fs.name}.png"
                title = f"{fs.name} (min={fs.min:.4e}, max={fs.max:.4e})"
                plot_field(mesh, fs.name, plot_path, title=title, spatial_dim=spatial_dim)
                plots.append(str(plot_path))
            except Exception as e:
                warnings.append(f"Plot failed for '{fs.name}': {e}")

    return PostProcessResult(
        file=file_path.name,
        mesh=mesh_info,
        fields=fields,
        qa_warnings=warnings,
        plots=plots,
    )


def compare_results(result_files: list[Path], field_name: str = None) -> dict:
    """Compare field statistics across multiple result files (cross-solver).

    Args:
        result_files: List of VTU/XDMF paths from different solvers
        field_name: Specific field to compare (auto-detect if None)

    Returns:
        Comparison dict with per-file stats and agreement metrics.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    comparison = {"files": [], "agreement": {}}

    for fp in result_files:
        try:
            mesh = read_mesh(fp)
            fields = extract_field_stats(mesh)

            if field_name:
                target = [f for f in fields if f.name == field_name]
            else:
                target = fields[:1]  # first field

            if target:
                fs = target[0]
                comparison["files"].append({
                    "file": fp.name,
                    "parent": fp.parent.name,
                    "field": fs.name,
                    "min": fs.min,
                    "max": fs.max,
                    "mean": fs.mean,
                    "n_points": mesh.n_points,
                })
            else:
                comparison["files"].append({
                    "file": fp.name,
                    "error": f"Field '{field_name}' not found",
                })
        except Exception as e:
            comparison["files"].append({
                "file": fp.name,
                "error": str(e),
            })

    # Compute agreement between solvers
    valid = [f for f in comparison["files"] if "max" in f]
    if len(valid) >= 2:
        maxes = [f["max"] for f in valid]
        mean_max = np.mean(maxes)
        max_deviation = max(abs(m - mean_max) for m in maxes)
        rel_deviation = max_deviation / abs(mean_max) if mean_max != 0 else float('inf')
        comparison["agreement"] = {
            "n_solvers": len(valid),
            "max_values": maxes,
            "mean_max": float(mean_max),
            "max_absolute_deviation": float(max_deviation),
            "max_relative_deviation": float(rel_deviation),
            "agreement_pct": float(max(0, (1 - rel_deviation) * 100)),
        }

    return comparison
