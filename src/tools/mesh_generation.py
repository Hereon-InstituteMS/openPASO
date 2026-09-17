"""
MCP tool for mesh generation using Gmsh.

Generates meshes on non-trivial geometries for use across ALL backends:
- L-shaped domain (corner singularity benchmark)
- Plate with circular hole (stress concentration)
- Channel with cylinder obstacle (CFD benchmark, Schäfer-Turek)
- Custom rectangular domains with refinement

Output formats: .msh (Gmsh), .xdmf (FEniCS), .e (Exodus for 4C)
Cross-solver mesh transfer: one mesh → multiple solvers.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("openpaso.mesh")

from core.output_paths import output_dir as _output_dir  # noqa: E402
_MESH_OUTPUT_DIR = _output_dir("meshes")


def _generate_l_domain_2d(mesh_size: float = 0.05, output_path: Path = None) -> Path:
    """Generate L-shaped domain mesh. Classic FEM benchmark with corner singularity."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("L-domain")

    # L-domain: [-1,1]^2 minus [0,1]x[-1,0] — matches deal.II hyper_L(-1,1)
    # Points (counterclockwise)
    p1 = gmsh.model.geo.addPoint(-1, -1, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(0, -1, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size * 0.3)  # Fine at re-entrant corner
    p4 = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
    p5 = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)
    p6 = gmsh.model.geo.addPoint(-1, 1, 0, mesh_size)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Physical groups for BCs
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l1], tag=1, name="bottom")
    gmsh.model.addPhysicalGroup(1, [l2], tag=2, name="inner_vertical")
    gmsh.model.addPhysicalGroup(1, [l3], tag=3, name="inner_horizontal")
    gmsh.model.addPhysicalGroup(1, [l4], tag=4, name="right")
    gmsh.model.addPhysicalGroup(1, [l5], tag=5, name="top")
    gmsh.model.addPhysicalGroup(1, [l6], tag=6, name="left")
    gmsh.model.addPhysicalGroup(2, [s], tag=100, name="domain")

    gmsh.model.mesh.generate(2)

    if output_path is None:
        output_path = _MESH_OUTPUT_DIR / "l_domain.msh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elements = len(gmsh.model.mesh.getElements(2)[1][0]) if gmsh.model.mesh.getElements(2)[1] else 0
    gmsh.finalize()

    return output_path, n_nodes, n_elements


def _generate_plate_with_hole_2d(mesh_size: float = 0.05, radius: float = 0.2,
                                   width: float = 2.0, height: float = 1.0,
                                   output_path: Path = None) -> Path:
    """Plate with circular hole — stress concentration benchmark."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate-with-hole")

    # Rectangle
    rect = gmsh.model.occ.addRectangle(-width/2, -height/2, 0, width, height)
    # Circular hole at center
    hole = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    # Boolean difference
    gmsh.model.occ.cut([(2, rect)], [(2, hole)])
    gmsh.model.occ.synchronize()

    # Mesh refinement near hole
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [c[1] for c in gmsh.model.getEntities(1)])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size * 0.3)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", radius * 0.5)
    gmsh.model.mesh.field.setNumber(2, "DistMax", radius * 3)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Physical groups
    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=100, name="domain")

    curves = gmsh.model.getEntities(1)
    for i, c in enumerate(curves):
        gmsh.model.addPhysicalGroup(1, [c[1]], tag=i+1)

    gmsh.model.mesh.generate(2)

    if output_path is None:
        output_path = _MESH_OUTPUT_DIR / "plate_with_hole.msh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elements = len(gmsh.model.mesh.getElements(2)[1][0]) if gmsh.model.mesh.getElements(2)[1] else 0
    gmsh.finalize()

    return output_path, n_nodes, n_elements


def _generate_channel_with_cylinder_2d(mesh_size: float = 0.05, cyl_radius: float = 0.05,
                                         cyl_center: tuple = (0.2, 0.2),
                                         channel_length: float = 2.2, channel_height: float = 0.41,
                                         output_path: Path = None) -> Path:
    """Channel with cylinder obstacle — DFG/Schäfer-Turek CFD benchmark geometry."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("channel-cylinder")

    # Channel
    rect = gmsh.model.occ.addRectangle(0, 0, 0, channel_length, channel_height)
    # Cylinder
    cyl = gmsh.model.occ.addDisk(cyl_center[0], cyl_center[1], 0, cyl_radius, cyl_radius)
    # Cut
    gmsh.model.occ.cut([(2, rect)], [(2, cyl)])
    gmsh.model.occ.synchronize()

    # Refine near cylinder
    gmsh.model.mesh.field.add("Distance", 1)
    curves = gmsh.model.getEntities(1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [c[1] for c in curves])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size * 0.2)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", cyl_radius)
    gmsh.model.mesh.field.setNumber(2, "DistMax", cyl_radius * 10)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=100, name="fluid")
    for i, c in enumerate(curves):
        gmsh.model.addPhysicalGroup(1, [c[1]], tag=i+1)

    gmsh.model.mesh.generate(2)

    if output_path is None:
        output_path = _MESH_OUTPUT_DIR / "channel_cylinder.msh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elements = len(gmsh.model.mesh.getElements(2)[1][0]) if gmsh.model.mesh.getElements(2)[1] else 0
    gmsh.finalize()

    return output_path, n_nodes, n_elements


def _convert_msh_to_xdmf(msh_path: Path) -> Path:
    """Convert Gmsh .msh to XDMF for FEniCS."""
    import meshio
    mesh = meshio.read(str(msh_path))
    # Extract triangles only (2D)
    cells = [c for c in mesh.cells if c.type == "triangle"]
    if not cells:
        cells = [c for c in mesh.cells if c.type in ("quad", "tetra", "hexahedron")]
    if cells:
        # Keep only the cells and points
        out_mesh = meshio.Mesh(
            points=mesh.points[:, :3] if mesh.points.shape[1] > 3 else mesh.points,
            cells=cells,
        )
        xdmf_path = msh_path.with_suffix(".xdmf")
        meshio.write(str(xdmf_path), out_mesh)
        return xdmf_path
    return msh_path


def register_mesh_tools(mcp: FastMCP):

    @mcp.tool()
    def generate_mesh(geometry: str, mesh_size: float = 0.05,
                      params: str = "{}") -> str:
        """Generate a mesh on a non-trivial geometry using Gmsh.

        Available geometries:
        - 'l_domain': L-shaped domain (corner singularity benchmark)
        - 'plate_with_hole': Rectangle with circular hole (stress concentration)
        - 'channel_cylinder': Channel with cylinder obstacle (DFG CFD benchmark)
        - 'rectangle': Simple rectangle with custom dimensions

        The mesh is saved in Gmsh (.msh) format and also converted to
        XDMF for FEniCS compatibility. Use the returned path in subsequent
        simulation calls.

        Args:
            geometry: Geometry type (see list above)
            mesh_size: Target element size (smaller = finer mesh)
            params: JSON parameters for geometry customization
                    e.g. '{"radius": 0.3, "width": 4.0}' for plate_with_hole
        """
        try:
            import gmsh
        except ImportError:
            return "ERROR: Gmsh not installed. Run: pip install gmsh"

        try:
            param_dict = json.loads(params)
        except json.JSONDecodeError:
            return f"Invalid params JSON: {params}"

        generators = {
            "l_domain": _generate_l_domain_2d,
            "plate_with_hole": _generate_plate_with_hole_2d,
            "channel_cylinder": _generate_channel_with_cylinder_2d,
        }

        gen = generators.get(geometry.lower())
        if not gen:
            return f"Unknown geometry: {geometry}. Available: {list(generators.keys())}"

        try:
            msh_path, n_nodes, n_elements = gen(mesh_size=mesh_size, **param_dict)

            # Convert to XDMF for FEniCS
            try:
                xdmf_path = _convert_msh_to_xdmf(msh_path)
                xdmf_msg = f"XDMF: {xdmf_path}"
            except Exception as e:
                xdmf_msg = f"XDMF conversion failed: {e}"

            return json.dumps({
                "geometry": geometry,
                "mesh_file": str(msh_path),
                "format": "Gmsh (.msh)",
                "n_nodes": n_nodes,
                "n_elements": n_elements,
                "mesh_size": mesh_size,
                "xdmf": xdmf_msg,
                "usage": {
                    "fenics": f"mesh = meshio.read('{msh_path}') or gmsh.read('{msh_path}')",
                    "dealii": "Use GridIn to read .msh, or use built-in GridGenerator::hyper_L()",
                    "fourc": "Convert to Exodus (.e) via meshio for 4C",
                },
            }, indent=2)

        except Exception as e:
            return f"Mesh generation failed: {e}"
