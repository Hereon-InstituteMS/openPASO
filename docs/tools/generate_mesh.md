# `generate_mesh`

**Builds a mesh with Gmsh: a rectangle, an L-shape, a plate with a hole, a channel, or your own shape.**

Group: Run.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `geometry` | string | yes |  |
| `mesh_size` | number | no | `0.1` |
| `output_dir` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Generate a mesh using Gmsh for non-trivial geometries.
    
    Args:
        geometry: One of the built-in geometries:
            - "l_domain"          — 2D L-shaped domain
            - "plate_with_hole"   — 2D plate with circular hole
            - "channel_cylinder"  — 2D channel with cylindrical obstacle
            (No "custom" passthrough yet — passing any other name
            returns a 'Unknown geometry' message with this list.)
        mesh_size: Target element size
        output_dir: Where to save (auto if empty)
    ```
