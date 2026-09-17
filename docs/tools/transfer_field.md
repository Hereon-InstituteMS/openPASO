# `transfer_field`

**Moves a result from one solver's output into another solver's input.**

Group: Two solvers on one problem.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `source_vtu` | string | yes |  |
| `field_name` | string | yes |  |
| `interface_coord` | number | yes |  |
| `interface_axis` | integer | no | `0` |
| `target_format` | string | no | `'json'` |
| `output_path` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Extract field values at an interface from a VTU file and format for transfer.
    
    Universal data connector for cross-solver coupling. Reads VTU output
    from any solver, extracts values at the interface plane, and formats
    them for the target solver's expected input shape.
    
    FIELD (VOLUME) COUPLINGS: pass `interface_axis=-1`. Not every coupling
    exchanges data on a surface. In thermo-structural interaction both
    participants own the WHOLE body and exchange volume fields — the
    temperature one way and the volumetric strain the other — so there is no
    interface plane, no normal and no flux to balance, and a plane slice
    cannot express the exchange at all. With `interface_axis=-1` every point
    in the file is taken, `interface_coord` is ignored, and the points come
    back in a fixed lexicographic order (the `couple` driver relaxes export
    vectors entry by entry, so the order must not move between iterations).
    
    Args:
        source_vtu: Path to VTU result file from the source solver.
        field_name: Field to extract (e.g. 'temperature', 'displacement').
        interface_coord: Coordinate value defining the interface plane.
            Ignored when interface_axis is -1.
        interface_axis: Axis perpendicular to interface (0=x, 1=y, 2=z), or
            -1 for the WHOLE VOLUME (field coupling, see above).
        target_format: Output format. Options:
            - "json"        — interface coordinates + values (default)
            - "fenics"      — Python BoundaryCondition snippet (Dirichlet
                              at this interface), saved as .py
            - "4c_neumann"  — 4C-format YAML snippet for a Neumann
                              boundary condition, saved as .yaml
        output_path: Where to save the formatted output. If empty,
            auto-generated next to the source VTU as
            'interface_<field_name>.<ext>'.
    
    Returns:
        A summary string with the interface min/max/mean and the path
        of the saved file.
    ```
