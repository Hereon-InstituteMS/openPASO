# `verify_mesh_independence`

**Refines the mesh step by step and reports whether the answer stopped changing.**

Group: Check the answer.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `input_template` | string | yes |  |
| `resolution` | number | yes |  |
| `refinement_factor` | number | no | `2.0` |
| `levels` | integer | no | `1` |
| `parameter_kind` | string | no | `'divisions'` |
| `field` | string | no | `''` |
| `probe_points` | string | no | `''` |
| `rel_tol` | number | no | `0.01` |
| `job_name` | string | no | `''` |
| `np` | integer | no | `1` |
| `critic_approved` | boolean | no | `False` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Heuristic mesh-independence study for problems WITHOUT an exact
    solution: re-run the SAME problem at successively refined
    resolutions and accept it as converged only if ALL monitored
    quantities stop changing materially.
    
    MMS convergence tests need a manufactured exact solution; real
    application problems have none. This tool automates the
    established recourse: halve the discretisation length (once by
    default, more via `levels`), then compare (a) a volume-weighted
    global L2 norm and the global max of the primary field, (b) the
    field value at probe points (auto-chosen from the mesh — field
    hotspot, domain centre, off-centre interior points — or supplied
    explicitly), and (c) any scalar QoIs the script writes to
    results_summary.json. Verdict: CONVERGED only if every monitored
    quantity changes by less than `rel_tol` on the finest refinement
    step; otherwise NOT CONVERGED, with all numbers in the report.
    
    The input template must contain the placeholder __RESOLUTION__
    where the characteristic discretisation parameter goes, e.g.
    `nx = __RESOLUTION__`. For Python-scripted solvers (fenics,
    ngsolve, skfem, dune) the template is the solve script itself; for
    compiled/file-input solvers (fourc, dealii, kratos, febio) it is a
    generator script that writes the input file, exactly as in
    run_with_generator. The solve must write the primary field as
    nodal data in a VTU/VTK/VTP result file.
    
    IMPORTANT — this tool checks discretisation convergence only. It
    does not validate the model physics; have the MANDATORY critic
    review the setup and pass critic_approved=True as with the run
    tools.
    
    Args:
        solver: Backend name (any registered backend).
        input_template: Solve/generator script containing __RESOLUTION__.
        resolution: Coarsest value of the discretisation parameter.
        refinement_factor: Refinement per level (default 2 = halving h).
        levels: Number of refinements (default 1; runs levels+1 cases).
        parameter_kind: 'divisions' (parameter counts elements; refining
            multiplies) or 'size' (parameter is h; refining divides).
        field: Field name to monitor (default: auto-select from result).
        probe_points: Optional JSON list of probe coordinates, e.g.
            "[[0.5, 0.5], [0.25, 0.75]]" (default: auto from the mesh).
        rel_tol: Acceptance threshold on relative change (default 0.01).
        job_name: Optional study directory name.
        np: MPI processes per run.
        critic_approved: True only after the critic approved the setup.
    ```
