# `visualize`

**Makes plots and statistics from a result, with automatic sanity checks.**

Group: Run.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `job_id` | string | no | `''` |
| `work_dir` | string | no | `''` |
| `action` | string | no | `'summary'` |
| `field` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Post-process and visualize simulation results.
    
    Args:
        job_id: Job ID from run_simulation (or leave empty and set work_dir)
        work_dir: Direct path to results directory
        action: What to do. Options:
            - "summary" — field statistics + results_summary.json content
              (default; the fastest pulse-check on a finished run)
            - "list" — list every result file under the work dir
            - "plot" — generate a PNG of the named field (needs field=)
            - "validate" — automated sanity checks across the first
              3 result files: NaN/Inf detection, constant-field
              detection, suspiciously-large-magnitude detection
              (>1e15). Use after summary when a field looks wrong.
        field: Specific field name to plot (e.g. 'temperature', 'displacement')
    ```
