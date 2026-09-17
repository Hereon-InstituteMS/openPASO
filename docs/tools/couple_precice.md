# `couple_precice`

**Couples two solvers through the preCICE library instead of openPASO's own driver.**

Group: Two solvers on one problem.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `participants` | string | yes |  |
| `data` | string | yes |  |
| `exchanges` | string | yes |  |
| `work_dir` | string | yes |  |
| `scheme` | string | no | `'serial-explicit'` |
| `dimensions` | integer | no | `2` |
| `max_time` | number | no | `10.0` |
| `time_window` | number | no | `1.0` |
| `timeout` | integer | no | `1800` |
| `max_iterations` | integer | no | `20` |
| `convergence_tol` | number | no | `1e-06` |
| `relaxation` | number | no | `0.5` |
| `mapping` | string | no | `'nearest-neighbor'` |
| `extra_env` | string | no | `''` |
| `critic_approved` | boolean | no | `False` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    GENERAL preCICE coupling of ARBITRARY codes/paradigms, end-to-end.
    
    Have an independent critic review the setup before coupling; pass
    critic_approved=True only after that review.
    
    The standard-library (preCICE) path for cross-code coupling — works for any
    number of participants, any data fields, any exchange pattern. openPASO generates
    the preCICE config and launches every participant's solver command. Use this
    when each side is a separate executable/script that talks preCICE (e.g. a DSMC
    particle code <-> a FEM solid; FSI; TSI). Each backend's preCICE participant
    pattern is available via knowledge(topic='precice', solver=...).
    
    Args (all JSON strings except scheme/numbers):
        participants: list of {"name","mesh","writes":[data],"reads":[data],
                      "command":[argv...]} — one per coupled code.
        data:      list of {"name","type":"scalar"|"vector"}.
        exchanges: list of {"data","from","to"} — one per coupled field.
        work_dir:  directory to run in (config + participant cwd).
        scheme:    serial-explicit|serial-implicit|parallel-explicit|parallel-implicit.
                   An EXPLICIT scheme takes one pass per time window and measures
                   no convergence at all — it cannot establish a coupled fixed
                   point, and the verdict here says so rather than implying one.
        dimensions, max_time, time_window, timeout: coupling controls.
        max_iterations, convergence_tol, relaxation: implicit-scheme controls
                   (ignored for explicit). These were previously not forwarded
                   at all, so every implicit coupling ran on the defaults
                   whatever the caller asked for.
        mapping:   nearest-neighbor|nearest-projection. Mapped with
                   constraint="consistent", which preserves nodal values and
                   NOT integrals — a flux/force field on a non-matching
                   interface is therefore not conserved, and the tool says so.
        extra_env: optional JSON dict of extra env (e.g. {"LD_LIBRARY_PATH":...,
                   "PYTHONPATH":...}) for the participant processes.
    
    Returns: JSON {exit_codes_ok, exchanged, coupling_converged, returncodes,
        config, logs, evidence, validation, checks_not_run}. Every participant
        exiting 0 is NOT by itself a coupling: an implicit scheme that exhausts
        max-iterations without meeting its convergence measure logs that and
        exits 0, and two scripts that never call preCICE at all exit 0 too. The
        verdict is built from preCICE's own per-window record, and an explicit
        scheme — which measures no convergence — is reported as unmeasured
        rather than as converged.
    ```
