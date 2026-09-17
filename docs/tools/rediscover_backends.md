# `rediscover_backends`

**Looks for solvers again, for example after you installed one.**

Group: Set up and develop.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `confirm` | boolean | no | `False` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Probe the system for solver backends and report findings.
    
    Searches pip packages, conda environments, common build directories,
    and source roots. In developer mode, reports git branch and status.
    
    Args:
        confirm: If True, save the discovered config for future sessions.
                 If False (default), just report what was found.
    ```
