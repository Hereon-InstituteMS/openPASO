# `write_participant_contract`

**Writes the fill-in participant script for one solver of a coupled problem into your folder: everything around the solve, which you then write yourself.**

Group: Two solvers on one problem.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `path` | string | yes |  |
| `variant` | string | no | `''` |
| `overwrite` | boolean | no | `False` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Write the served participant CONTRACT for `solver` to `path`, solve elided.
    
    The file is byte-for-byte the text `knowledge(topic='coupling', solver=...,
    signal='participant[:<variant>]:part<k>')` serves in parts, concatenated:
    the contract with its handshake, its checks and its recovery, and the
    SOLVE elided where the banner sits. It is not a runnable program; fill the
    marked hole(s) yourself. `variant` is one of thermoelastic, neumann,
    elastic, transient, 3d (the same words the knowledge door takes).
    Refuses to overwrite an existing file unless overwrite=True.
    ```
