# `check_input`

**Checks an input file for known mistakes before anything is run.**

Group: Run.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `input_path` | string | no | `''` |
| `input_content` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Name the defects of an input deck or script BEFORE it runs: the setup checks the run
    tools apply, as a standalone gate for a code you run yourself (a coupling participant's
    own deck). For 4C every section name is judged by the INSTALLED binary's own grammar
    (`4C -p`) and the closest known names are listed, every condition's E id is checked
    against the topology sections (4C drops a condition on an undefined id silently and
    finishes on the wrong problem), and the measured TSI / Scalar_Transport deck defects
    are named -- all of them in one call, where the binary stops at the first. Reads the
    file, writes nothing; the findings are advisory and name no fix beyond the defect.
    
    Args:
        solver: backend name (e.g. 'fourc')
        input_path: path of the deck or script to check (or pass input_content)
        input_content: the text itself, when no file exists yet
    ```
