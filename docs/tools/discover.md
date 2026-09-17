# `discover`

**Lists the solvers, which ones are installed on this machine, and what each can do.**

Group: Find out.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `query` | string | no | `'list'` |
| `solver` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Discover available solvers and their capabilities.
    
    Args:
        query: What to discover. Options:
            - "list" — list all solvers with status
            - "physics" — list all physics types per solver
            - "capabilities" — full capabilities matrix
            - "recommend" — recommend solver for a physics (set solver= to physics name)
            - "coupling" — how to couple two codes together: which tool,
              which backend can take which side, and the exact knowledge
              calls that return each side's participant contract (handshake,
              sign convention, flux recovery, exports schema; the solve is
              yours to write)
        solver: Filter by solver name, or physics name for "recommend"
    ```
