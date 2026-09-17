# `developer`

**Reads and changes a solver's own source code, then rebuilds it.**

Group: Set up and develop.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `action` | string | yes |  |
| `solver` | string | no | `''` |
| `keyword` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Developer tools: architecture, source files, capabilities matrix.
    
    Args:
        action: What to surface. Options:
            - "architecture" — extension points + source-tree
              layout for the requested solver
            - "files" — source-file listing filtered by keyword
            - "capabilities" — full backend × physics × variant
              matrix dump
        solver: Backend name
        keyword: File pattern for "files" action
    ```
