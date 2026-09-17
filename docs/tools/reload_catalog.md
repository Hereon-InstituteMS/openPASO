# `reload_catalog`

**Reloads the knowledge catalogue after it changed.**

Group: Set up and develop.

## Parameters

This tool takes no parameters.

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Hot-reload the per-backend KNOWLEDGE dicts from disk.
    
    Closes the gap identified by the
    mcp-catalog-staleness-runtime-isolation post-mortem
    (2026-06-01): the MCP server normally imports
    src/backends/<be>/generators/<physics>.py modules ONCE at
    startup and never refreshes them, so catalog edits made
    during a long-running session are invisible. Postmortems
    in data/postmortems/ are scanned on every request (already
    hot), but pitfall dicts are not.
    
    This tool walks every imported `backends.<be>.generators.*`
    and `backends.<be>.backend` module, runs importlib.reload
    on each, and re-runs load_all_backends() so the registry
    re-binds the backend objects to the refreshed module
    attributes. After the call, the very next
    mcp__openpaso__knowledge call returns the on-disk
    catalog without having to restart Claude Code.
    
    Returns a one-line summary of which modules were
    successfully reloaded vs which raised, so the caller can
    tell when a syntax error in a newly-edited generator
    prevented its module from re-importing (in that case the
    OLD dict is still served from the previous import).
    ```
