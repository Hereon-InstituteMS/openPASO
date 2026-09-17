# `session_insights`

**Summarises what happened so far in this session.**

Group: Find out.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `action` | string | no | `'review'` |
| `path` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Review knowledge discovered during this session or from saved
    journals on disk.
    
    Two flows are supported:
    
    * In-session flow: call ``review`` -> ``approve_all`` /
      ``reject_all`` during the live MCP session to surface
      candidates from the current journal and save approved ones
      to ``data/community_knowledge/pending/``.
    * Ingest flow: call ``ingest`` with ``path`` pointing at a
      previously-saved session journal (``data/sessions/session_*.json``,
      which the server writes on shutdown) or at a directory of
      such files.  Candidates are surfaced just like ``review``
      and can be approved with ``approve_all``.
    
    Args:
        action:
            - "review" — show candidate knowledge from the current
              session for approval
            - "ingest" — load saved journal(s) from ``path`` and
              analyse them; requires ``path``
            - "approve_all" — approve all pending candidates and
              save to community_knowledge/pending/
            - "reject_all" — dismiss all pending candidates
            - "stats" — current session statistics
        path: file or directory used by the ``ingest`` action;
            ignored otherwise.  Directories are scanned for
            ``session_*.json``.
    ```
