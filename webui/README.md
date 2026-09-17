# openPASO WebUI

Browser-based front-end for the LangGraph + openPASO-MCP driver in
`langgraph_eval/`. Single-page app, no build step.

## What it has

- **Prompt box** + streamed chat log.
- **Model selector**: `qwen2.5-7b` / `14b` / `32b` (talks to vLLM
  servers on ports 8000-8002) or `mock` (no GPU; uses the same fake
  OpenAI server the smoke tests use).
- **MCP servers**: checkbox list (currently openPASO; designed to take
  more servers without code changes — add an entry to
  `webui/config.MCP_SERVERS`).
- **Mode buttons**: `plan` (every tool call needs Approve/Reject in the
  UI) · `accept` (auto-approves, default) · `autonomous` (same as
  accept, no UI interruption).
- **Event log** colour-coded by type: user/agent message, tool-call
  pending/executing/result, sub-agent spawn/return, token counts,
  status, errors.
- **Sub-agent visibility**: when the agent uses `spawn_subagent`, the
  spawn and the sub-agent's eventual return are first-class events;
  the depth-limited recursion is preserved.
- **Token counter** for the active session, persisted to the session
  JSON.
- **Session save / load / restart**: every session is a JSON file at
  `data/webui_sessions/<id>.json` and rehydrates on reconnect.
- **File browser** rooted at `eval_interactive/` with safe path
  resolution (symlink/escape attempts return 403). Clicking a file
  shows it in the side panel.
- **Auto-visualisation**:
  - `.csv` → Plotly line plot when columns are numeric.
  - `.json` → pretty-printed.
  - `.vtu` / `.pvd` / `.vtk` → vtk.js scaffold (see “Extending” below).
  - `.h5` / `.xdmf` → top-level dataset keys.
  - `.py` / `.cc` / `.yaml` / etc. → text view.
  - images → inline.
- **Interactive parameter sliders**: opening a Python script extracts
  top-level numeric assignments (`N = 32`, `dt = 0.005`, …) and lets
  you tweak them; the “Re-run with edited parameters” button sends a
  prompt that asks the agent to rerun the script with the new values.

## Running

```bash
# one-time deps install (lives in the LangGraph venv, not the main .venv)
.venv-lg/bin/pip install fastapi 'uvicorn[standard]' python-multipart \
                          pyvista websockets pytest

# start the UI
.venv-lg/bin/uvicorn webui.app:app --reload --port 8080
# open http://localhost:8080
```

The mock model works without any vLLM running, so the UI is fully
usable for click-testing flows before plugging in real Qwen weights.

To use real Qwens, start the vLLM servers as documented in
`langgraph_eval/README.md` (`./start_vllm.sh 7b`, etc.) and pick the
matching model in the sidebar.

## Tests

```bash
.venv-lg/bin/pytest webui/tests -v
```

10 tests, ~5 seconds, no GPU required. Covers config endpoints,
sandbox traversal safety (escape attempts blocked), file
classification, parameter extraction, session round-trip, end-to-end
agent flow (parent → spawn_subagent → critic returns → done) with the
mock LLM, and plan-mode tool-call gating with approval.

## Extending

* **VTK rendering.** `static/app.js::renderVtk(url)` is a stub. The
  raw file is exposed at `/sandbox-file/<rel>`; wire vtk.js's
  `HTTPDataAccessHelper` + reader for the file's format and attach to
  `#vtkRoot`. Stub left intentionally short so it's obvious where to
  put the renderer.
* **More MCP servers.** Add a row to `webui/config.MCP_SERVERS` with
  the command/args/cwd/env; the UI picks it up automatically.
* **More reasoning visibility.** If you swap in a reasoning model that
  emits `<think>…</think>` blocks, intercept `agent_chunk` events in
  the runner, classify, and emit a distinct `reasoning_chunk` event;
  the frontend's `eventClass()` switch already has space for it.
* **More tool gates.** `runner._wrap_tool` already attaches plan-mode
  approval to every tool. Per-tool policies (e.g., auto-approve
  `web_search` but gate `run_bash`) are a 5-line change in `_wrap_tool`.

## File layout

```
webui/
├── app.py             # FastAPI + WebSocket
├── runner.py          # Agent factory with event-streamed tools
├── sessions.py        # JSON snapshots
├── files.py           # Safe sandbox traversal
├── viz.py             # File → Plotly/vtk descriptor
├── config.py          # Models, MCP servers, modes
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── tests/
    └── test_app.py
```
