# How it is built

```
You ──▶ AI model (an app, or your API key) ──▶ MCP ──▶ openPASO (src/server.py)
                                                            │
   ┌──────────┬──────────┬────────┬─────────┬────────┬──────┴───┬────────┬────────┐
FEniCSx    deal.II       4C    NGSolve   skfem    Kratos     DUNE    FEBio   SPARTA
(Python)    (C++)     (YAML)  (Python) (Python)  (JSON)   (Python)  (XML)  (native)
```

| Part | Where | What it does |
|---|---|---|
| Server | `src/server.py` | the MCP server the AI app or `run_agent.py` talks to |
| Backends | `src/backends/<solver>/` | one package per solver: the physics it supports, input writers, known traps |
| Registry | `src/core/registry.py` | loads the backends as plugins and finds which are installed |
| Tools | `src/tools/` | the [tools](../tools/index.md) the model calls |
| Knowledge | `data/`, `src/backends/*/` | settings and traps read from the version you actually installed |
| Coupling | `src/core/coupling_driver.py` | runs two participants until they agree ([coupling](coupling.md)) |
| preCICE bridge | `src/core/precice_config.py` | the `couple_precice` tool |
| Command line agent | `run_agent.py`, `langgraph_eval/` | Option B |
| Install check | `check_install.py` | the key-free check |
