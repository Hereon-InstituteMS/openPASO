# Troubleshooting

## Install

| What you see | What to do |
|---|---|
| `Unknown compiler(s)`, `metadata-generation-failed` on numpy | You are on Python 3.13 without a C compiler. Install one, or use Python 3.12. See [Install](getting-started/install.md) |
| `No module named 'mcp.server.fastmcp'` | An old openPASO allowed mcp 2.x. Update openPASO and run `pip install -e .` again |
| `openPASO itself does not import` | The virtual environment is not active: `source .venv/bin/activate` |
| Kratos installs, then fails to import | Use `pip install KratosMultiphysics-all`, not plain `KratosMultiphysics` |
| `Please run pip install mpi4py before rerunning your Dune script.` | `pip install mpi4py` |
| A solver works in the terminal but openPASO says it is missing | Its variable (`FEBIO_BINARY`, `FENICS_PYTHON`, ...) must also be in your AI app's `env` block |
| deal.II examples do not compile | Your deal.II is older than 9.3. Build a newer one, or use another solver |

## Option A: the app does not see openPASO

| What you see | What to do |
|---|---|
| `claude mcp list` shows openpaso as failed | Run `claude mcp get openpaso`. `PYTHONPATH` must be a full path starting with `/`. If it shows only `src`, remove it and add it again from inside the openPASO folder |
| openpaso is missing in another folder | It was added for one folder only. Add it again with `-s user` |
| `⏸ Pending approval` | You used `-s project` or `-s local`. Remove it and add it again with `-s user` |
| The app lists it, but every tool call fails | The `command` must point at `.venv/bin/python` inside openPASO, not at your system Python |
| Claude Desktop shows nothing at all | Quit the app completely; check the JSON is valid; check no `/path/to/` is left; on Windows, double every backslash |

## Option B: `run_agent.py`

| Message | What to do |
|---|---|
| `there is no .env file yet` | `cp .env.example .env`, then paste your key into it |
| `OPENROUTER_API_KEY is empty in ...` | The key line in `.env` is blank. Paste your key after `OPENROUTER_API_KEY=` |
| `OpenRouter rejected the key` | The key is wrong. Check it at <https://openrouter.ai/keys> |
| `your OpenRouter account is out of credit` | Add credit at <https://openrouter.ai/credits> |
| `the chosen model cannot use tools` | Pick a model with tool support on <https://openrouter.ai/models> |
| `a Python package is missing` | `pip install -r langgraph_eval/requirements-langgraph.txt` |
| `the openPASO server has no Python to run in` | Your `.venv` is missing. Redo [Install](getting-started/install.md) |

Still stuck? [Open an issue](contribute.md#report-a-problem). Say what you ran, what you expected,
and paste the output of `python check_install.py`.
