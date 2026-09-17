# Option B: your own API key

**OpenRouter** gives you one key for many AI models, and you pay for what you use. Three steps, all
from the openPASO folder:

```bash
# 1. Copy the settings file
cp .env.example .env

# 2. Open .env in any editor and paste your key after OPENROUTER_API_KEY=
#    Get a key at https://openrouter.ai/keys
#    A model is already chosen for you, so you can leave the rest alone.

# 3. Ask for a simulation (the prompt must show (.venv); if not: source .venv/bin/activate)
python run_agent.py "Solve the Poisson equation on the unit square with scikit-fem, and report the maximum value."
```

!!! tip "Your key stays on your computer"
    It lives only in the file `.env` in the openPASO folder. git never uploads that file.

!!! tip "Start with a cheap model"
    `.env.example` lists several and already selects one. To use another, change `OPENPASO_MODEL` in
    `.env`, or add `--model some/model-id` to the command.

## Did it work?

`run_agent.py` prints everything as it happens:

- `→ toolname(...)`: the model is calling a solver tool
- `← toolname: N line(s)`: the tool answered
- plain text: the model talking to you
- `done` on the last line: the run finished

The files the solvers produce are written to **`simulation_outputs/`** inside openPASO (change it
with `OPENPASO_OUTPUT_DIR` in `.env`).

If something went wrong, `run_agent.py` says so in one sentence and tells you what to do. The common
cases are listed under [Troubleshooting](../troubleshooting.md).
