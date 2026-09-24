# Option A: an AI app

The app brings the AI model. openPASO brings the solvers. You only connect the two once.

**Installed with `pip install openpaso`?** Then the server is the `openpaso` command and the
setup is one line -- Claude Code: `claude mcp add openpaso -s user -e PYVISTA_OFF_SCREEN=true -- openpaso`;
Claude Desktop or Cursor: `"command": "openpaso"` (and `"args": []`) in the configuration below,
no `PYTHONPATH` or `VIRTUAL_ENV` needed. The instructions that follow are for a checkout.

=== "Claude Code"

    Run this once, from the openPASO folder:

    ```bash
    claude mcp add openpaso -s user \
      -e PYTHONPATH="$PWD/src" \
      -e VIRTUAL_ENV="$PWD/.venv" \
      -e PYVISTA_OFF_SCREEN=true \
      -- "$PWD/.venv/bin/python" -m server
    ```

    `$PWD` means "the folder I am in right now", so it fills in the full path for you. Three details
    matter, and each one fails **silently** if you get it wrong:

    - **`-e` must come before the `--`.** Everything after `--` is the command itself.
    - **The paths must be full paths.** Your app starts openPASO from its own folder, so a relative
      path like `src` points at nothing.
    - **`-s user`** makes openPASO available in every folder. Without it, it only exists in this one.

    Check that it worked:

    ```bash
    claude mcp list          # openpaso should be listed, and ✔ Connected
    ```

    If Claude Code was already running, restart it. If you see `⏸ Pending approval`, you used
    `-s project` or `-s local`: remove it with `claude mcp remove openpaso` and run the command above again.

=== "Claude Desktop"

    First get your full path. In the openPASO folder run `pwd` and copy what it prints.

    Then open **Settings → Developer → Edit Config** and add this block.
    **`/path/to/openPASO` appears three times. Replace all three.**

    ```json
    {
      "mcpServers": {
        "openpaso": {
          "command": "/path/to/openPASO/.venv/bin/python",
          "args": ["-m", "server"],
          "env": {
            "PYTHONPATH": "/path/to/openPASO/src",
            "VIRTUAL_ENV": "/path/to/openPASO/.venv",
            "PYVISTA_OFF_SCREEN": "true"
          }
        }
      }
    }
    ```

    - **If the file is empty**, paste the whole block.
    - **If it already has an `"mcpServers"` section**, add only the `"openpaso": { ... }` part inside
      it, with a comma after the entry before it. A second outer `{ ... }` makes the file invalid, and
      the app ignores it without a word.
    - Before saving, search the block for `/path/to/`. If anything is left, you missed one.

    **Then quit Claude Desktop completely and start it again.** Closing the window is not enough.

    **Check:** open the tools menu in the message box; `openpaso` should be listed. If not: you did
    not fully restart; the file is not valid JSON (paste it into a JSON checker); or the `command` path
    does not exist (`ls /path/to/openPASO/.venv/bin/python`).

=== "Windows (Claude Desktop)"

    Two things change, and both fail silently if you skip them: the Python program is at
    `.venv\Scripts\python.exe`, and **every backslash must be written twice** in JSON.

    ```json
    {
      "mcpServers": {
        "openpaso": {
          "command": "C:\\Users\\you\\openPASO\\.venv\\Scripts\\python.exe",
          "args": ["-m", "server"],
          "env": {
            "PYTHONPATH": "C:\\Users\\you\\openPASO\\src",
            "VIRTUAL_ENV": "C:\\Users\\you\\openPASO\\.venv",
            "PYVISTA_OFF_SCREEN": "true"
          }
        }
      }
    }
    ```

    Get your own path with `cd` (Command Prompt) or `pwd` (PowerShell) in the openPASO folder, then
    double every backslash when you paste it.

=== "Cursor, Windsurf, others"

    Use the command `/path/to/openPASO/.venv/bin/python`, the arguments `-m server`, and **all three**
    settings from the Claude Desktop `env` block. Then quit the app completely and start it again.

!!! important "Solver settings must also go into the app"
    Your AI app does not start openPASO from your terminal, so it never sees your `export` lines or
    your `~/.bashrc`. Every solver variable you set, such as `FEBIO_BINARY`, `FENICS_PYTHON`,
    `FOURC_BINARY`, `SPARTA_BINARY` or `DUNE_PYTHON`, must also go into the `env` block as another
    `"NAME": "value"` line. Otherwise the solver works in your terminal and openPASO still reports it
    as missing.

## Ask for a simulation

Inside the app, simply write:

> Solve the Poisson equation on a unit square with a known exact solution, and check that the error
> falls at the expected rate.

See [what a run looks like](first-run.md), and [Troubleshooting](../troubleshooting.md) if the app
does not see openPASO.
