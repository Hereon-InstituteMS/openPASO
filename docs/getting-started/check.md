# Check that it works

One command. It needs no API key, no AI app and no network:

```bash
python check_install.py
```

It prints which of the nine solvers openPASO can see on your machine, and for each one it cannot
see, the command that installs it. If you installed scikit-fem, it must appear with a tick:

```
✔ Python 3.12 — supported.
✔ openPASO imports, and its tools are registered.

Solvers openPASO can see on this machine — 1 of 9:

  ✔ scikit-fem
      scikit-fem 12.0.2 at /home/you/openPASO/.venv/bin/python
  ✘ NGSolve
      not installed — to get it:  python -m pip install ngsolve
  ...

✔ Install is working: 1 solver(s) ready.
```

**One tick is enough to go on.**

| If it says | Do this |
|---|---|
| `openPASO itself does not import` | Your virtual environment is not active: `source .venv/bin/activate`, then try again |
| `No solver is usable` | `pip install scikit-fem`, then check again |
| a solver you installed is missing | If you set its location with `export`, run the check in the same terminal. For an AI app, the same variable also has to go into the app's settings (see [Option A](../use/ai-app.md)) |

Next: [choose how to use it](../use/index.md).
