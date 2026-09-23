# Web interface

A browser page where you describe a simulation in plain words and watch openPASO pick a solver,
write its input, run it and report back. Every step is visible while it happens.

It is a third way to use openPASO, beside [an AI app](ai-app.md) and [your own key](api-key.md).

## Start it

From the openPASO folder, with your virtual environment active:

```bash
pip install -e ".[webui]"
uvicorn webui.app:app --port 8080
```

Then open **http://localhost:8080**. The ready-built interface is included, so you do not need
Node.js to use it.

## Choose a model

The interface needs an AI model, and says for each one whether it can work right now:

| Group | What it is | What it needs |
|---|---|---|
| **Hosted on OpenRouter** | pay per use; the price is shown | your key, in `.env` in the openPASO folder |
| **Claude Code** | the `claude` command on your own machine, on your own Claude account | Claude Code installed and signed in once |
| **On this machine** | a local model server | that server running |

A model that cannot work is shown with the reason and cannot be picked. If none works, the start
page says so and lists the ways to get one.

## Ask for a simulation

Write what you want in normal words, for example:

> Flow past a cylinder at Reynolds number 100. Show me the wake, and compare the drag with
> published values.

You can **attach files** (a mesh, a geometry, an input deck, measurement data). They land in the
run's `uploads/` folder and the model is told where they are.

Before starting, choose how much openPASO may do on its own:

- **Run without asking** — it runs tools, commands and solvers by itself.
- **Ask before each step** — every step waits for your *Run this step* or *Skip it*.
  (Not available with Claude Code.)

## While it runs

- **The run lives on the server, not in your browser tab.** Close the tab, go back, or open the run
  in another tab: it keeps going. Several runs can work at the same time.
- **Sending a message during a run is a correction.** It is handed to the model **when the current
  step finishes**, not immediately, so nothing is interrupted half-way. A message after the run has
  ended is a follow-up in the same conversation.
- **"End this step"** ends the processes of one step that is stuck, for example a solver that hangs.
  The run itself continues.
- **"Stop"** ends the whole run and everything it started, solvers included, and tells you how many
  processes it ended.
- You can watch what the model does, the pictures the run produced, and the fields it wrote.

## What the result means

When a turn ends, the interface says how it ended. The words are exact:

| It says | It means |
|---|---|
| **Finished** | A solver really ran **and** openPASO checked the result |
| **Ran, not verified** | A solver produced a result, but it was not checked. The numbers may still be wrong |
| **Ended** | The turn ended without a solver result. The interface says what happened instead: no tools were used, a solver call computed nothing, or the numbers came from a script the model wrote itself |
| **Stopped** | You ended it |

"Finished" is the only one that includes a check of the answer.

## Files and records

- **Files** shows the run's own folder.
- **Download the run record** gives you the prompt, the model, every step and a checksum for every
  file produced.
- Deleting a run removes its record and its folder.

!!! warning "Files outside the run's folder stay behind"
    A run may read and write outside its own folder, because a solver sometimes has to. Anything it
    wrote elsewhere is **not** removed when you delete the run. If that matters to you, run
    openPASO under a separate user account or in a container.

## Good to know

- Run records are JSON files in `data/webui_sessions/`, and run folders are in `eval_interactive/`.
- The interface is younger than the rest of openPASO and changes often. If something does not work,
  [please report it](../contribute.md#report-a-problem).
