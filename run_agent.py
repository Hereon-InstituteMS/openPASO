#!/usr/bin/env python3
"""Run one simulation task with an AI model of your choice.

This is the simplest way to use openPASO. You need two things:
an OpenRouter API key, and a description of what you want to simulate.

    cp .env.example .env        # then paste your key into .env
    python run_agent.py "Solve the Poisson equation on a unit square
                         and check the convergence rate"

The model reads your sentence, picks a solver, writes the input file,
runs the solver, and checks the answer. Everything it does is printed
as it happens, so you can watch and stop it at any time with Ctrl-C.

If you would rather use openPASO from Claude Code, Claude Desktop or
Cursor, you do not need this file at all. See "Quick start" in README.md.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def load_env_file(path: Path) -> None:
    """Read KEY=value lines from .env into the environment.

    Kept deliberately small so that openPASO needs no extra package just to
    read one file. A value already set in the shell always wins, because an
    explicit choice should not be overridden by a file.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


def explain_and_exit(problem: str, fix: str, *, started: bool = False) -> None:
    """Print one sentence and one instruction, then stop.

    `started` SEPARATES TWO VERY DIFFERENT FAILURES. Everything used to print
    "openPASO cannot start", including a run that had started fine and worked
    for a hundred steps: a coupled task that hit the model's own context limit
    after 104 steps, and a trial that reached its step budget having already
    produced solver output, both told the reader the server had not started.
    That sends them to debug the wrong end of the system entirely.
    """
    headline = ("openPASO stopped partway." if started
                else "openPASO cannot start.")
    print(f"\n  {headline}\n\n  Problem: {problem}\n  Fix:     {fix}\n",
          file=sys.stderr)
    raise SystemExit(1)


def build_model(model_id: str, api_key: str, temperature: float):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=OPENROUTER_URL,
        api_key=api_key,
        model=model_id,
        temperature=temperature,
        timeout=600,
    )


def _web_search_tool():
    """The search the critic is told to use, or nothing at all.

    Its package is optional, so this returns None when it is absent rather than handing the model
    a tool that answers with an install hint. The tool itself distinguishes a blocked search from
    an empty web (langgraph_eval.agent.web_search) -- measured, a throttled DuckDuckGo returns an
    EMPTY LIST rather than raising, and five "no results" in one run were read as "the web has
    nothing", after which a critic quoted benchmark values from memory. Nothing is added on top of
    that wording here, so the model is told once.
    """
    try:
        import ddgs                                     # noqa: F401
    except Exception:                                   # noqa: BLE001
        try:
            import duckduckgo_search                    # noqa: F401
        except Exception:                               # noqa: BLE001
            return None
    from langgraph_eval.agent import web_search
    return web_search          # already a tool; passing it through keeps one description


def _spawn_subagent_tool(*, model_id: str, api_key: str, tools):
    """A sub-agent on the SAME model, for the critic and for coupled workers.

    openPASO's served instructions require an independent critic on every major step, and on a
    coupled problem require each participant script to be written by a worker sub-agent. Without
    this tool that text asks for something this path cannot do. The sub-agent gets the parent's
    tools except this one, so it cannot spawn further.

    It is the same model as the parent, so it is a second opinion, not an independent one. That
    limit is stated to the caller rather than implied away.
    """
    from langchain_core.tools import StructuredTool

    ROLES = {
        "critic": ("You are a ruthlessly critical reviewer. Challenge every parameter choice, "
                   "check units, look for sign errors, verify boundary conditions, and validate "
                   "against literature. A finding blocks only if it names a concrete, checkable "
                   "defect. Answer APPROVED: <why> or REJECTED: <the defect and its fix>."),
        "worker": ("You are a worker. Carry out exactly the step you were given, using the tools, "
                   "and report what you produced and where it is."),
        "verifier": ("You are an independent verifier. Re-derive the requested quantity by "
                     "another route and compare numerically."),
        "researcher": ("You are a research assistant. Find authoritative sources and summarise "
                       "what they say."),
    }

    async def spawn_subagent(role: str, task: str, context: str = "") -> str:
        try:
            from langchain.agents import create_agent as _make
            kw = "system_prompt"
        except ImportError:                              # noqa: BLE001
            from langgraph.prebuilt import create_react_agent as _make
            kw = "prompt"
        child_tools = [t for t in tools if getattr(t, "name", "") != "spawn_subagent"]
        agent = _make(build_model(model_id, api_key, 0.3), tools=child_tools,
                      **{kw: ROLES.get(role, ROLES["researcher"])})
        print(f"  ↳ {role} sub-agent working", flush=True)
        try:
            out = await agent.ainvoke({"messages": [("user", f"Task: {task}\n\nContext:\n{context}")]},
                                      {"recursion_limit": 40})
            report = str(out["messages"][-1].content)
            # Silence is not assent; see langgraph_eval.agent for the run that proved it.
            if report.strip():
                return report
            # One sentence for a silent sub-agent, in all three places a person can read it:
            # here, the campaign path and the browser interface.
            from langgraph_eval.agent import SILENT_SUBAGENT_REPORT   # noqa: PLC0415
            who = role if role in ("critic", "worker", "verifier", "researcher") else "sub-agent"
            return SILENT_SUBAGENT_REPORT.format(who=who)
        except Exception as exc:                         # noqa: BLE001
            return f"[sub-agent failed: {type(exc).__name__}: {exc}]"

    return StructuredTool.from_function(
        coroutine=spawn_subagent, name="spawn_subagent",
        description=("Spawn a sub-agent that runs on the same model as you: role in "
                     "{critic, worker, verifier, researcher}, task = what it should do, "
                     "context = the facts it needs. It has your tools but cannot spawn further. "
                     "Same model as you, so it is a second opinion, not an independent one."))


async def run(task: str, *, model_id: str, api_key: str, workdir: Path | None,
              temperature: float, step_limit: int, keep_going: int = 0) -> int:
    # LangGraph 1.0 moved this factory and renamed its prompt argument.
    try:
        from langchain.agents import create_agent
        prompt_kwarg = "system_prompt"
    except ImportError:
        from langgraph.prebuilt import create_react_agent as create_agent
        prompt_kwarg = "prompt"

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "src"))
    from core.instructions import INSTRUCTIONS
    from langgraph_eval.agent import openpaso_mcp_tools_session

    print(f"  model    {model_id}", flush=True)
    print(f"  task     {task}", flush=True)
    print("  starting the openPASO server ...\n", flush=True)

    # THE MODEL NEEDS A FILESYSTEM, NOT ONLY SOLVERS.
    #
    # This used to pass the MCP solver tools and nothing else -- no way to
    # write a file, no way to run a command. Measured on a coupled task: with
    # no filesystem the model used `run_simulation` as a substitute shell, and
    # since every call creates a fresh timestamped output directory it rebuilt
    # its participant tree EIGHTEEN times in eighteen places, calling couple()
    # fifteen times because it never had a stable tree to couple. The
    # evaluation harness gives its agent write_file and run_bash twenty times
    # each; the product path gave neither, and any task needing more than one
    # file was quietly impossible.
    #
    # The three factories are called directly rather than through
    # `_host_tools`, which builds `spawn_subagent` against a hard-coded
    # localhost vLLM (useless here) and `web_search` from a dependency that is
    # declared only in langgraph_eval/requirements-langgraph.txt. Both are
    # provided below instead: openPASO's own served instructions tell the model
    # to spawn a critic on every major step and a worker per step of a coupled
    # problem, so a path that does not carry the tool asks for something the
    # model cannot do. The browser interface builds its own for the same reason.
    from langgraph_eval.agent import (_bash_tool_for, _read_write_tools_for,
                                      cleanup_sandbox_scratch)

    print(f"  working in {workdir}", flush=True)
    print("  shell commands run with your own permissions, in that folder\n",
          flush=True)
    try:
        async with openpaso_mcp_tools_session(workdir, surface="all",
                                              isolate=False) as tools:
            # audit_on_submit stays at its default False: that flag switches on
            # the campaign's grading hooks (RESULT.txt, COULD_NOT_COMPLETE, the
            # *_level*.csv deliverable family), which mean nothing here.
            # advice=True turns on the LINT family only -- the checks that read
            # the script the model just wrote and name a call known to stop the
            # run on this install, or a deliverable filled in with a literal.
            # audit_on_submit stays False, so the campaign's grading hooks
            # (RESULT.txt, COULD_NOT_COMPLETE, the *_level*.csv family) stay off.
            tools = list(tools) + [
                _bash_tool_for(workdir, isolate=False, budget_note=False,
                               advice=True),
                *_read_write_tools_for(workdir, advice=True),
            ]
            searcher = _web_search_tool()
            if searcher is not None:
                tools.append(searcher)
            tools.append(_spawn_subagent_tool(
                model_id=model_id, api_key=api_key, tools=tools))
            print(f"  {len(tools)} tools ready\n" + "─" * 72, flush=True)
            agent = create_agent(build_model(model_id, api_key, temperature),
                                 tools=tools, **{prompt_kwarg: INSTRUCTIONS})
            history = [("user", task)]
            final = None
            for attempt in range(keep_going + 1):
                final = None
                async for step in agent.astream(
                        {"messages": history},
                        {"recursion_limit": step_limit},
                        stream_mode="values"):
                    message = step["messages"][-1]
                    final = message
                    history = list(step["messages"])
                    kind = getattr(message, "type", "")
                    for call in getattr(message, "tool_calls", None) or []:
                        print(f"  → {call['name']}({_short(call.get('args'))})", flush=True)
                    if kind == "tool":
                        # The solver's own answer goes to the model, not to the
                        # screen; a raw tool payload is long and is not written
                        # for a reader.
                        lines = str(getattr(message, "content", "")).count("\n") + 1
                        print(f"    ← {message.name}: {lines} line(s)", flush=True)
                        continue
                    text = getattr(message, "content", "")
                    if text and kind == "ai" and not getattr(message, "tool_calls", None):
                        print(f"\n{text}\n", flush=True)
                if attempt >= keep_going:
                    break
                print(f"  ── the model stopped without a tool call; continuing "
                      f"({attempt + 1} of {keep_going}) ──", flush=True)
                history.append(("user", _KEEP_GOING_NUDGE))
            print("─" * 72)
            if final is None:
                print("  the model returned nothing")
            elif keep_going == 0 and not _looks_finished(final):
                # A ONE-SHOT RUN ENDS WHEN THE MODEL STOPS CALLING TOOLS, AND A
                # MODEL OFTEN STOPS BY ASKING A QUESTION. Measured on a coupled
                # task here: it wrote both participant scripts, hit an API error,
                # printed "Would you like me to continue debugging ...?" and the
                # run ended at 20 of 200 allowed steps with none of the requested
                # files written. Nothing said the job was unfinished.
                print("  done — but the model ended by asking rather than "
                      "finishing.")
                print("  If you want it to carry on by itself, re-run with "
                      "--keep-going 8.")
            else:
                print("  done")
    finally:
        # webui/runner.py and validation/run_validation.py both do this;
        # the scratch path is a deterministic digest of the workdir, so
        # without it /tmp state survives between runs in the same folder.
        cleanup_sandbox_scratch(workdir)
    return 0


_KEEP_GOING_NUDGE = (
    "Your last turn ended without a tool call, so this task is not finished. "
    "Nothing will stop you except the step budget. Continue from exactly where "
    "you stopped, and do not ask whether to continue -- there is nobody to "
    "answer. If you have genuinely exhausted what you can do, say so plainly "
    "and state what is missing."
)


def _looks_finished(message) -> bool:
    """Did the model end by finishing, or by asking?

    Deliberately crude: it only decides which of two closing lines to print,
    never whether to keep working. A question mark in the last sentence of a
    final message is the shape of "shall I go on?", which is the case worth
    naming.
    """
    text = str(getattr(message, "content", "") or "").strip()
    return not text.endswith("?")


def _short(args, width: int = 90) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return text if len(text) <= width else text[:width - 3] + "..."



def _resolve_workdir(given) -> Path:
    """Where the model may write, and where `run_bash` starts.

    It used to default to None, which meant the MCP server was never told where
    the work belonged and the shell had no cwd at all. The help text already
    promised "the current folder"; this makes that true, and creates the folder
    because `run_bash` passes it as `cwd` and will not create it itself.
    """
    path = Path(given) if given else Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one simulation task with openPASO and an AI model.",
        epilog='example:  python run_agent.py "Solve Poisson on a unit square '
               'and verify the convergence rate"')
    ap.add_argument("task", nargs="*", help="what you want to simulate, in normal words")
    ap.add_argument("--model", help="model id; overrides OPENPASO_MODEL from .env")
    ap.add_argument("--workdir", type=Path,
                    help="folder for this run's files (default: the current folder)")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="0.0 is the most repeatable, 1.0 the most varied (default 0.2)")
    ap.add_argument("--keep-going", type=int, default=0, metavar="N",
                    help="if the model stops without finishing, tell it to "
                         "continue, up to N times. Use it for unattended runs; "
                         "0 (the default) ends when the model stops.")
    ap.add_argument("--step-limit", type=int, default=200,
                    help="stop after this many agent steps (default 200)")
    args = ap.parse_args()

    load_env_file(REPO / ".env")

    task = " ".join(args.task).strip()
    if not task:
        explain_and_exit("you did not say what to simulate",
                         'put the task in quotes, for example:\n'
                         '           python run_agent.py "Solve Poisson on a unit square"')

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        env_file = REPO / ".env"
        if not env_file.is_file():
            explain_and_exit(
                "there is no .env file yet, so openPASO has no API key",
                "from the openPASO folder run\n"
                "             cp .env.example .env\n"
                "           then open .env and paste your key from\n"
                "           https://openrouter.ai/keys after OPENROUTER_API_KEY=")
        explain_and_exit(
            f"OPENROUTER_API_KEY is empty in {env_file}",
            "open that file and paste your key from\n"
            "           https://openrouter.ai/keys after OPENROUTER_API_KEY=")

    model_id = (args.model or os.environ.get("OPENPASO_MODEL", "")).strip()
    if not model_id:
        explain_and_exit(
            "no model chosen",
            "set OPENPASO_MODEL in .env, or pass --model.\n"
            "           Pick one from https://openrouter.ai/models")

    try:
        return asyncio.run(run(task, model_id=model_id, api_key=api_key,
                               workdir=_resolve_workdir(args.workdir),
                               temperature=args.temperature,
                               step_limit=args.step_limit,
                               keep_going=max(0, args.keep_going)))
    except KeyboardInterrupt:
        print("\n  stopped by you")
        return 130
    except ModuleNotFoundError as missing:
        explain_and_exit(
            f"a Python package is missing: {missing.name}",
            "install the agent packages with\n"
            "           pip install -r langgraph_eval/requirements-langgraph.txt")
    except BaseException as error:               # noqa: BLE001 - reported below
        # By here the session has been entered, so any failure is mid-run.
        explain_and_exit(*_diagnose(error), started=True)
    return 1


def _diagnose(error: BaseException) -> tuple[str, str]:
    """Turn whatever went wrong into one sentence and one instruction.

    Exceptions arrive wrapped in an ExceptionGroup, because the MCP session
    and the agent run in the same task group, so the text is searched rather
    than the type matched.
    """
    def unwrap(exc, depth=0):
        """Every message in the chain: groups, causes and contexts alike."""
        if exc is None or depth > 8:
            return []
        found = [f"{exc.__class__.__name__}: {exc}"]
        for sub in getattr(exc, "exceptions", ()):       # ExceptionGroup
            found += unwrap(sub, depth + 1)
        found += unwrap(exc.__cause__, depth + 1)
        found += unwrap(exc.__context__, depth + 1)
        return found

    messages = unwrap(error)
    text = " | ".join(dict.fromkeys(messages))
    low = text.lower()
    # THE RUN GOT SOMEWHERE. Both of these arrive only after the server is up
    # and the model has been working, so they are reported as what they are.
    if "recursion limit" in low or "graphrecursionerror" in low:
        return ("the model reached its step budget before finishing",
                "raise it with  --step-limit N  (the default is 200), and see\n"
                "           --keep-going N if it also stops without asking")
    if "maximum context length" in low or "context_length" in low:
        return ("the conversation outgrew the model's context window",
                "start a fresh run on a smaller piece of the task, or pick a\n"
                "           model with a larger context at https://openrouter.ai/models")
    if "401" in low or "no auth" in low or "invalid api key" in low:
        return ("OpenRouter rejected the key",
                "check OPENROUTER_API_KEY in your .env file against\n"
                "           https://openrouter.ai/keys")
    if "402" in low or "credit" in low or "quota" in low:
        return ("your OpenRouter account is out of credit",
                "add credit at https://openrouter.ai/credits")
    if "404" in low and "model" in low:
        return (f"OpenRouter does not know the model id",
                "copy an id exactly from https://openrouter.ai/models\n"
                "           and put it in OPENPASO_MODEL in your .env file")
    if "does not support tool" in low or "tool use" in low:
        return ("the chosen model cannot use tools",
                "pick a model listed with tool support on\n"
                "           https://openrouter.ai/models")
    if "no interpreter found" in low:
        return ("the openPASO server has no Python to run in",
                "create it with  python3 -m venv .venv && .venv/bin/pip install -e .\n"
                "           or set OPENPASO_PYTHON in your .env file")
    specific = [m for m in messages if "TaskGroup" not in m]
    return ((specific[-1] if specific else text)[:400] or error.__class__.__name__,
            "if this is not clear, open an issue and paste this message")


if __name__ == "__main__":
    raise SystemExit(main())
