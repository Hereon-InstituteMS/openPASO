<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/openPASO_koralle_dunkel.gif">
    <source media="(prefers-color-scheme: light)" srcset="logo/openPASO_graphit_transparent.gif">
    <img src="logo/openPASO_koralle_dunkel.gif" alt="openPASO" width="200">
  </picture>
</p>

<h1 align="center">openPASO</h1>

<p align="center"><b>open Platform for Agentic Simulation and Optimization</b></p>

<p align="center">
  <a href="https://pypi.org/project/openpaso/"><img src="https://img.shields.io/pypi/v/openpaso?style=flat-square&color=FF6B4A&label=PyPI" alt="openpaso on PyPI"></a>
  <a href="https://open-paso.github.io/openPASO/"><img src="https://img.shields.io/badge/docs-website-FF6B4A?style=flat-square" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-64748B?style=flat-square" alt="MIT licence"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10--3.13-64748B?style=flat-square" alt="Python 3.10 to 3.13"></a>
  <a href="https://doi.org/10.5281/zenodo.20543501"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20543501-64748B?style=flat-square" alt="DOI"></a>
  <a href="https://open-paso.github.io/openPASO/solvers/"><img src="https://img.shields.io/badge/solvers-9-FF6B4A?style=flat-square" alt="9 solvers"></a>
</p>

<p align="center">
  <i>Describe a physics problem in plain words.<br>
  An AI model picks a solver, writes its input, runs it, and checks the answer.</i>
</p>

<p align="center">
  <a href="https://open-paso.github.io/openPASO/">
    <img src="docs/assets/openPASO_film.gif" alt="openPASO in 29 seconds: describe the physics, openPASO picks the solver, runs it, and checks the result against the literature" width="800">
  </a>
  <br><sub>Click the film to watch it in full quality on the documentation website.</sub>
</p>

> [!WARNING]
> **openPASO is under active development, and we invite you to help.** It works and is used for
> real simulations, but it is young: things change quickly and you will find rough edges. Try it,
> tell us what broke, and share what you know about a solver.
> See [Contribute](https://open-paso.github.io/openPASO/contribute/).

## What it does

Simulation programs, called **solvers**, compute how things physically behave: how a part bends,
how heat spreads, how air flows. They are powerful and hard to use, and each one has its own input
format and its own traps. openPASO puts **nine of them behind one door** and lets an AI model open
it: you describe what you want in a normal sentence, and the model picks a solver, writes the input,
runs it, and checks that the answer is computed correctly.

## Quick start

You need Python 3.10–3.13 and about ten minutes. One solver is enough to begin.

```bash
pip install openpaso             # the server, with scikit-fem as a first solver
openpaso                         # starts the MCP server on stdio; Ctrl-C to stop
```

`openpaso` is the command your AI app's MCP configuration points at. Working from a checkout
instead (to change the code, or to run `check_install.py`, which lists the solvers openPASO can
use on your machine without a key or network):

```bash
git clone https://github.com/open-PASO/openPASO.git
cd openPASO
python3 -m venv .venv && source .venv/bin/activate
pip install -e . scikit-fem
python check_install.py
```

Then connect an AI model, one of two ways:

- **Option A — an AI app you already have** (Claude Code, Claude Desktop, Cursor): no extra cost.
  [Set it up →](https://open-paso.github.io/openPASO/use/ai-app/)
- **Option B — your own OpenRouter key**: one command per simulation, any model.
  [Set it up →](https://open-paso.github.io/openPASO/use/api-key/)

## Solvers

| | | |
|---|---|---|
| [scikit-fem](https://open-paso.github.io/openPASO/solvers/skfem/) | [NGSolve](https://open-paso.github.io/openPASO/solvers/ngsolve/) | [Kratos Multiphysics](https://open-paso.github.io/openPASO/solvers/kratos/) |
| [DUNE-fem](https://open-paso.github.io/openPASO/solvers/dune/) | [FEniCSx](https://open-paso.github.io/openPASO/solvers/fenics/) | [deal.II](https://open-paso.github.io/openPASO/solvers/dealii/) |
| [FEBio](https://open-paso.github.io/openPASO/solvers/febio/) | [4C Multiphysics](https://open-paso.github.io/openPASO/solvers/fourc/) | [SPARTA](https://open-paso.github.io/openPASO/solvers/sparta/) |

## Documentation

Everything else is on the **[documentation website](https://open-paso.github.io/openPASO/)**:
installing each solver, both ways of use step by step, all 24 tools the model gets, how openPASO
checks an answer, coupling two solvers on one problem, troubleshooting, and a glossary for every word.

## Contribute

Reports of what did not work, solver traps you know, and plain-language fixes to the documentation
are all very welcome. One rule stands above the rest: **every improvement must help all simulations,
not one example.** See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Contribute page](https://open-paso.github.io/openPASO/contribute/).

## Licence and citation

MIT licence. openPASO was first published as **OASiS**; the archived releases and the paper use that
name. If you use openPASO in research, please cite:

```bibtex
@software{openpaso2026,
  title  = {openPASO: an open-source multi-physics and multi-code framework
            for verified computer simulations},
  author = {Hermann, Alexander and Shojaei, Arman and Scheider, Ingo and Cyron, Christian},
  year   = {2026},
  doi    = {10.5281/zenodo.20543501},
  url    = {https://github.com/open-PASO/openPASO}
}
```

<p align="center">
  <sub>Helmholtz-Zentrum Hereon · Institute of Materials Mechanics &nbsp;·&nbsp;
  Hamburg University of Technology</sub>
</p>
