---
hide:
  - navigation
  - toc
---

<div class="openpaso-hero" markdown>

![openPASO](assets/logo.png){ .logo }

# openPASO

**open Platform for Agentic Simulation and Optimization**

<p class="tagline">Describe a physics problem in plain words.<br>
An AI model picks a solver, writes its input, runs it, and checks the answer.</p>

<video class="openpaso-film" controls muted loop playsinline poster="assets/openPASO_film.gif">
  <source src="assets/openPASO_film.mp4" type="video/mp4">
  <img src="assets/openPASO_film.gif" alt="openPASO in 29 seconds">
</video>

[Install](getting-started/install.md){ .md-button .md-button--primary }
[How to use it](use/index.md){ .md-button }
[Contribute](contribute.md){ .md-button }

</div>

!!! warning "openPASO is under active development"
    It works, and people use it for real simulations, but it is young. Things change from week
    to week, some solvers are much better supported than others, and you will find rough edges.
    **We warmly invite you to help**: try it, tell us what broke, and add what you know about a
    solver. See [Contribute](contribute.md).

## What it does

A **solver** is a program that computes how something physically behaves: how a metal part bends,
how heat spreads through a wall, how air flows around a cylinder. Most of them use the **finite
element method**: they cut the object into many small pieces, a **mesh**, and compute the answer
piece by piece.

These programs are powerful and hard to use. Each has its own input format, its own words for the
same thing, and its own traps. Learning one takes weeks.

openPASO puts **nine such programs behind one door** and lets an AI model open it. You write what
you want in a normal sentence. The model chooses a suitable program, writes the input for it, runs
it, reads the real output, and checks whether the answer is right. openPASO gives the model what it
needs for that: what each setting means **in the version you have installed**, which settings break,
what an error message really means, and how to check a result.

!!! note "What openPASO checks, and what it does not"
    openPASO checks whether the **numbers are computed correctly**: that the solver really ran, and
    that the answer stops changing as the mesh gets finer. It does **not** check whether your model
    describes reality. Choosing the right physics is still your job.

## Where to go next

<div class="grid cards" markdown>

- **New here?** Start with [Install](getting-started/install.md). It takes about ten minutes, and
  one solver is enough.
- **Already installed?** [Choose how to use it](use/index.md): with an AI app you already have, or
  with your own API key.
- **Curious what it can do?** Browse the [solvers](solvers/index.md) and the
  [tools](tools/index.md) the model gets.
- **Want to know if you can trust it?** Read [Checking the answer](how-it-works/checking.md).

</div>
