# What a run looks like

This is a real run of Option B, so you can see the shape of a session before you start one. With
Option A you ask the same thing inside your app and see the same tool calls there.

```console
$ python run_agent.py "Use the discover tool to list available solvers, then answer in one sentence."

  model    deepseek/deepseek-v4.1-flash
  starting the openPASO server ...

  24 solver tools ready
────────────────────────────────────────────────────────────────────────
  → discover(query='list')
    ← discover: 1 line(s)

All nine backends are available on this install: 4C Multiphysics,
FEniCSx (dolfinx 0.10.0), deal.II 9.8.0-pre, FEBio, NGSolve 6.2.2604,
scikit-fem 12.0.1, Kratos Multiphysics 10.3, DUNE-fem, and SPARTA (DSMC).

────────────────────────────────────────────────────────────────────────
  done
```

## Things to ask for

- **Classic equations**: Poisson, heat conduction, diffusion
- **Solids**: bending and stretching, large deformation, plasticity, contact, vibration frequencies
- **Fluids**: cavity flow, flow past an obstacle, flow in a channel
- **Transport over time**: heat that changes with time, reactions that spread
- **Electromagnetics**: waves, cavity resonances, magnets
- **Particle methods**: materials modelled as many interacting particles
- **Two solvers on one problem**: see [coupling](../how-it-works/coupling.md)
- **Meshes** with Gmsh: an L-shape, a plate with a hole, a channel, or your own
- **Accuracy studies**: refine the mesh step by step and measure how fast the error shrinks

Good prompts say what the physics is, the shape and size, what is fixed at the edges, and what number
or picture you want back. For example:

> Flow past a cylinder at Reynolds number 100. Show me the wake, and compare the drag coefficient with
> published values.
