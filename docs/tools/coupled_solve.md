# `coupled_solve`

**A ready-made coupled setup for common pairs of physics.**

Group: Two solvers on one problem.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `problem` | string | no | `'heat_dd'` |
| `solver_a` | string | no | `'fenics'` |
| `solver_b` | string | no | `'fourc'` |
| `nx` | integer | no | `32` |
| `ny` | integer | no | `32` |
| `max_iter` | integer | no | `20` |
| `tol` | number | no | `1e-06` |
| `relaxation` | number | no | `1.0` |
| `params` | string | no | `'{}'` |
| `critic_approved` | boolean | no | `False` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    LEGACY cross-solver coupling — FIXED toy geometries only. PREFER `couple`.
    
    DEPRECATED: this tool only handles a fixed enum of benchmark problems on a
    hardcoded unit-square split at x=0.5 (heat_dd/poisson_dd/one_way/tsi_dd/...).
    For ANY real or non-benchmark coupling use the general `couple` tool, which is
    physics-agnostic and validates the result for silent-wrong (flux balance,
    convergence, finiteness). Only use coupled_solve to reproduce the legacy
    benchmarks.
    
    Domain A (Dirichlet at interface) supports: fenics, ngsolve, skfem, dune.
    Domain B (Neumann at interface) supports: fenics, fourc, ngsolve, skfem, dune.
    Any combination of these works for heat_dd and poisson_dd problems.
    
    Args:
        problem: 'heat_dd', 'poisson_dd', 'one_way',
                 'poisson_dd_study', 'l_bracket_tsi', 'heat_dd_precice'.
                 'tsi_dd' is REMOVED: it reported converged=True,
                 iterations=1, residual=0.0 on a run that did one thermal
                 solve and one one-way structural solve and never fed
                 anything back. Two-way TSI goes through `couple` — see the
                 shipped participant_tsi_* scripts.
        solver_a, solver_b: Backend names
        nx, ny: Elements per direction
        max_iter: Max iterations
        tol: Convergence tolerance
        relaxation: Under-relaxation parameter
        params: JSON with additional parameters
        critic_approved: Set True after critic review
    ```
