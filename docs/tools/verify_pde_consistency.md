# `verify_pde_consistency`

**Checks that the computed field actually satisfies the equation it claims to solve.**

Group: Check the answer.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solution_files` | string | yes |  |
| `source_term` | string | yes |  |
| `coefficient` | string | no | `'1.0'` |
| `domain` | string | no | `'[[0,1],[0,1]]'` |
| `equation` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Does your field actually satisfy the equation the task stated?
    
    A refinement study CANNOT answer this. Measured over 464 runs, result
    sets with a complete set of levels self-converge at a median order
    of 1.96 to 1.99 — the discretisation is fine — while a field that
    converges cleanly to the WRONG function looks identical in that study.
    Your own mesh-independence verdict does not separate them either: it
    catches three quarters of the wrong runs and also fires on half the
    correct ones.
    
    This checks the weak identity that any solution of
    -div(K grad u) = f obeys for a smooth v vanishing on the boundary:
    
        integral of u * (L* v)  ==  integral of f * v
    
    using ONLY the operator and source from your task and the values you
    already wrote. There is no reference solution in it.
    
    Measured separation, by execution: a field that solves the stated
    problem drives this residual down by roughly a factor of four per
    refinement, from 2.3e-02 to 2.6e-04 over four levels; a field that does
    not leaves it FLAT, near 6.4 at every level. The verdict is decided by
    whether it falls, not by any particular value.
    
    Args:
        solution_files: comma-separated paths to your solution CSVs, one
            per mesh level, each `x, y, u` (or `x, y, z, u`) with a header.
        source_term: the task's source term, in the task's own Python
            notation, e.g. "36*x**3*y - 20*x**3/3 + 8/3".
        coefficient: "2.5" for a scalar, or a symmetric tensor written as
            "[[3,-1],[-1,2]]". Must match what the task prescribes.
        domain: the box the problem lives on, "[[x0,x1],[y0,y1]]".
        equation: your task's equation, copied from its EQUATION line.
            REQUIRED. This check implements the second-order scalar
            diffusion form only, and it refuses anything else rather than
            answering about an operator it does not model.
    ```
