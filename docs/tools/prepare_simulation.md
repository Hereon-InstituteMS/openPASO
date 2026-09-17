# `prepare_simulation`

**The usual first step: collects knowledge, examples and a starting input file in one call.**

Group: Find out.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `physics` | string | yes |  |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Prepare everything needed to set up a simulation — in ONE call.
    
    Returns: knowledge + real test file examples + generated template.
    This eliminates 3 separate tool calls before every simulation.
    
    Supports fuzzy matching: e.g. 'magnetostatics' finds 'maxwell',
    'thermal' finds 'heat', and method-qualified requests select distinct
    recipes. Preserve qualifiers from the problem: use phrases such as
    'nearly incompressible Taylor-Hood elasticity', 'steady SIPG
    advection-diffusion', or 'Crank-Nicolson transient heat', rather than
    reducing them to a generic family.
    
    Args:
        solver: Backend name (e.g. 'fourc', 'fenics', 'ngsolve')
        physics: Physics and required method (e.g. 'poisson',
                 'nearly incompressible Taylor-Hood elasticity',
                 'steady SIPG advection-diffusion')
    ```
