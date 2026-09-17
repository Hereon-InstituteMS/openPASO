# openPASO — Project Instructions

## How to Handle Complex Simulation Tasks

For any non-trivial simulation task, break the work into sub-agents:

1. **Research agent**: Search online for the specific problem — find literature values, validated setups, correct parameters. Report findings.

2. **Setup agent**: Using the research findings, build the input files. Use `get_example_inputs()` to study real test files from the solver's test suite. Use `run_with_generator()` for simulations that need programmatic mesh/particle generation.

3. **Critic agent**: Independently review the setup. Check every parameter against what the research agent found. Verify units, discretization, boundary conditions, material properties. Search online to validate. If the simulation fails after multiple attempts with the chosen solver, consider whether an alternative solver might be more suitable — `prepare_simulation` shows what alternatives exist. Report any issues.

4. **Execution agent**: Fix any issues the critic found, then run the simulation.

5. **Results critic agent**: After the simulation completes, independently verify the results. Use `visualize(action='validate')` to check for NaN, constant fields, suspiciously large values. Check that field magnitudes are physically reasonable for the problem. Compare against known analytical solutions or literature values where possible.

6. **Coupling agent** (if multi-solver): Transfer fields between solvers, set up the coupling, run the second solver, produce combined output.

## Developer Mode: Extending Solvers

When you hit a **solver limitation** — a missing material model, element type, boundary condition, or physics module — don't just accept it. You have full source access via `developer(action='architecture', solver)` and the `<SOLVER>_ROOT` environment variables. Consider:

1. **Diagnose**: Read the solver source to understand why the capability is missing
2. **Assess**: Is it a small addition (new material, new BC) or a major feature? Report this to the user.
3. **Propose**: Tell the user "I could implement X in the solver source to fix this — should I proceed?"
4. **Implement** (if authorized): Modify source files, rebuild, re-run with the fix

This is what makes this agent a development **partner**, not just an operator. Even if you don't implement the fix, always report when a solver limitation is the root cause of a problem — don't silently accept workarounds.

## Mandatory Quality Control

- Every parameter must be justified — cite source (literature, test file, or derivation)
- For challenging problems, always search online for validated configurations
- Never take shortcuts on discretization — use what the literature recommends
- Use `run_with_generator()` for large simulations (particles, fine meshes) — write a Python generator script instead of inline content

## Gate: No Simulation Without Critic Approval

**Do not call `run_simulation()` or `run_with_generator()` until a critic agent has reviewed and approved the setup.** This is a hard precondition, not a workflow suggestion.

Before every simulation run, spawn a critic agent. The critic must:
- Check every numerical parameter against the research findings or literature
- Verify units and dimensional consistency
- Verify discretization is adequate (mesh size, particle spacing, time step) — compare against published benchmarks for this problem
- Verify boundary conditions and loading match the physical problem
- Search online if anything is uncertain
- Report a clear APPROVE or REJECT with reasons

If the critic rejects, fix the issues and re-submit to the critic. Only run the simulation after approval.

When creating your task plan, **include the critic review as an explicit task before each simulation task**. If the critic task is not in your plan, the plan is incomplete.
