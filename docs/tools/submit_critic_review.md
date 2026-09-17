# `submit_critic_review`

**Records the independent review of the setup that has to happen before a run.**

Group: Check the answer.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `findings` | string | yes |  |
| `setup` | string | no | `''` |
| `coupling_args` | string | no | `''` |
| `ttl_s` | number | no | `3600.0` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Put an independent critic's review of a setup ON RECORD, so a run of
    that setup can be verified.
    
    REQUIRED, and the most common mistake: pass EXACTLY ONE of `setup` or
    `coupling_args`. `setup` is the deck text for run_simulation /
    run_with_generator / verify_mesh_independence; `coupling_args` is a
    JSON object for `couple` / `couple_precice` / `coupled_solve`. Passing
    neither — or both — is refused, and the refusal comes AFTER you have
    written the review, so the review is wasted. Measured over the
    development runs, half of all reviews handed in were rejected this way.
    
    openPASO's critic requirement is enforced, not requested. The run and
    coupling tools do not take your word for it: they look up whether THIS
    server holds a review of the EXACT setup being executed. Passing
    critic_approved=True without a matching review here leaves the result
    NOT VERIFIED, whatever else the run does.
    
    The workflow is: spawn a sub-agent as an independent critic; have it
    challenge the parameters, units, discretisation, problem statement and
    boundary conditions, and cross-check against literature and benchmarks;
    then submit what it actually found here; then run.
    
    The review is bound to the setup by digest, so a setup edited after
    review no longer matches and must be reviewed again. That is deliberate:
    reviewing a clean deck and running a different one is the obvious way to
    defeat a critic requirement, and it is the route this closes.
    
    This server cannot judge whether a critique was any GOOD — it is not an
    oracle for review quality. It enforces that a substantive review of this
    setup exists and is auditable, and refuses a review too short to have
    said anything.
    
    Args:
        solver: the backend the run will use. For `couple` pass "couple",
            for `couple_precice` pass "couple_precice", and for the legacy
            `coupled_solve` pass "<solver_a>-><solver_b>".
        findings: what the critic actually checked and concluded. Substance
            is required; an empty approval is indistinguishable from no
            review and is refused.
        setup: for run_simulation / run_with_generator /
            verify_mesh_independence — the EXACT deck text you will run
            (input_content, generator_script, or input_template).
        coupling_args: for the coupling tools instead of `setup` — a JSON
            object of the arguments you will pass. Keys per tool:
            coupled_solve: problem, solver_a, solver_b, nx, ny, max_iter,
            tol, relaxation, params; couple: participants, max_iter, tol,
            accelerator, theta, monolithic, probe; couple_precice:
            participants, data, exchanges, scheme, dimensions, max_time,
            time_window, max_iterations, convergence_tol, relaxation,
            mapping. Pass EVERY key for the tool you will call, with the
            values you will call it with — a missing or different key is a
            different setup and the run will come back NOT VERIFIED. For
            `couple` the CONTENTS of each participant's script are part of
            the setup too, so the scripts must already be written when the
            review is submitted, and editing one afterwards invalidates it.
        ttl_s: how long the review stays valid (default 1 hour).
    
    Returns: JSON with a `critic_token`. Passing it to run_simulation or
        run_with_generator makes the review single-use and binds it to that
        job; omitting it still works, since the deck is matched by digest.
    ```
