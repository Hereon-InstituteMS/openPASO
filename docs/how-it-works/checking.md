# Checking the answer

A wrong result that looks right is worse than no result. openPASO is built around that sentence.
These are the checks, in plain words.

## 1. A known exact answer to compare against

For many problems openPASO can build the problem backwards from an answer it already knows (this is
called a *manufactured solution*). Then the error is a number, not an opinion.

## 2. The error must shrink at the right speed

Make the mesh twice as fine, and the error must fall by the amount the mathematics predicts. If it
does not, the run failed, however good the picture looks. The tool `verify_mesh_independence` does
this for a single solver, and `couple_levels` for two coupled solvers.

## 3. The answer must satisfy its own equation

`verify_pde_consistency` takes the field the solver produced and checks that it really satisfies the
equation it claims to solve, without needing the exact answer.

## 4. Proof that the solver really ran

openPASO reads the solver's own console output and its own data files. A number with no run behind
it is reported as exactly that. A file that exists is not accepted as proof: some solvers write
complete-looking output files for a run that computed nothing.

## 5. A second opinion before the run

A second AI instance reviews the setup before anything runs: units, boundary conditions, mesh
fineness, material values. openPASO keeps the record of that review and looks it up, instead of
believing a claim that it happened.

## 6. Known traps are named before they cost a run

Every input file and script is checked against the solver traps openPASO knows, **at the moment it
is written**. When a run fails, the error message is matched against measured failures, and the
reply names the cause and the fix.

## What is reported as failure

- Two coupled solvers that never agree: failure, never a result.
- A coupling that "converged" but exchanged nothing across the interface: named as such.
- A result file filled with a constant, or with zeros where a field was expected: named as such.
- A residual that looks excellent while the two sides' files disagree: named as such.

!!! note "Verification, not validation"
    All of this checks that the numbers are **computed correctly**. None of it checks that your
    model describes **reality**. That is validation, and it stays your job.
