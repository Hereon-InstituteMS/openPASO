# Two solvers on one problem

Some problems are best split in two: a solid and a fluid, a hot part and a cold part, a region
where one method is good and a region where another is. openPASO can give each half to a different
solver and make them agree.

## How it works

1. The problem is split along an **interface**, the boundary the two halves share.
2. Each half is a small script for its own solver, called a **participant**. Every participant
   follows one simple rule: it reads what the other side sent from `imports.json`, solves once, and
   writes its own interface values to `exports.json`.
3. openPASO's driver runs the two participants in turn. One side receives a value (for example a
   temperature), the other a flux (for example a heat flow).
4. This repeats until both sides agree at the interface to within a tolerance.
5. The whole thing is repeated on finer meshes (`couple_levels`), so the accuracy of the coupled
   answer can be measured as well.

Because each side only talks through those two small files, **any pair of solvers can be coupled**,
also solvers that were never designed to work together.

## What openPASO checks

- that each participant follows the file contract, before it runs;
- that the two sides actually agree, measured from their own exported values, not from an internal
  number that can look converged while the sides still disagree;
- that the interface carried something: a coupling that exchanged nothing converges immediately and
  looks perfect, so openPASO names it;
- that what leaves one side arrives at the other (`verify_interface_flux`);
- that the error falls at the right rate as the meshes get finer.

## preCICE

[preCICE](https://precice.org/) is a separate, widely used coupling library. openPASO can use it
instead of its own driver through the `couple_precice` tool.
