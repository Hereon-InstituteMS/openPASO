"""The SPARTA input-script grammar — ONE copy, served wherever it is needed.

WHY THIS EXISTS. SPARTA is a BINARY driven by a command script, so the script is
this backend's run interface in exactly the way exports.json is the coupling
driver's: it cannot be guessed, and no run happens without it. openPASO elides "the
solve" by design — correct for a Python library, and for a file-driven code it
leaves the agent with nothing to start from.

MEASURED on the payload a single-code SPARTA agent received for `heat`: 1,084
characters in total, with no `fix` and no `run` — the two commands without which
the binary does nothing at all. The same shape of gap was measured for 4C (where
it cost three development runs of one coupled problem their whole attempt) and
for FEBio.

WHAT IS SERVED IS THE GRAMMAR, NOT A SOLVE. Every number is an arbitrary
placeholder; the agent still chooses its own domain, grid, species, mixture and
sampling. The command sequence and its ORDER are taken from this repo's own
executed SPARTA fixtures.

SPARTA IS A DSMC CODE, NOT A FEM CODE. It samples a stochastic estimate, so a
Monte-Carlo noise floor exists that no tolerance can go below — which is why a
residual that stops falling is not automatically a defect.
"""

SPARTA_INPUT_GRAMMAR = """\
THE SPARTA INPUT-SCRIPT GRAMMAR — you cannot guess it, and SPARTA is a BINARY
driven by this script. Every NUMBER is an ARBITRARY PLACEHOLDER; the commands
and THEIR ORDER are what is documented. Order matters: a command that refers to
something not yet defined is an error.

    seed            12345
    dimension       2
    global          gridcut 0.0 comm/sort yes
    boundary        p p p                 # per axis: p periodic, o outflow,
                                          # s specular, r diffuse-reflect
    create_box      0 1e-3 0 1e-3 -0.5 0.5
    create_grid     10 10 1               # 2-D: the z count must be 1
    balance_grid    rcb cell

    species         ar.species Ar         # a species FILE must exist
    mixture         gas Ar temp 300.0
    global          nrho 1e21 fnum 1e11   # number density and particles/real
    collide         vss gas ar.vss        # a collision FILE must exist

    timestep        1e-8
    compute         gt temp
    fix             f ave/time 10 10 100 c_gt    # nevery nrepeat nfreq
    stats           100
    stats_style     step cpu np f_f
    run             300

  * `create_box` TAKES THREE PAIRS EVEN IN 2-D. The z pair is still required;
    give it a unit-thickness slab such as -0.5 0.5.
  * `create_grid`'s THIRD COUNT MUST BE 1 in two dimensions, or the run is
    silently three-dimensional and every per-cell quantity changes meaning.
  * `species` AND `collide` READ FILES (ar.species, ar.vss ship with SPARTA).
    Copy them next to the script or give a path; a missing file aborts.
  * `fix ave/time nevery nrepeat nfreq` MUST SATISFY
        nevery * nrepeat <= nfreq   and   nfreq % nevery == 0
    or SPARTA rejects it. The averaging WINDOW is nevery*nrepeat steps ending at
    each nfreq multiple, so a quantity is an average over that window and not an
    instantaneous value — read a converged value from the LAST window, not from
    a single step.
  * NOTHING IS COMPUTED WITHOUT `run`, and nothing is SAMPLED without a `fix`
    or `compute`. A script that ends after `collide` exits cleanly having done
    nothing, which reads as success.

  READING THE RESULT. SPARTA prints a stats table whose columns follow
  `stats_style`, and ends with
      Loop time of <t> on <p> procs for <n> steps with <m> particles
  which is the line that proves it ran and carries numbers to cross-check
  (steps, particle count). It also writes log.sparta in the working directory.

  IT IS A STOCHASTIC METHOD. Two runs with different `seed` values give
  different numbers, and the spread is physical, not a bug: there is a
  Monte-Carlo noise floor no tolerance can go below. Report a quantity with the
  sampling window that produced it, and if a residual stops falling at that
  floor say so rather than tightening the tolerance.
"""
