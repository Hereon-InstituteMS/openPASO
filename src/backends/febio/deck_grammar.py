"""The FEBio .feb grammar — ONE copy, served by every path that needs it.

WHY THIS EXISTS, and why it is not a violation of Option B. FEBio is a BINARY
that consumes an XML deck, so the deck is this backend's run interface in
exactly the way exports.json is the coupling driver's: it cannot be guessed, and
no solve happens without it. openPASO elides "the solve" by design — correct for a
Python library, and for a file-driven code it leaves the agent unable to start.

MEASURED on the payload a single-code FEBio agent receives (21,787 characters
for `heat`): no `<febio_spec`, no `<MeshDomains`, no `<Boundary`, no `<node id=`
and no `fix=`. The corpus is rich on material models and pitfalls and silent on
the document that carries them. The same shape of gap was measured for 4C, where
it cost all three openPASO-arm runs of coupled cell C2 their whole attempt.

WHAT IS SERVED IS THE GRAMMAR, NOT A SOLVE. Every number below is an arbitrary
placeholder; the agent still chooses its own mesh, material parameters, boundary
data and body force, and still interprets its own results. The skeleton was
taken from a deck assembled by this repo's own fixture library and RUN: exit 0,
"N O R M A L   T E R M I N A T I O N", 54 equations.

THE TRAPS BELOW WERE EACH MEASURED BY EXECUTION on this FEBio build (4.12):
they are the same body-force findings already verified in the FEBio backend's
own trap text, restated here because an agent that cannot write the document
never reaches them.
"""

FEBIO_DECK_GRAMMAR = """\
THE FEBio DECK GRAMMAR — you cannot guess it, and FEBio is a BINARY, so the
deck is this backend's run interface in the way exports.json is the driver's.
Every NUMBER below is an ARBITRARY PLACEHOLDER; the sections and their order
are what is documented. A deck with these sections runs.

    <?xml version="1.0" encoding="ISO-8859-1"?>
    <febio_spec version="4.0">
    <Module type="solid"/>
      <Control>
        <analysis>STATIC</analysis>
        <time_steps>2</time_steps>
        <step_size>0.5</step_size>
        <solver type="solid">
          <symmetric_stiffness>symmetric</symmetric_stiffness>
        </solver>
      </Control>
      <Material>
        <material id="1" name="Material1" type="isotropic elastic">
          <density>1.0</density><E>1000.0</E><v>0.25</v>
        </material>
      </Material>
      <Mesh>
        <Nodes name="AllNodes">
          <node id="1">0,0,0</node>
        </Nodes>
        <Elements type="hex8" name="Part1">
          <elem id="1">1,2,3,4,5,6,7,8</elem>
        </Elements>
        <NodeSet name="bottom">1</NodeSet>
      </Mesh>
      <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
      </MeshDomains>
      <Boundary>
        <bc name="fix" type="zero displacement" node_set="bottom">
          <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
        </bc>
      </Boundary>
    </febio_spec>

  * `<MeshDomains>` IS NOT OPTIONAL. It binds each element block to a material
    by NAME; without it the elements have no constitutive law and FEBio stops.
  * A NODE SET IS REFERENCED BY NAME from `<bc node_set="...">`, and the set
    must be declared inside `<Mesh>`.
  * PLANE STRAIN IN A 3-D CODE is a slab one element thick with the
    out-of-plane displacement fixed on BOTH z faces — a `zero displacement` bc
    with `<z_dof>1</z_dof>` on each. Read the task's own prescription for the
    through-thickness element count and follow it rather than refining there.

  A POSITION-DEPENDENT BODY FORCE, measured on this build (4.12):
  * It IS writable, as a `<body_load type="const">` inside `<Loads>` whose
    component carries `type="math"`:
        <Loads>
          <body_load type="const">
            <z type="math">-1*X^2</z>
          </body_load>
        </Loads>
  * OMITTING `type="math"` IS USUALLY SILENT: the component is read as a
    number, the leading numeric prefix is taken and the rest of the expression
    is DISCARDED, so `-1*X^2` becomes the constant -1 and the run succeeds with
    the wrong load. Measured: the resulting displacement was bit-identical to a
    constant-load run.
  * `**` IS A PARSE ERROR here, not a silent one: `-1*X**2` gives
        Token expected (position 6)
    while `-1*X^2` is accepted. The task states its source term in Python
    notation, so rewrite EVERY term.
  * A `<body_load>` placed in `<LoadData>` is a hard `unrecognized tag`
    failure. The tempting next step — deleting the tag — removes the load
    entirely and the run then succeeds with f = 0, which is worse.

  READ THE LOG, NOT THE EXIT CODE. FEBio prints
      N O R M A L   T E R M I N A T I O N
  letter-spaced on success and
      E R R O R   T E R M I N A T I O N
  on failure, and reports `Nr of equations ......... : <n>`, which is the
  number to cross-check against your mesh. Grepping for the contiguous string
  "normal termination" never matches.
"""
