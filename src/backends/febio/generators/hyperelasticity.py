"""FEBio hyperelasticity generators and knowledge.

FEBio Module type: 'solid'. Registered and run-proved material types:
'neo-Hookean', 'Mooney-Rivlin', 'Ogden', 'Yeoh', 'Veronda-Westmann',
'Holmes-Mow'.

'neo-Hookean' and 'Holmes-Mow' are COUPLED (compressible) formulations
parameterised by E and v. 'Mooney-Rivlin', 'Ogden', 'Yeoh' and
'Veronda-Westmann' are UNCOUPLED formulations and additionally require
a bulk modulus <k>.
"""


def _hex_block(nx, ny, nz):
    """Structured hex8 unit cube. Returns (nodes, elems, nid)."""
    def nid(i, j, k):
        return 1 + i + (nx + 1) * j + (nx + 1) * (ny + 1) * k
    nodes = [(nid(i, j, k), i / nx, j / ny, k / nz)
             for k in range(nz + 1)
             for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    e = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elems.append((e, [nid(i, j, k), nid(i + 1, j, k),
                                  nid(i + 1, j + 1, k), nid(i, j + 1, k),
                                  nid(i, j, k + 1), nid(i + 1, j, k + 1),
                                  nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]))
                e += 1
    return nodes, elems, nid


def _hyperelasticity_3d_cube(params: dict) -> str:
    """Resolved 3D cube under prescribed-displacement uniaxial loading.

    Default: 4x4x4 hex8, neo-Hookean, 30% nominal compression applied
    over 10 equal load steps. params: E, nu, displacement, n (elements
    per edge), time_steps, material, material_params.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    dz = params.get("displacement", -0.3)
    n = int(params.get("n", 4))
    nsteps = int(params.get("time_steps", 10))
    mat = params.get("material", "neo-Hookean")
    matp = params.get("material_params", f"<E>{E}</E><v>{nu}</v>")

    nodes, elems, nid = _hex_block(n, n, n)
    xn = "\n".join(f'      <node id="{i}">{x:.10g},{y:.10g},{z:.10g}</node>'
                   for i, x, y, z in nodes)
    xe = "\n".join(f'      <elem id="{i}">' + ",".join(str(c) for c in cc) + "</elem>"
                   for i, cc in elems)
    bot = ",".join(str(nid(i, j, 0)) for j in range(n + 1) for i in range(n + 1))
    top = ",".join(str(nid(i, j, n)) for j in range(n + 1) for i in range(n + 1))

    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>{nsteps}</time_steps>
    <step_size>{1.0 / nsteps:.10g}</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <max_refs>25</max_refs>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="{mat}">
      <density>1.0</density>{matp}
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
{xn}
    </Nodes>
    <Elements type="hex8" name="Part1">
{xe}
    </Elements>
    <NodeSet name="fix_bottom">{bot}</NodeSet>
    <NodeSet name="load_top">{top}</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">{dz}</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile>
      <node_data data="x;y;z" file="pos.txt"/>
      <element_data data="sx;sy;sz;J" file="el.txt"/>
    </logfile>
    <plotfile type="febio">
      <var type="displacement"/><var type="stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "hyperelasticity": {
        "description": (
            "Nonlinear hyperelasticity with FEBio under Module "
            "type='solid'. Coupled (compressible) models take E and v; "
            "uncoupled models take their own constants plus a bulk "
            "modulus k. The deck skeleton — section order, name= on "
            "materials, MeshDomains linking by name, load controllers — "
            "is identical to linear_elasticity; see that entry first."
        ),
        "input_format": "FEBio XML (.feb), version 4.0",
        "solver": "Newton-Raphson, default quasi-Newton strategy BFGS",
        "templates": [
            "hyperelasticity_3d_cube — resolved 4x4x4 hex8 cube, "
            "neo-Hookean, uniaxial prescribed displacement over 10 load "
            "steps; params E, nu, displacement, n, time_steps, material, "
            "material_params.",
        ],
        "materials": {
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Holmes-Mow": {"E": "Young's modulus", "v": "Poisson's ratio",
                           "beta": "power exponent"},
            "Mooney-Rivlin": {"c1": "1st constant", "c2": "2nd constant",
                              "k": "bulk modulus, REQUIRED"},
            "Yeoh": {"c1": "1st constant", "c2": "2nd constant",
                     "c3": "3rd constant (c4, c5, c6 also accepted)",
                     "k": "bulk modulus, REQUIRED"},
            "Veronda-Westmann": {"c1": "stress-like constant, must be > 0",
                                 "c2": "dimensionless exponent, must be > 0",
                                 "k": "bulk modulus, REQUIRED"},
            "Ogden": {"c1": "1st modulus (c2..c6 also accepted)",
                      "m1": "1st exponent, must be non-zero "
                            "(m2..m6 also accepted)",
                      "k": "bulk modulus, REQUIRED"},
        },
        "quasi_newton_strategies": (
            "The FENEWTONSTRATEGY_ID types registered in 4.12 are BFGS, "
            "Broyden, JFNK, 'modified Newton' and 'full Newton'. Only "
            "BFGS and Broyden accept a <max_ups> parameter."
        ),
        "pitfalls": [
            (
                "[Numerical] Uncoupled hyperelastic materials REQUIRE a "
                "bulk modulus <k>; omitting it is rejected before "
                "anything is solved. Mooney-Rivlin, Ogden, Yeoh and "
                "Veronda-Westmann are uncoupled and need it; neo-Hookean "
                "and Holmes-Mow are coupled, take E and v, and have no "
                "<k> parameter at all. "
                "WRONG: <material id=\"1\" name=\"M1\" "
                "type=\"Mooney-Rivlin\"><density>1.0</density>"
                "<c1>100.0</c1><c2>50.0</c2></material>. "
                "RIGHT: <material id=\"1\" name=\"M1\" "
                "type=\"Mooney-Rivlin\"><density>1.0</density>"
                "<c1>100.0</c1><c2>50.0</c2><k>10000.0</k></material>. "
                "Other run-proved spellings: "
                "type=\"Ogden\" with <c1>100.0</c1><m1>2.0</m1>"
                "<c2>50.0</c2><m2>-2.0</m2><k>10000.0</k>; "
                "type=\"Yeoh\" with <c1>100.0</c1><c2>20.0</c2>"
                "<c3>5.0</c3><k>10000.0</k>; "
                "type=\"Veronda-Westmann\" with <c1>100.0</c1>"
                "<c2>5.0</c2><k>10000.0</k>; "
                "type=\"Holmes-Mow\" with <E>1000.0</E><v>0.3</v>"
                "<beta>1.5</beta>. "
                "Signal: `K must be a positive number.` and exit 1. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] Splitting the load into more steps is the "
                "fix for a non-converging large-deformation run, but the "
                "threshold is far higher than 10% strain and it depends "
                "on the material and on the sign of the load. Swept "
                "time_steps = 1 / 2 / 10 / 50 against stretch levels on "
                "2x2x2 and 4x4x4 hex8 meshes, E = 1000, v = 0.3, STATIC. "
                "A SINGLE load step converged for neo-Hookean at 10%, "
                "50% and 100% nominal tensile strain on both meshes, and "
                "on up to 300%; the first single-step failure in tension "
                "was at 400% on both meshes. Mooney-Rivlin is weaker: it "
                "converged in one step at 10% and 50% and failed at 100% "
                "tensile strain, where TWO steps were already enough. "
                "Compression is much harder than tension: a single step "
                "carried neo-Hookean to 50% shortening on both meshes and "
                "failed at 70%, where 10 steps converged. Beyond 85% "
                "shortening the 4x4x4 mesh failed at 1, 10 and 50 steps "
                "alike — more substeps stopped helping. The failure is "
                "also not monotone in the load level: on the 2x2x2 mesh a "
                "single step failed at 400% tensile strain and succeeded "
                "at 500%. "
                "WRONG: assuming 10 substeps are needed past 10% strain, "
                "or assuming more substeps always fix a failure. "
                "RIGHT: start with <time_steps>10</time_steps>"
                "<step_size>0.1</step_size> and raise the count only "
                "after a run fails; if raising it does not help, the "
                "elements are inverting and the mesh or the load path is "
                "the problem. "
                "Signal: `N negative jacobians detected.` (N is the "
                "count of inverted integration points) inside an ERROR "
                "box, then `------- failed to converge at time : <t>`, "
                "then `E R R O R   T E R M I N A T I O N` and exit 1. "
                "`Number of time steps completed` reports how far it "
                "got before the failure. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] <density> is OPTIONAL on every FEBio solid "
                "material and silently defaults to 1.0. Under "
                "<analysis>STATIC</analysis> with no body load that is "
                "harmless; under <analysis>DYNAMIC</analysis> it is a "
                "clean run with the wrong physics. Verified on a "
                "neo-Hookean cube under DYNAMIC uniaxial loading: "
                "omitting the tag reproduced the <density>1.0</density> "
                "answer to every printed digit, while the intended "
                "<density>1000</density> gave a completely different "
                "deformation, with nothing in the log to flag it. "
                "<density>0.0</density> is also accepted and reduces "
                "DYNAMIC to the STATIC answer, again to every printed "
                "digit. "
                "WRONG: <material id=\"1\" name=\"M1\" "
                "type=\"neo-Hookean\"><E>1000.0</E><v>0.3</v></material> "
                "under <analysis>DYNAMIC</analysis>. "
                "RIGHT: <material id=\"1\" name=\"M1\" "
                "type=\"neo-Hookean\"><density>1000.0</density>"
                "<E>1000.0</E><v>0.3</v></material>. "
                "Signal: none — the run ends "
                "`N O R M A L   T E R M I N A T I O N` with exit 0 and "
                "all steps completed either way. Always write <density> "
                "explicitly for any dynamic or body-loaded model. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] Choose STATIC for quasi-static loading and "
                "DYNAMIC only when the load ramp is comparable to the "
                "wave transit time of the body. The inertial error is "
                "CONTINUOUS in the ramp rate — there is no rate at "
                "which DYNAMIC leaves a static answer untouched, only "
                "rates at which the difference stops mattering. "
                "Measure the transit time as t_w = L x sqrt(density / "
                "E) and compare it to the ramp duration. "
                "Executed on a 4x4x4 neo-Hookean cube, E = 1000, "
                "density = 1, edge length 1 (t_w about 0.03 time "
                "units), ramping the same prescribed stretch over 1.0, "
                "0.1, 0.03 and 0.01 time units in 100 steps under both "
                "analyses and comparing the free lateral surface: at "
                "about thirty transit times the two analyses agree to "
                "well under a percent; at about three transit times "
                "they differ by roughly ten percent; at one transit "
                "time by more again; and below about a third of a "
                "transit time the free surface moves in the WRONG "
                "DIRECTION — it expands outward while the specimen is "
                "being stretched, instead of contracting by Poisson's "
                "effect. That sign reversal is the unmistakable "
                "symptom; the percentage errors above it are a "
                "gradient, not a cliff, so do not look for a "
                "threshold. An earlier catalogued claim of a roughly "
                "50% displacement overshoot at the load-onset step was "
                "not reproduced at any ramp rate. "
                "WRONG: <analysis>DYNAMIC</analysis> with a load ramp "
                "shorter than the wave transit time, for a problem where "
                "equilibrium was wanted. "
                "RIGHT: <analysis>STATIC</analysis>, or keep DYNAMIC and "
                "make time_steps x step_size at least ten times the "
                "transit time L x sqrt(density / E). "
                "Signal: none — both analyses end "
                "`N O R M A L   T E R M I N A T I O N` with exit 0 and "
                "all steps completed. Detect it by running the same deck "
                "under both analyses and comparing a free-surface node "
                "out of <Output><logfile><node_data data=\"x;y;z\" "
                "file=\"pos.txt\"/></logfile></Output>. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] FINAL TIME = time_steps x step_size, not a step "
                "count. FEAnalysis::Solve() runs while endtime minus the "
                "current time exceeds eps, with endtime = step_size x "
                "time_steps, so writing time_steps=10 with step_size=1.0 "
                "because you wanted ten increments up to t=1 instead runs "
                "to t=10, and a load controller with "
                "<extend>CONSTANT</extend> then holds full load for nine "
                "extra time units. "
                "WRONG: <time_steps>10</time_steps>"
                "<step_size>1.0</step_size> with a load curve whose last "
                "point is <pt>1,1</pt>. "
                "RIGHT: <time_steps>10</time_steps>"
                "<step_size>0.1</step_size> with that same load curve. "
                "Signal: the run is clean, "
                "`N O R M A L   T E R M I N A T I O N` with exit 0 and "
                "`Number of time steps completed` equal to 10 in BOTH "
                "cases — the step count cannot tell them apart. The last "
                "`*Time` block in the logfile CSV reads 10 for the first "
                "and 1 for the second. Always compare the final `*Time` "
                "against time_steps x step_size. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Solver] <max_ups>, the number of quasi-Newton updates "
                "between stiffness reformations, belongs to the "
                "<qn_method> property, one level DEEPER than in the 3.x "
                "schema, and only the BFGS and Broyden strategies accept "
                "it. "
                "WRONG: <solver type=\"solid\"><max_ups>10</max_ups>"
                "</solver>. "
                "WRONG: <solver type=\"solid\">"
                "<qn_method type=\"JFNK\"><max_ups>10</max_ups>"
                "</qn_method></solver> — and the same for "
                "type=\"modified Newton\" and type=\"full Newton\". "
                "RIGHT: <solver type=\"solid\">"
                "<symmetric_stiffness>symmetric</symmetric_stiffness>"
                "<qn_method type=\"BFGS\"><max_ups>10</max_ups>"
                "</qn_method></solver>, or type=\"Broyden\" in place of "
                "BFGS. "
                "Signal: tag \"max_ups\" (line N) : `unrecognized tag` "
                "with `Reading file ...FAILED!` and exit 1, both for the "
                "flat form and for the three strategies that do not own "
                "the parameter; an unregistered strategy name gives "
                "tag \"qn_method\" (line N) : `invalid value for "
                "attribute \"type\"` instead. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "hyperelasticity_3d_cube": _hyperelasticity_3d_cube,
}
