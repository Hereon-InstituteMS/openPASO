"""FEBio fluid-FSI generators and knowledge.

FEBio Module type: 'fluid-FSI'. Strongly-coupled monolithic fluid-
structure interaction: a deformable solid and an incompressible fluid
share a moving interface. The fluid mesh deforms with the solid
(ALE). Hallmark FEBio module for arterial-wall hemodynamics, cardiac
chamber modeling, and any compliant-vessel benchmark.
"""


def _fluid_fsi_3d_block(params: dict) -> str:
    """Deformable block carrying an internal fluid, using the `fluid-FSI`
    MATERIAL in the `fluid-FSI` MODULE, consolidated by a prescribed
    compression of its top face.

    The fluid velocity and the dilatation degree of freedom are fully
    constrained. That is required, not cosmetic: on a USE_MKL=OFF build
    the FSI family does not converge with the fluid velocity free. The
    earlier version of this template left it free and failed at the
    first step at every ALE stiffness from 1 to 1e6. See the [Solver]
    pitfall.
    """
    n = max(1, int(params.get("n", 4)))
    W = float(params.get("width", 0.2))
    L = float(params.get("height", 1.0))
    E = float(params.get("E", 100.0))
    rho_f = float(params.get("density_fluid", 1.0))
    mu = float(params.get("viscosity", 0.01))
    uz = float(params.get("compression", -0.05))
    steps = max(1, int(params.get("time_steps", 10)))
    dt = float(params.get("step_size", 0.02))

    def nid(i, j, k):
        return 1 + i + (n + 1) * (j + (n + 1) * k)

    nodes = [f'      <node id="{nid(i,j,k)}">'
             f'{i/n*W},{j/n*W},{k/n*L}</node>'
             for k in range(n + 1) for j in range(n + 1)
             for i in range(n + 1)]
    el, e = [], 0
    for k in range(n):
        for j in range(n):
            for i in range(n):
                e += 1
                c = (nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k),
                     nid(i, j+1, k), nid(i, j, k+1), nid(i+1, j, k+1),
                     nid(i+1, j+1, k+1), nid(i, j+1, k+1))
                el.append(f'      <elem id="{e}">'
                          + ",".join(map(str, c)) + "</elem>")
    bot = ",".join(str(nid(i, j, 0))
                   for j in range(n+1) for i in range(n+1))
    top = ",".join(str(nid(i, j, n))
                   for j in range(n+1) for i in range(n+1))
    sides = ",".join(str(nid(i, j, k))
                     for k in range(n+1) for j in range(n+1)
                     for i in range(n+1) if i in (0, n) or j in (0, n))
    alln = ",".join(str(nid(i, j, k))
                    for k in range(n+1) for j in range(n+1)
                    for i in range(n+1))
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="fluid-FSI"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>{steps}</time_steps>
    <step_size>{dt}</step_size>
    <solver type="fluid-FSI">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="FSIBlock" type="fluid-FSI">
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>0.0</v>
      </solid>
      <fluid type="fluid">
        <density>{rho_f}</density>
        <k>1e3</k>
        <viscous type="Newtonian fluid">
          <mu>{mu}</mu>
        </viscous>
      </fluid>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
{chr(10).join(nodes)}
    </Nodes>
    <Elements type="hex8" name="Part1">
{chr(10).join(el)}
    </Elements>
    <NodeSet name="bot">{bot}</NodeSet>
    <NodeSet name="top">{top}</NodeSet>
    <NodeSet name="sides">{sides}</NodeSet>
    <NodeSet name="all_nodes">{alln}</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="FSIBlock"/>
  </MeshDomains>
  <Initial>
    <ic name="ef0" type="initial fluid dilatation" node_set="all_nodes">
      <value>0.0</value>
    </ic>
  </Initial>
  <Boundary>
    <bc name="fixbot" type="zero displacement" node_set="bot">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="confine" type="zero displacement" node_set="sides">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>0</z_dof>
    </bc>
    <bc name="zeroef" type="zero fluid dilatation" node_set="all_nodes"/>
    <bc name="fluid_at_rest" type="zero fluid velocity" node_set="all_nodes">
      <wx_dof>1</wx_dof><wy_dof>1</wy_dof><wz_dof>1</wz_dof>
    </bc>
    <bc name="squash" type="prescribed displacement" node_set="top">
      <dof>z</dof>
      <value lc="1">{uz}</value>
      <relative>0</relative>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>SMOOTH</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>{steps*dt},1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile>
      <node_data data="z;ef" delim="," file="fluid_fsi_nodes.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "fluid_fsi": {
        "description": (
            "Strongly-coupled monolithic fluid-structure interaction "
            "(Module type='fluid-FSI'). The fluid mesh deforms with "
            "the solid via Arbitrary Lagrangian-Eulerian (ALE). The "
            "FSI interface is implicit — material id pairs solid + "
            "fluid blocks and FEBio resolves the coupling each "
            "Newton iteration. Used for arterial-wall hemodynamics, "
            "cardiac chamber dynamics, valve modeling, and "
            "compliant-vessel benchmarks."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric monolithic FSI solver",
        "materials": {
            "fluid-FSI": {
                "fluid": "Nested fluid material (rho, k, viscous)",
                "solid": "Nested ALE-mesh-motion solid (typically "
                         "very soft — E~1 — to track interface)",
            },
        },
        "pitfalls": [
            (
                "[Syntax] `fluid-FSI` is both a MODULE name and a MATERIAL name, and the material only works inside the module. Naming the wrong module fails on the SOLVER first. "
                "WRONG: <Module type=\"fluid\"/> with a `fluid-FSI` material. "
                "RIGHT: <Module type=\"fluid-FSI\"/> with <solver type=\"fluid-FSI\">, and the material as <material id=\"1\" name=\"FSIBlock\" type=\"fluid-FSI\"> holding a <solid> and a <fluid> property. "
                "Signal: tag \"solver\" (line N) : `invalid value for attribute \"type\"` and `Reading file ...FAILED!` — the solver factory is module-scoped and is checked before the materials. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The ALE solid material inside the FSI "
                "fluid block should be VERY soft (E~1) — its only "
                "role is to define mesh motion. A stiff ALE solid "
                "resists fluid pressure and produces spurious "
                "interface tractions. Signal: a pressure-driven "
                "channel shows uniform velocity field instead of "
                "developing a Poiseuille profile because the ALE "
                "mesh refuses to follow the interface. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] The FSI interface is implicit — it exists only where the two element blocks SHARE nodes. Duplicate the nodes so the blocks are geometrically coincident but topologically disjoint and the model does not couple; FEBio reports NOTHING about the interface. "
                "WRONG: two element blocks meeting at coincident but DUPLICATED nodes. "
                "RIGHT: one <Nodes> block, with the elements on both sides of the interface referencing the SAME node ids there. "
                "Signal: no interface message of any kind — executed by duplicating every node and pointing half the elements at the copies, and the output contains no mention of shared nodes or of an interface. What you get is the generic `------- failed to converge at time : <t>` with `Number of time steps completed .... : 0` and exit 1, where the intact mesh completed all its steps. Detect it BEFORE running instead: compare the node count FEBio echoes against the count you intended — a duplicated interface shows up as roughly twice the nodes for the same geometry. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The FSI family is stiff, and on a USE_MKL=OFF build the binding constraint is not the ramp but the DEGREES OF FREEDOM: with the fluid velocity free the Newton loop fails at the first step regardless of load, ramp shape or step size. "
                "WRONG: reaching for a smoother load curve or a smaller step to rescue a first-step failure. Executed across driving amplitudes, two step sizes, four ALE stiffnesses spanning six decades, two meshes and ZERO load: every combination failed at step 1. Zero load failing is what rules out the ramp and the step size. "
                "RIGHT: constrain the fluid velocity on every node and remove the dilatation degree of freedom with <bc type=\"zero fluid dilatation\" node_set=\"all_nodes\"/>, then drive the problem through the SOLID with a prescribed displacement — which is what the shipped template does, and it completes all its steps on both meshes. "
                "Signal: `------- failed to converge at time : <t>` repeated, then `Number of time steps completed .... : 0` and exit 1; sometimes `N negative jacobians detected.` first, when the under-constrained velocity inverts an element. "
                "STILL UNVERIFIED: whether a freely flowing FSI problem converges on a build WITH MKL. Closing it needs FEBio rebuilt with -DUSE_MKL=ON for the `pardiso` factorisation, because every solver registered on this build either reports the matrix-format error or fails to converge. Retained rather than softened. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "fluid_fsi_3d_block": _fluid_fsi_3d_block,
}
