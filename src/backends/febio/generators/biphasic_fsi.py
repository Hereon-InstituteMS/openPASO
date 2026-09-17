"""FEBio biphasic-FSI generators and knowledge.

FEBio Module type: 'biphasic-FSI'. Combines a biphasic solid (porous
matrix + interstitial fluid) with an adjacent free fluid domain — the
biphasic side fluxes interstitial fluid across the interface into the
free-fluid side. Used for blood-tissue interaction (perfused
myocardium, atherosclerotic plaque, cartilage-synovial fluid), drug
elution from porous stents, and any scenario where fluid leaves /
enters a porous tissue and continues as free flow.
"""


def _biphasic_fsi_3d_block(params: dict) -> str:
    """Porous tissue block consolidated by a prescribed compression of
    its top face, using the `biphasic-FSI` MATERIAL inside the
    `fluid-FSI` MODULE.

    FEBio 4.12 has no `biphasic-FSI` module. The registry has
    `fluid-FSI.biphasic-FSI [FEMATERIAL_ID]`, i.e. biphasic-FSI is a
    material that lives in the fluid-FSI module. The earlier version of
    this template emitted <Module type="biphasic-FSI"/> and SEGFAULTED
    the solver with no diagnostic at all.

    The interstitial fluid velocity and the dilatation degree of freedom
    are fully constrained here. That is not decoration: on a USE_MKL=OFF
    build this is the configuration that converges. Freeing the fluid
    velocity leaves the momentum problem under-constrained and the
    Newton loop fails at the first step, at every load level tested
    including zero load, on every mesh tested. See the [Solver] pitfall.
    """
    n = max(1, int(params.get("n", 4)))
    W = float(params.get("width", 0.2))
    L = float(params.get("height", 1.0))
    E = float(params.get("E", 100.0))
    phi0 = float(params.get("phi0", 0.2))
    perm = float(params.get("permeability", 1.0e-3))
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
    <material id="1" name="Tissue" type="biphasic-FSI">
      <phi0>{phi0}</phi0>
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
      <permeability type="perm-const-iso">
        <perm>{perm}</perm>
      </permeability>
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
    <SolidDomain name="Part1" mat="Tissue"/>
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
      <node_data data="z;ef" delim="," file="biphasic_fsi_nodes.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "biphasic_fsi": {
        "description": (
            "Coupled biphasic-tissue + free-fluid FSI (Module "
            "type='biphasic-FSI'). The biphasic side resorbs / "
            "releases interstitial fluid into the free-fluid side "
            "across a shared interface. Canonical for blood-tissue "
            "perfusion (myocardium, plaque), cartilage-synovial "
            "fluid interaction, drug elution from porous stents, "
            "and bioreactor scaffolds with media perfusion."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Monolithic non-symmetric biphasic-FSI solver",
        "materials": {
            "biphasic-FSI": {
                "fluid": "Nested free-fluid material (rho, k, "
                         "viscous)",
                "solid": "Nested ALE-mesh-motion solid (very soft, "
                         "E~1) for the free-fluid side",
            },
            "biphasic (paired)": "On the porous-tissue side, use a "
                                 "standard 'biphasic' material with "
                                 "nested <solid> and "
                                 "<permeability>.",
        },
        "pitfalls": [
            (
                "[Syntax] `biphasic-FSI` is a MATERIAL, not a MODULE, and naming it as a module SEGFAULTS the solver with no diagnostic of any kind. FEBio 4.12 registers exactly ten modules and this is not one of them; the registry entry is `fluid-FSI.biphasic-FSI [FEMATERIAL_ID]`, i.e. a material inside the fluid-FSI module. "
                "WRONG: <Module type=\"biphasic-FSI\"/>. "
                "RIGHT: <Module type=\"fluid-FSI\"/> ... <material id=\"1\" name=\"Tissue\" type=\"biphasic-FSI\"><phi0>0.2</phi0><solid type=\"neo-Hookean\"><density>1.0</density><E>100.0</E><v>0.0</v></solid><fluid type=\"fluid\"><density>1.0</density><k>1e3</k><viscous type=\"Newtonian fluid\"><mu>0.01</mu></viscous></fluid><permeability type=\"perm-const-iso\"><perm>1e-3</perm></permeability></material>. "
                "Signal: stdout stops mid-line at `Reading file <name>.feb ...` with no `SUCCESS!`, no `FAILED!`, no ERROR box and no .log file, and the process is killed by SIGSEGV (signal 11; a shell reports exit status 139). A wrapper that only greps for the word error sees a completely silent failure. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d. The shipped template emitted the WRONG form until it was repaired in the same pass; the fixture is scripts/tier2_fixtures/febio/unknown_module_type_segfaults.)"
            ),
            (
                "[Input] The `biphasic-FSI` material takes THREE REQUIRED PROPERTIES, each named separately when it is missing — a <solid>, a <fluid> and a <permeability> — plus <phi0> (solid volume fraction), which is a PARAMETER with a default and is NOT reported when omitted. The <solid> and <fluid> slots are inherited from the plain `fluid-FSI` material; <phi0> and <permeability> are what biphasic-FSI adds on top. There is also an OPTIONAL <solvent_supply>. THIS CORRECTS AN EARLIER VERSION OF THIS ENTRY, which said the material needs FOUR things and each missing one is named separately. Executed: a deck with <phi0> deleted READS, RUNS and reaches normal termination with no message of any kind, so it is three named errors and one silent default. Whether the silent default changes the answer is NOT established: on the shipped template, where the interstitial velocity and the dilatation are fully constrained and the problem is driven through the solid, the logged fields agree to within solver noise at the shipped value, at zero and at a high value — the deck cannot see phi0 at all. Note also that this deck runs bicgstab on a non-symmetric matrix and is NOT bit-reproducible run to run, so compare its output with a tolerance, never with a checksum. "
                "WRONG: a `biphasic-FSI` material with only <solid> and <fluid>, i.e. the plain fluid-FSI shape. "
                "RIGHT: <Module type=\"fluid-FSI\"/> ... <material id=\"1\" name=\"Tissue\" type=\"biphasic-FSI\"><phi0>0.2</phi0><solid type=\"neo-Hookean\"><density>1.0</density><E>100.0</E><v>0.0</v></solid><fluid type=\"fluid\"><density>1.0</density><k>1e3</k><viscous type=\"Newtonian fluid\"><mu>0.01</mu></viscous></fluid><permeability type=\"perm-const-iso\"><perm>1e-3</perm></permeability></material>. "
                "Signal: `Component \"Tissue\" needs to have property \"permeability\" defined (line N)`, with the quoted name changing to whichever property is absent, then `Reading file ...FAILED!`. The material name in the message is the one you gave in name=, so it points straight at the offending block. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Solver] On a USE_MKL=OFF build the biphasic-FSI system converges only with the interstitial fluid velocity FULLY constrained and the dilatation degree of freedom removed. Leaving the fluid velocity free is not a tuning problem — it fails at the first step regardless of load. "
                "WRONG: relying on load reduction or step reduction to rescue it. Executed at driving amplitudes of 0.001, 0.01 and 0.05, at step sizes 0.005 and 0.02, at solid stiffness 1 and 100, on 2- and 4-element meshes, and AT ZERO LOAD: every single combination failed at the first step. Zero load failing is the decisive result — it rules out the load, the ramp and the step size, and points at the DOF constraints. "
                "RIGHT: add <bc type=\"zero fluid dilatation\" node_set=\"all_nodes\"/> and a `zero fluid velocity` BC with all three of <wx_dof>, <wy_dof>, <wz_dof> set on EVERY node, plus an <Initial> `initial fluid dilatation`. Then drive the problem through the SOLID instead — a prescribed displacement on one face — which is what the shipped template does and which converges on both meshes at compressions of 2, 5 and 10 percent. "
                "Signal: `------- failed to converge at time : <t>` repeated, `Number of time steps completed .... : 0` and exit 1. Sometimes `N negative jacobians detected.` instead, when the under-constrained velocity field inverts an element first. "
                "STILL UNVERIFIED: whether a genuinely flowing biphasic-FSI problem — free interstitial velocity, flow driven across the porous interface — converges on a build WITH MKL. Closing that needs FEBio rebuilt with -DUSE_MKL=ON so the `pardiso` factorisation is available, since every solver registered on this build either reports the matrix-format error or fails to converge here. The claim is retained, not softened. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Solver] The biphasic-FSI stiffness matrix is NON-SYMMETRIC, so both <symmetric_stiffness> and <linear_solver> must be set explicitly; the skyline default of a USE_MKL=OFF build cannot store the matrix. "
                "WRONG: <solver type=\"fluid-FSI\"/> on its own. "
                "RIGHT: <solver type=\"fluid-FSI\"><symmetric_stiffness>non-symmetric</symmetric_stiffness><linear_solver type=\"bicgstab\"/></solver>. "
                "Signal: the deck reads `...SUCCESS!`, then an ERROR box reading `The selected linear solver does not support the requested matrix format.` / `Please select a different linear solver.`, then `Number of time steps completed .... : 0` and exit 1. Executed across the registered solvers: `bicgstab` is the one that works on this build; `schur` additionally refuses to even construct, reporting `Component \"linear_solver\" needs to have property \"A_solver\" defined`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "biphasic_fsi_3d_block": _biphasic_fsi_3d_block,
}
