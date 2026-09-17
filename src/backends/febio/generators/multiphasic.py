"""FEBio multiphasic generators and knowledge.

FEBio Module type: 'multiphasic'. Biphasic poroelasticity + transport of
one or more solutes through the interstitial fluid. The canonical FEBio
module for charged-hydrated tissue mechanics (proteoglycan-rich
cartilage, electrolyte transport in soft tissue, ion diffusion through
porous scaffolds).
"""


def _multiphasic_3d_diffusion(params: dict) -> str:
    """Confined-compression-style multiphasic block: solid skeleton +
    interstitial fluid + one neutral solute (Na+ analogue). Prescribed
    concentration on top face, zero-concentration on bottom face.
    Steady-state advection-diffusion of the solute through the
    poroelastic solid.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.0)
    perm = params.get("permeability", 1.0e-3)
    diff = params.get("diffusivity", 1.0e-4)
    c_top = params.get("c_top", 1.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="multiphasic"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>10</time_steps>
    <step_size>1.0</step_size>
    <solver type="multiphasic">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Globals>
    <Constants>
      <T>298</T>
      <R>8.314e-6</R>
      <Fc>9.65e-5</Fc>
    </Constants>
    <Solutes>
      <solute id="1" name="Na">
        <charge_number>0</charge_number>
        <molar_mass>22.99</molar_mass>
        <density>1.0</density>
      </solute>
    </Solutes>
  </Globals>
  <Material>
    <material id="1" name="Material1" type="multiphasic">
      <phi0>0.2</phi0>
      <fixed_charge_density>0.0</fixed_charge_density>
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </solid>
      <permeability type="perm-const-iso">
        <perm>{perm}</perm>
      </permeability>
      <osmotic_coefficient type="osm-coef-const">
        <osmcoef>1.0</osmcoef>
      </osmotic_coefficient>
      <solute sol="1">
        <diffusivity type="diff-const-iso">
          <free_diff>{diff}</free_diff>
          <diff>{diff}</diff>
        </diffusivity>
        <solubility type="solub-const">
          <solub>1.0</solub>
        </solubility>
      </solute>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,1</node>
      <node id="6">1,0,1</node>
      <node id="7">1,1,1</node>
      <node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="Material1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="bottom">1,2,3,4</NodeSet>
    <NodeSet name="top">5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="c_bot" type="prescribed concentration" node_set="bottom">
      <dof>c1</dof>
      <value lc="1">0.0</value>
    </bc>
    <bc name="c_top" type="prescribed concentration" node_set="top">
      <dof>c1</dof>
      <value lc="1">{c_top}</value>
    </bc>
    <bc name="drain" type="zero fluid pressure" node_set="top"/>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="effective fluid pressure"/>
      <var type="effective solute concentration"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "multiphasic": {
        "description": (
            "Multiphasic FEBio module — biphasic poroelasticity + one or "
            "more solutes diffusing through the interstitial fluid. Used "
            "for charged-hydrated tissues (cartilage with proteoglycans), "
            "electrolyte transport, drug delivery through porous media, "
            "and growth factor diffusion in tissue scaffolds."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric Newton-Raphson (multiphasic solver)",
        "materials": {
            "multiphasic": {
                "phi0": "Solid volume fraction at reference",
                "fixed_charge_density": "Density of fixed charges in the solid (per unit reference volume). Use 0.0 for neutral, non-zero for proteoglycan-style charged tissue.",
                "solid": "Nested solid material (neo-Hookean, Holmes-Mow, ...)",
                "permeability": "Nested permeability law (perm-const-iso, perm-Holmes-Mow, ...)",
                "osmotic_coefficient": "Nested osmotic-coefficient law (osm-coef-const, ...)",
                "solute": "One or more nested <solute> blocks, each referencing a Globals/Solutes entry via sol=N.",
            },
        },
        "pitfalls": [
            (
                "[Input] <Globals><Solutes> is REQUIRED: it is what creates the concentration degrees of freedom c1, c2, ... Without it the solutes do not exist, and the failure surfaces on the first thing that REFERENCES one. "
                "WRONG: omitting the <Globals><Solutes> block while a <bc type=\"prescribed concentration\"> uses <dof>c1</dof>. "
                "RIGHT: <Globals><Solutes><solute id=\"1\" name=\"Na\"><charge_number>1</charge_number><molar_mass>22.99</molar_mass><density>1.0</density></solute></Solutes></Globals> BEFORE the <Material> section. "
                "Signal: tag \"dof\" (line N) : invalid value: c1 and `Reading file ...FAILED!` — the line number points at the BOUNDARY CONDITION, not at the missing Globals block, so the message sends you to the wrong section. The same message with a different suffix appears for a solute index that was never declared, e.g. invalid value: c9. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] A `biphasic` material does not accept <solute> children; solutes require the `multiphasic` material and the `multiphasic` module. "
                "WRONG: a <material type=\"biphasic\"> with a <solute> child. "
                "RIGHT: <Module type=\"multiphasic\"/> with <material type=\"multiphasic\"> holding <phi0>, <solid>, <permeability>, <osmotic_coefficient> and one <solute> per species. "
                "Signal: tag \"solute\" (line N) : `unrecognized tag` and `Reading file ...FAILED!`. It is reported as an unknown TAG rather than as an unsupported feature, so nothing in the message mentions multiphasic. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] For charged tissue, set "
                "fixed_charge_density > 0 AND charge_number != 0 on "
                "the solute. Otherwise the Donnan-equilibrium / "
                "electrostatic effects vanish and the tissue behaves "
                "as a neutral biphasic. Signal: prescribed-"
                "concentration step shows no swelling at the loaded "
                "boundary — proteoglycan response (expected ~10-30% "
                "swell strain) is absent. (Audit 2026-06-02.)"
            ),
            (
                "[Input] A `prescribed concentration` BC selects its species with a <dof> child naming the concentration DOF — c1 for the first declared solute, c2 for the second — NOT with a <sol> child and not by position. "
                "WRONG: <bc type=\"prescribed concentration\" node_set=\"top\"><sol>1</sol><value lc=\"1\">1.0</value></bc>. "
                "RIGHT: <bc name=\"c_top\" type=\"prescribed concentration\" node_set=\"top\"><dof>c1</dof><value lc=\"1\">1.0</value></bc>. "
                "Signal: tag \"sol\" (line N) : `unrecognized tag`, and for an undeclared species tag \"dof\" (line N) : invalid value: c9. Both are parse-time and both name the offending tag. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "multiphasic_3d_diffusion": _multiphasic_3d_diffusion,
}
