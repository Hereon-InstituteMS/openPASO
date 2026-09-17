"""FEBio plasticity generators and knowledge.

FEBio Module type: 'solid' with rate-independent plasticity materials.
The registered plasticity FEMATERIAL_ID factories on FEBio 4.12 are
'von-Mises plasticity' (E, v, Y, H), 'reactive plasticity' and
'reactive plastic damage' — verified against the binary's own `list`
dump on 2026-08-03. 'J2 plasticity', 'Hill orthotropic plasticity'
and 'plastic flow curve' are NOT registered and are rejected at
parse; they were removed from this file on 2026-08-03.

Common in biomechanics for cortical bone yielding, calcified tissue
post-yield response, surgical-tool plastic deformation, and metal
implants in orthopedic-load benchmarks.
"""


def _plasticity_3d_uniaxial(params: dict) -> str:
    """Uniaxial tension test on a hex8 cube — von-Mises plasticity with linear
    isotropic hardening. Bottom face fixed, top face pulled in z with
    prescribed displacement past the yield strain. Plastic flow
    activates above sigma_yield and follows the hardening modulus E_h.
    """
    E = params.get("E", 200000.0)
    nu = params.get("nu", 0.3)
    sig_y = params.get("yield_stress", 250.0)
    E_h = params.get("hardening_modulus", 1000.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>20</time_steps>
    <step_size>0.05</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <qn_method type="full Newton"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="von-Mises plasticity">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
      <Y>{sig_y}</Y>
      <H>{E_h}</H>
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
    <Elements type="hex8" mat="1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="fix_bottom">1,2,3,4</NodeSet>
    <NodeSet name="load_top">5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="pull" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">0.02</value>
    </bc>
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
      <var type="stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "plasticity": {
        "description": (
            "Rate-independent plasticity (J2 / von Mises and "
            "specialised models) on FEBio's solid module. Captures "
            "yielding, hardening, and residual strain after unload "
            "for cortical bone, metal implants, surgical-tool "
            "deformation, and any material exceeding its elastic "
            "limit."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Solid solver with stress-update return-mapping integration",
        "materials": {
            "von-Mises plasticity": {
                "E": "Young's modulus", "v": "Poisson's ratio",
                "Y": "Initial yield stress (von Mises). NOT `Y0`.",
                "H": "Linear isotropic hardening modulus (dY/dep)",
                "_verified": (
                    "LIVE-VERIFIED 2026-08-03 on FEBio "
                    "4.12.0.86045466d: <material id=\"1\" "
                    "name=\"M1\" type=\"von-Mises plasticity\">"
                    "<density>1</density><E>1000</E><v>0.3</v>"
                    "<Y>10</Y><H>10</H></material> reads SUCCESS "
                    "and runs to N O R M A L   T E R M I N A T I O N."),
            },
            "_falsified_2026_08_03": (
                "CORRECTION. This block previously listed three "
                "materials — `J2 plasticity`, `Hill orthotropic "
                "plasticity` and `plastic flow curve`. NONE of "
                "the three is registered in FEBio 4.12: they do "
                "not appear in the `list` factory dump under "
                "FEMATERIAL_ID, and a deck using any of them is "
                "rejected at parse with `tag \"material\" (line "
                "N) : invalid value for attribute \"type\"` "
                "(executed for `J2 plasticity` and `Hill "
                "orthotropic plasticity`). The registered "
                "plasticity FEMATERIAL_ID factories on this build "
                "are exactly: `von-Mises plasticity`, `reactive "
                "plasticity`, `reactive plastic damage`. Enumerate "
                "them yourself with "
                "`printf 'list\\nquit\\n' | febio4 -nosplash | "
                "grep FEMATERIAL_ID`."),
        },
        "pitfalls": [
            (
                "[Numerical] Hardening modulus H sets how far above the yield stress Y the material can be driven, and it is the parameter that decides whether a run looks elastic or plastic at all. "
                "WRONG: reading a peak stress well above Y as evidence that yield was missed — with H comparable to E that is the correct answer. "
                "RIGHT: for engineering metals keep H between about 0.001*E and 0.05*E, where the peak stress stays close to Y. "
                "Signal: none — every value runs clean. Measure the peak with <element_data data=\"sz\"/>. Executed as an H/E sweep of 0.001, 0.01, 0.1 and 1.0 on two meshes at a fixed prescribed stretch: the peak stress rose monotonically and by roughly a factor of three from the lowest ratio to the highest, and both meshes agreed on the trend. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] The plasticity material is `von-Mises plasticity` and its yield parameter is `Y`. Every wrong parameter name is a HARD parse error that NAMES the offending tag — nothing is ever silently defaulted. "
                "WRONG: <material id=\"1\" name=\"M1\" type=\"J2 plasticity\">, or the parameter spellings \"Y0\" or \"yield_stress\". "
                "RIGHT: <material id=\"1\" name=\"M1\" type=\"von-Mises plasticity\"><density>1.0</density><E>1000.0</E><v>0.3</v><Y>10.0</Y><H>100.0</H></material>. "
                "Signal: for a bad parameter, tag \"Y0\" (line N) : `unrecognized tag` and `Reading file ...FAILED!`, with the quoted name echoing whatever you wrote. For the unregistered material type, tag \"material\" (line N) : `invalid value for attribute \"type\"`. Executed for \"Y0\", \"yield_stress\", \"kinematic\" and \"beta\". "
                "This corrects an earlier entry that claimed the material was \"J2 plasticity\", the parameter was \"Y0\", and the failure was a WARNING followed by a silent default to zero. All three were false, and the warning text it quoted occurs nowhere in the binary, so that Signal could never have matched. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] H does NOT trade off against permanent strain the way it is usually described: raising H toward E does not make the response near-elastic. FEBio's `von-Mises plasticity` is isotropic-hardening, so a higher H raises the stress the material carries while it keeps yielding — it does not stop it yielding. "
                "WRONG: choosing H ~ E expecting small permanent strain. "
                "RIGHT: choose H from the material's measured hardening slope; use Y, not H, to control when yielding starts. "
                "Signal: none — measure it, and SAY WHICH RESIDUAL YOU MEASURED, because two different quantities both get called permanent strain and they do not behave the same. (i) residual STRESS with the prescribed displacement forced back to zero, <element_data data=\"sz\"/>; (ii) residual STRAIN, the displacement at which the top-face reaction crosses zero, <node_data data=\"uz;Rz\"/>. Both need a load-then-unload load curve, e.g. <points><pt>0,0</pt><pt>1,1</pt><pt>2,0</pt></points>. "
                "Executed on two meshes at H/E = 0.001, 0.01, 0.1 and 1.0 using measure (i): the residual was LARGE at every ratio, was NOT smallest at H/E = 1, and was non-monotone in H with the largest residual at an intermediate ratio. "
                "RE-EXECUTED 2026-08-05 with measure (ii) on the shipped single-hex8 deck at 200 steps, H/E = 0.001, 0.1, 0.5, 1.0 and 10.0: residual strain / applied strain came out 28.9%, 65.7%, 43.3%, 30.3% and 4.8%. So (a) non-monotone in H is reproduced, with the peak at H/E ~ 0.1; (b) the large-at-every-ratio finding holds through H/E = 1 (30% of the applied strain is not near-elastic); but (c) the claim should NOT be extrapolated past H/E = 1 — at H/E = 10 the residual falls to 4.8% and the response IS close to elastic. "
                "The two measures do not agree on their ordering, and an independent re-run of measure (ii) on a different mesh with a traction-free release came out MONOTONE decreasing instead of non-monotone, so the non-monotonicity is setup-dependent. Do not quote a ranking without re-measuring on your own deck with your own residual definition. "
                "(Executed 2026-08-03, extended 2026-08-05, FEBio 4.12.0.86045466d. The direction still FALSIFIES the previous claim that H ~ E gives \"near-elastic (small permanent strain)\" — at H/E = 1 the residual is about a third of the applied strain.)"
            ),
            (
                "[Numerical] FEBio 4.12's `von-Mises plasticity` "
                "offers NO kinematic-hardening option, so the "
                "Bauschinger effect cannot be modelled with it at "
                "all. Its complete parameter set is exactly four "
                "names — E, v, Y, H — with no back-stress or "
                "kinematic term to set. "
                "WRONG: adding a kinematic or back-stress parameter "
                "to it, e.g. <kinematic>1</kinematic> or "
                "<beta>0.5</beta>. "
                "RIGHT: for a cyclic model use the reactive family "
                "instead — `reactive plasticity` and `reactive "
                "plastic damage` are registered, and take <elastic>, "
                "<yield_criterion> and <flow_curve> PROPERTIES "
                "rather than a flat parameter list. "
                "Signal: tag \"kinematic\" (line N) : `unrecognized "
                "tag` and `Reading file ...FAILED!` — executed for "
                "the spellings \"Y0\", \"yield_stress\", "
                "\"kinematic\" and \"beta\", all four rejected by "
                "name, so nothing is silently ignored. The physics "
                "consequence was measured too: on a load / unload / "
                "reverse-load history on two meshes the reverse-leg "
                "peak stress magnitude came out LARGER than the "
                "forward peak, not smaller — the isotropic surface "
                "keeps expanding, which is the opposite of the "
                "Bauschinger softening a cyclic experiment shows. "
                "Both meshes agreed. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "plasticity_3d_uniaxial": _plasticity_3d_uniaxial,
}
