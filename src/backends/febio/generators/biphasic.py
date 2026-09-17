"""FEBio biphasic generators and knowledge.

FEBio Module type: 'biphasic'. Solid skeleton + interstitial fluid with
explicit permeability. The hallmark FEBio module for soft-tissue mechanics
(cartilage, intervertebral disc, intervertebral disc, etc.).
"""


def _biphasic_3d_confined(params: dict) -> str:
    """Confined-compression biphasic test — solid skeleton with
    interstitial fluid, isotropic permeability.

    Top face drained (zero pore pressure); bottom face fixed.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.0)
    perm = params.get("permeability", 1.0e-3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="biphasic"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>10</time_steps>
    <step_size>1.0</step_size>
    <solver type="biphasic">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="biphasic">
      <phi0>0.2</phi0>
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </solid>
      <permeability type="perm-const-iso">
        <perm>{perm}</perm>
      </permeability>
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
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
    <bc name="drain" type="zero fluid pressure" node_set="load_top"/>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
</febio_spec>
'''


KNOWLEDGE = {
    "biphasic": {
        "description": "Biphasic poroelasticity — solid skeleton + interstitial fluid (FEBio Module type='biphasic')",
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric Newton-Raphson (biphasic solver)",
        "materials": {
            "biphasic": {
                "phi0": "Solid volume fraction at reference",
                "solid": "Nested solid material (e.g. neo-Hookean)",
                "permeability": "Nested permeability model "
                                "(perm-const-iso, perm-Holmes-Mow, perm-exp-iso, etc.)",
            },
        },
        "pitfalls": [
            "[Syntax] A biphasic model needs <Module type=\"biphasic\"/>, and with the module left as solid the first failure is the <analysis> WORD, not the material. Each module installs its own analysis vocabulary: solid has STATIC and DYNAMIC, biphasic has STEADY-STATE and TRANSIENT. "
            "WRONG: <Module type=\"solid\"/> with <analysis>STEADY-STATE</analysis>. "
            "RIGHT: <Module type=\"biphasic\"/> with <analysis>STEADY-STATE</analysis> and <solver type=\"biphasic\"><symmetric_stiffness>non-symmetric</symmetric_stiffness><linear_solver type=\"bicgstab\"/></solver>. "
            "Signal: tag \"analysis\" (line N) : invalid value: STEADY-STATE and `Reading file ...FAILED!`. The message names the analysis word, so it reads as if that word were wrong when the MODULE is what is wrong — check the module first. Note that an unregistered MODULE name is far worse than either: it SEGFAULTS with no message at all. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)",
            "[Syntax] biphasic material requires NESTED <solid> and "
            "<permeability> elements with their own type attribute. "
            "Flat parameter lists (E, v directly inside biphasic) "
            "are not valid. "
            "Signal: tag \"<param>\" (line N) : `unrecognized tag` and `Reading file ...FAILED!`, where <param> echoes whatever flat parameter you wrote. Executed with a <nu> inside `isotropic elastic` and a <Y0> inside `von-Mises plasticity`: both are named back verbatim, and a missing nested PROPERTY reports differently again, as `Component \"<name>\" needs to have property \"permeability\" defined (line N)`. So a flat parameter and a missing property are distinguishable from the message alone. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)",
            "[Numerical] Pore-pressure boundary conditions are their "
            "own BC types, separate from displacement BCs. Both names "
            "are registered in the biphasic module and were read off "
            "the binary: `biphasic.zero fluid pressure [FEBC_ID]` and "
            "`biphasic.prescribed fluid pressure [FEBC_ID]`. "
            "BUT DO NOT READ A ZERO PRESSURE FIELD AS A MISSING "
            "DRAINAGE BC — check the <analysis> word first, because "
            "that is the far more common cause and it is the one the "
            "shipped template hits. EXECUTED 2026-08-05 on "
            "biphasic_3d_confined, which has a `zero fluid pressure` "
            "BC on its loaded face: under "
            "<analysis>STEADY-STATE</analysis> the element pressure is "
            "EXACTLY 0 at every one of the ten steps, because "
            "steady-state sets div w = 0 and there is no transient to "
            "solve at all. Change the single word to "
            "<analysis>TRANSIENT</analysis> and the pressure comes "
            "alive, decaying to 1.6e-8 by t_end and converging on the "
            "same solid stress sz = -105.5556. So the fluid phase and "
            "the permeability are live; the analysis word had switched "
            "them off. "
            "WRONG: hunting for a missing drainage BC when the "
            "pressure field is flat. "
            "RIGHT: run the same deck once as STEADY-STATE and once as "
            "TRANSIENT and compare — if only the transient run has "
            "pressure, the BCs were never the problem. "
            "Signal: none from the solver; both runs exit 0 with "
            "`N O R M A L   T E R M I N A T I O N`. Log "
            "<element_data data=\"p\"/> and require it non-zero. "
            "STILL UNVERIFIED: whether OMITTING the drainage BC "
            "entirely also produces a flat pressure field rather than "
            "a diagnostic. Closing it needs the same deck with the "
            "`zero fluid pressure` BC deleted, under TRANSIENT so a "
            "pressure field exists to stagnate. Retained rather than "
            "softened. "
            "(Analysis-word half executed 2026-08-05, FEBio "
            "4.12.0.86045466d; BC names read from the FEBC_ID "
            "registry.)",
            "[Solver] Biphasic (and multiphasic / fluid / FSI) systems are "
            "NON-SYMMETRIC. The linear solver must support a non-symmetric "
            "matrix format. FEBio's default 'skyline' solver — the fallback "
            "when FEBio is built without Intel MKL — is symmetric-only. "
            "Signal: the deck READS cleanly (`Reading file ...SUCCESS!`) and "
            "then aborts with `The selected linear solver does not support "
            "the requested matrix format.`, `Number of time steps completed "
            "... : 0` and `E R R O R   T E R M I N A T I O N`, exit status 1 "
            "— note a mesh-only .xplt is still written (size is "
            "deck-specific: 1676 bytes on a one-hex8 confined-compression "
            "deck here, 1377 on the shipped biphasic_3d_confined template; "
            "corrected 2026-08-03, this entry previously quoted a fixed "
            "1353 bytes measured on a different, linear-elastic deck). "
            "Note the message WRAPS at 71 columns inside the ERROR box, so "
            "grep for the fragment `does not support the requested`, not the "
            "whole sentence. LIVE-VERIFIED 2026-08-03 on a USE_MKL=OFF "
            "source build of FEBio 4.12.0, sweeping all 20 registered "
            "FELINEARSOLVER_ID factories: only TWO of them run at all, and "
            "one of those is a trap. skyline, fgmres, cg, bipn, "
            "superlu_mt, accelerate, diagonal, hypre_gmres, hypre_pcg_amg, "
            "boomeramg, ichol and pardiso-project emit the message "
            "above; `LU` fails with `Linear solver failed to find solution`; "
            "`ilu0` and `ilut` with `Fatal error in factorization of "
            "stiffness matrix`; `block` with `An error occurred during "
            "preprocessing of linear solver`. TWO of the twenty never reach "
            "the matrix at all, because they are composite solvers that have "
            "to be given their sub-solvers: `strategy` is refused at parse "
            "with `Component \"linear_solver\" needs to have property "
            "\"solver1\" defined`, and `schur` with the same message naming "
            "\"A_solver\". An earlier version of this entry listed `strategy` "
            "and `schur` among the solvers that emit the matrix-format "
            "message; re-executed, neither does. That leaves `bicgstab` (the "
            "real answer) and `test` (a null solver — see the next pitfall). "
            "CORRECTIONS to the previous entry: `pardiso` is not even a valid "
            "type string without MKL (tag linear_solver (line N) : `"
            "invalid value for attribute \"type\"`) — the registered "
            "MKL-free spelling is `pardiso-project`; and `accelerate`, "
            "previously recommended here, does NOT accept the unsymmetric "
            "format on this build. Use <solver type=\"biphasic\">"
            "<linear_solver type=\"bicgstab\"/></solver> — but read the "
            "robustness caveat below before trusting it. "
            "FURTHER CORRECTION 2026-08-03: the earlier claim 'that leaves "
            "bicgstab (the real answer)' is too strong. Re-running the same "
            "20-solver sweep on "
            "an independently written one-hex8 confined-compression deck "
            "(prescribed uz = -0.1 over 10 x 0.1 transient steps), bicgstab "
            "ALSO failed, with `Linear solver failed to find solution. "
            "Aborting run.` at the first step — the same message LU gives. "
            "The other 19 outcomes reproduced exactly. bicgstab is the only "
            "solver that CAN run an unsymmetric system on a USE_MKL=OFF "
            "build; it is not one that reliably does.",
            "[Solver] `bicgstab` is registered with an OPTIONAL `pc_left` "
            "preconditioner property that defaults to NONE, so it is an "
            "unpreconditioned Krylov solve. It is a small-problem "
            "workaround, not a replacement for a direct unsymmetric solver: "
            "live-verified 2026-08-03, the same confined-compression "
            "biphasic deck converged on a single hex8 and FAILED on a 2x2x2 "
            "mesh. Signal: `Linear solver failed to find solution. Aborting "
            "run.` with `E R R O R   T E R M I N A T I O N` and exit 1 after "
            "the deck read SUCCESS — i.e. the same message `LU` gives, and "
            "easy to misread as a modelling problem when it is a solver "
            "problem. If real biphasic / multiphasic / fluid / FSI work is "
            "needed, rebuild FEBio with -DUSE_MKL=ON so the `pardiso` "
            "factorisation exists.",
            "[Solver] NEVER write <linear_solver type=\"test\"/>. It is a "
            "registered FELINEARSOLVER_ID factory, so it passes every "
            "validation, and NumCore/TestSolver.cpp BackSolve() sets the "
            "whole solution vector to 0 and returns true. FEBio then takes "
            "zero Newton increments, declares convergence, completes every "
            "requested time step and finishes cleanly. Signal: this is the "
            "one verified case on FEBio 4.12 where `N O R M A L   T E R M I "
            "N A T I O N`, exit status 0 AND a non-zero `Number of time "
            "steps completed` are all satisfied by a run that solved "
            "nothing — the standard acceptance checks cannot catch it, so "
            "it has to be excluded when the deck is generated. On a "
            "one-element biphasic deck it returns exactly what the "
            "prescribed boundary conditions impose by themselves, with every "
            "free degree of freedom left at zero — so the fluid pressure, "
            "the entire point of a biphasic run, comes back identically "
            "zero at every node. The size of the resulting error is "
            "deck-dependent and can be anything from enormous to "
            "invisible, so never rely on the answer merely LOOKING wrong. "
            "Detect it structurally instead: grep the deck for "
            "`type=\"test\"`, or assert that a field which must be "
            "non-zero is in fact non-zero. "
            "advanced to -3.60750105073 (44.5% off). Anything with interior "
            "degrees of freedom gets the zero field outright. "
            "(Live-verified 2026-08-03, FEBio 4.12.0; fixture "
            "scripts/tier2_fixtures/febio/null_test_linear_solver_reports_success.)",
            "[Solver] The <analysis> vocabulary is installed by the ACTIVE "
            "MODULE, so it differs from the solid module's. In `biphasic` "
            "(and `solute` / `multiphasic`) the legal words are "
            "`STEADY-STATE` and `TRANSIENT`; the solid module's `STATIC` / "
            "`DYNAMIC` are rejected, and the fluid-family modules use a "
            "third pairing (`STEADY-STATE` / `DYNAMIC`). Signal: tag "
            "\"analysis\" (line N) : invalid value: STATIC with `Reading "
            "file ...FAILED!` — a clean parse error, not a silent fallback "
            "to ordinal 0. Enum words are matched case-insensitively and the "
            "raw ordinal is also accepted, so `<analysis>1</analysis>` "
            "quietly selects TRANSIENT here and DYNAMIC in a solid deck; "
            "prefer the spelled-out word. (Live-verified 2026-08-03, FEBio "
            "4.12.0; enum tables from FEBiphasicAnalysis.cpp / "
            "FESolidAnalysis.cpp setEnums().)",
            "[Solver] <symmetric_stiffness>symmetric</symmetric_stiffness> "
            "is a legitimate workaround when only skyline is available: it "
            "changes the NEWTON TANGENT only, not the residual, so the "
            "converged answer is unchanged to solver tolerance. Verified on "
            "a confined-compression deck with strain-dependent "
            "perm-Holmes-Mow permeability (where the biphasic tangent really "
            "is unsymmetric): symmetric+skyline and non-symmetric+bicgstab "
            "agreed on sz to 6 significant digits over 4 steps "
            "(-126.160499 vs -126.160529). Signal: the cost is convergence "
            "robustness, not accuracy — expect more `------- failed to "
            "converge at time :` lines and more BFGS reformations, not a "
            "wrong stress. Do not treat a symmetric-stiffness run as "
            "automatically invalid. (Live-verified 2026-08-03, FEBio "
            "4.12.0.)",
            "[Syntax] FEBio 4.x <material> requires a 'name' attribute (in "
            "addition to id), and <MeshDomains> reference the material by that "
            "name (mat=\"<name>\"). Signal: omitting the name attribute fails at "
            "parse with 'tag \"material\" ... missing attribute \"name\"' "
            "(required in FEBioXML/FEBioMaterialSection.cpp). NodeSets declared "
            "in <Mesh> take a comma-separated node-id list "
            "(<NodeSet name=\"x\">1,2,3,4</NodeSet>), NOT <n id=.../> child "
            "elements, which FEBio 4.x rejects with 'invalid value'.",
        ],
    },
}


GENERATORS = {
    "biphasic_3d_confined": _biphasic_3d_confined,
}
