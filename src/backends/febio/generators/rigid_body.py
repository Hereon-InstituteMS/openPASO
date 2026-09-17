"""FEBio rigid-body and rigid-vs-deformable contact generators.

FEBio Module type: 'solid' with material type 'rigid body'. A rigid
body is a single element-block constrained to translate / rotate as a
single unit. Used for impactors, indenters, fixtures, articulating
joints (combined with rigid connectors), and contact-prescribed
boundary conditions.

Two templates are provided:

* ``rigid_body_3d_pushdown`` — prescribed rigid-body motion driving a
  deformable block through SHARED nodes (no contact search).
* ``rigid_contact_3d_indentation`` — a rigid flat punch pressed into a
  separate, non-shared deformable block through a real ``<Contact>``
  ``sliding-elastic`` interface. This is the template to copy for any
  indentation / impact / contact-mechanics problem.
"""


def _hex_block(x0, x1, y0, y1, z0, z1, nx, ny, nz, nid0=1, eid0=1):
    """Structured hex8 block. Returns (nodes, elems, nid) where
    ``nid(i, j, k)`` gives the node id at lattice position (i, j, k)."""
    def nid(i, j, k):
        return nid0 + i + (nx + 1) * j + (nx + 1) * (ny + 1) * k
    nodes = [(nid(i, j, k),
              x0 + (x1 - x0) * i / nx,
              y0 + (y1 - y0) * j / ny,
              z0 + (z1 - z0) * k / nz)
             for k in range(nz + 1)
             for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    e = eid0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elems.append((e, [nid(i, j, k), nid(i + 1, j, k),
                                  nid(i + 1, j + 1, k), nid(i, j + 1, k),
                                  nid(i, j, k + 1), nid(i + 1, j, k + 1),
                                  nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]))
                e += 1
    return nodes, elems, nid


def _xml_nodes(nodes, indent="      "):
    return "\n".join(f'{indent}<node id="{i}">{x:.10g},{y:.10g},{z:.10g}</node>'
                     for i, x, y, z in nodes)


def _xml_elems(elems, indent="      "):
    return "\n".join(f'{indent}<elem id="{i}">' + ",".join(str(n) for n in c) + "</elem>"
                     for i, c in elems)


def _xml_quads(facets, indent="      "):
    return "\n".join(f'{indent}<quad4 id="{k + 1}">' + ",".join(str(n) for n in f) + "</quad4>"
                     for k, f in enumerate(facets))


def _rigid_body_3d_pushdown(params: dict) -> str:
    """Rigid impactor (top block) pushes down on a deformable block
    (bottom) via prescribed rigid-body translation. The two blocks SHARE
    their interface nodes, so no contact search is involved — the rigid
    body simply drags the deformable surface with it.

    Use this when the contact is permanently bonded / the impactor never
    separates. For a real separable interface use
    ``rigid_contact_3d_indentation`` instead.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    dz = params.get("displacement", -0.1)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Deformable" type="isotropic elastic">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
    </material>
    <material id="2" name="Impactor" type="rigid body">
      <density>10.0</density>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,0.5</node>
      <node id="6">1,0,0.5</node>
      <node id="7">1,1,0.5</node>
      <node id="8">0,1,0.5</node>
      <node id="9">0,0,1</node>
      <node id="10">1,0,1</node>
      <node id="11">1,1,1</node>
      <node id="12">0,1,1</node>
    </Nodes>
    <Elements type="hex8" name="DeformablePart">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <Elements type="hex8" name="ImpactorPart">
      <elem id="2">5,6,7,8,9,10,11,12</elem>
    </Elements>
    <NodeSet name="base">1,2,3,4</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="DeformablePart" mat="Deformable"/>
    <SolidDomain name="ImpactorPart" mat="Impactor"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="base">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
  <Rigid>
    <rigid_bc name="impactor_lock" type="rigid_fixed">
      <rb>2</rb>
      <Rx_dof>1</Rx_dof>
      <Ry_dof>1</Ry_dof>
      <Ru_dof>1</Ru_dof>
      <Rv_dof>1</Rv_dof>
      <Rw_dof>1</Rw_dof>
    </rigid_bc>
    <rigid_bc name="impactor_push" type="rigid_displacement">
      <rb>2</rb>
      <dof>z</dof>
      <value lc="1">{dz}</value>
    </rigid_bc>
  </Rigid>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile>
      <node_data data="x;y;z" file="pos.txt"/>
      <element_data data="sz;J" file="el.txt"/>
    </logfile>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="rigid position"/>
    </plotfile>
  </Output>
</febio_spec>
'''


def _rigid_contact_3d_indentation(params: dict) -> str:
    """Rigid flat punch pressed into a deformable block across a real
    ``sliding-elastic`` contact interface.

    The two bodies have SEPARATE, non-matching node sets and start with a
    gap, so the interface is genuinely resolved by the contact search.
    The punch footprint spans ``4 x 4`` block elements, which gives a
    resolved contact patch (flat under the punch, rising to the free
    surface outside it).

    params: E, nu (block), nx/ny/nz (block refinement), inx/iny (punch
    refinement), gap, indentation, penalty, time_steps.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    nx = int(params.get("nx", 8))
    ny = int(params.get("ny", 8))
    nz = int(params.get("nz", 4))
    inx = int(params.get("inx", 4))
    iny = int(params.get("iny", 4))
    gap = params.get("gap", 0.05)
    depth = params.get("indentation", 0.10)
    penalty = params.get("penalty", 10.0)
    nsteps = int(params.get("time_steps", 10))

    z_top = 0.5
    z_punch0 = z_top + gap
    z_punch1 = z_punch0 + 0.2
    bn, be, bid = _hex_block(0, 1, 0, 1, 0, z_top, nx, ny, nz, nid0=1, eid0=1)
    pn, pe, pid = _hex_block(0.25, 0.75, 0.25, 0.75, z_punch0, z_punch1,
                             inx, iny, 1, nid0=len(bn) + 1, eid0=len(be) + 1)

    # block top surface, outward normal +z
    top = [[bid(i, j, nz), bid(i + 1, j, nz), bid(i + 1, j + 1, nz), bid(i, j + 1, nz)]
           for j in range(ny) for i in range(nx)]
    # punch bottom surface, outward normal -z
    bot = [[pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)]
           for j in range(iny) for i in range(inx)]
    base = ",".join(str(bid(i, j, 0)) for j in range(ny + 1) for i in range(nx + 1))

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
    <material id="1" name="Block" type="neo-Hookean">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
    </material>
    <material id="2" name="Punch" type="rigid body">
      <density>1.0</density>
    </material>
  </Material>
  <Mesh>
    <Nodes name="AllNodes">
{_xml_nodes(bn)}
{_xml_nodes(pn)}
    </Nodes>
    <Elements type="hex8" name="BlockPart">
{_xml_elems(be)}
    </Elements>
    <Elements type="hex8" name="PunchPart">
{_xml_elems(pe)}
    </Elements>
    <NodeSet name="base">{base}</NodeSet>
    <Surface name="BlockTop">
{_xml_quads(top)}
    </Surface>
    <Surface name="PunchBottom">
{_xml_quads(bot)}
    </Surface>
    <SurfacePair name="PunchOnBlock">
      <primary>BlockTop</primary>
      <secondary>PunchBottom</secondary>
    </SurfacePair>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="BlockPart" mat="Block"/>
    <SolidDomain name="PunchPart" mat="Punch"/>
  </MeshDomains>
  <Boundary>
    <bc name="fixbase" type="zero displacement" node_set="base">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
  <Rigid>
    <rigid_bc name="punch_lock" type="rigid_fixed">
      <rb>2</rb>
      <Rx_dof>1</Rx_dof>
      <Ry_dof>1</Ry_dof>
      <Ru_dof>1</Ru_dof>
      <Rv_dof>1</Rv_dof>
      <Rw_dof>1</Rw_dof>
    </rigid_bc>
    <rigid_bc name="punch_push" type="rigid_displacement">
      <rb>2</rb>
      <dof>z</dof>
      <value lc="1">{-(gap + depth):.10g}</value>
    </rigid_bc>
  </Rigid>
  <Contact>
    <contact type="sliding-elastic" surface_pair="PunchOnBlock">
      <laugon>PENALTY</laugon>
      <penalty>{penalty}</penalty>
      <auto_penalty>1</auto_penalty>
      <two_pass>1</two_pass>
      <tolerance>0.1</tolerance>
      <search_radius>1.0</search_radius>
    </contact>
  </Contact>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile>
      <node_data data="x;y;z" file="pos.txt"/>
      <element_data data="sz;J" file="el.txt"/>
    </logfile>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="contact pressure"/>
      <var type="contact gap"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "rigid_body": {
        "description": (
            "FEBio rigid-body material and rigid-vs-deformable contact. "
            "A 'rigid body' material is a single material id whose "
            "elements all translate and rotate as one unit (6 DOF total). "
            "Used for impactors, indenters, fixtures, articulating "
            "joints and any body whose internal deformation is "
            "irrelevant. Lives inside Module type='solid'. Rigid-body "
            "boundary conditions use a dedicated <Rigid> section, NOT "
            "<Boundary>. Contact between a rigid body and a deformable "
            "body needs a <Contact> section referencing a <SurfacePair> "
            "declared inside <Mesh>."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver, augmented with rigid-body DOFs (6 per rigid body)",
        "templates": [
            "rigid_body_3d_pushdown — prescribed rigid motion, shared "
            "interface nodes, no contact search.",
            "rigid_contact_3d_indentation — rigid flat punch on a "
            "separate deformable block through a sliding-elastic "
            "<Contact> interface; the one to copy for indentation, "
            "impact and contact-mechanics problems.",
        ],
        "materials": {
            "rigid body": {
                "density": "Mass density. OPTIONAL, defaults to 1.0.",
                "center_of_mass": "OPTIONAL. Omit it and FEBio computes "
                                  "the centre of mass from the element "
                                  "mass distribution. Supplying it "
                                  "OVERRIDES the computed value.",
                "override_com": "OPTIONAL boolean. Not needed to make "
                                "center_of_mass take effect.",
                "E": "OPTIONAL. Only used where a rigid body is tied to "
                     "deformable elements.",
                "v": "OPTIONAL Poisson's ratio (spelled 'v', not 'nu').",
            },
        },
        "contact_types": (
            "FESURFACEINTERACTION_ID types proven to run in the solid "
            "module on a rigid-vs-deformable pair: 'sliding-elastic', "
            "'sliding-facet-on-facet', 'sliding-node-on-facet'. Also "
            "registered: 'tied-elastic', 'tied-facet-on-facet', "
            "'tied-node-on-facet', 'mortar-sliding', 'mortar-tied', "
            "'sticky', 'contact potential', 'periodic boundary', "
            "'surface constraint'. The FEBio 2.x names "
            "'facet-to-facet sliding' and 'sliding_with_gaps' are NOT "
            "accepted by the 4.0 reader."
        ),
        "pitfalls": [
            (
                "[Syntax] Rigid-body BCs live in a top-level <Rigid> "
                "section, never in <Boundary>, and they name the rigid "
                "MATERIAL id with <rb>N</rb> — never a node set. "
                "WRONG: <Boundary><bc name=\"push\" type=\"prescribed "
                "displacement\" node_set=\"top\"><dof>z</dof>"
                "<value lc=\"1\">-0.4</value></bc></Boundary> where "
                "node_set 'top' holds nodes of a rigid material. "
                "WRONG: <Boundary><rigid_bc name=\"p\" "
                "type=\"rigid_displacement\"><rb>1</rb><dof>z</dof>"
                "<value lc=\"1\">-0.4</value></rigid_bc></Boundary>. "
                "RIGHT: <Rigid><rigid_bc name=\"p\" "
                "type=\"rigid_displacement\"><rb>1</rb><dof>z</dof>"
                "<value lc=\"1\">-0.4</value></rigid_bc></Rigid>. "
                "Signal: the first form READS cleanly ("
                "`Reading file ...SUCCESS!`) and then dies at model "
                "initialisation with `Rigid nodes cannot be "
                "prescribed.`, `Boundary condition 1 (push) failed to "
                "initialize` and `Model initialization failed`, exit 1 "
                "— so a wrapper that only checks the reader line "
                "believes the deck is fine. The second form is caught "
                "at parse with tag \"rigid_bc\" (line N) : `"
                "unrecognized tag`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] The rigid DOF spelling is DIFFERENT for each "
                "rigid_bc type — this is the single most common reason a "
                "rigid deck will not read. rigid_displacement takes "
                "LOWERCASE x/y/z; rigid_rotation takes CAPITAL-R "
                "Ru/Rv/Rw; rigid_fixed takes neither, it takes six "
                "boolean SUB-TAGS Rx_dof/Ry_dof/Rz_dof/Ru_dof/Rv_dof/"
                "Rw_dof. "
                "WRONG: <rigid_bc type=\"rigid_displacement\"><rb>2</rb>"
                "<dof>Rz</dof><value lc=\"1\">-0.1</value></rigid_bc>. "
                "WRONG: <rigid_bc type=\"rigid_rotation\"><rb>1</rb>"
                "<dof>u</dof><value lc=\"1\">0.5</value></rigid_bc>. "
                "RIGHT: <Rigid>"
                "<rigid_bc name=\"lock\" type=\"rigid_fixed\"><rb>2</rb>"
                "<Rx_dof>1</Rx_dof><Ry_dof>1</Ry_dof><Ru_dof>1</Ru_dof>"
                "<Rv_dof>1</Rv_dof><Rw_dof>1</Rw_dof></rigid_bc>"
                "<rigid_bc name=\"push\" type=\"rigid_displacement\">"
                "<rb>2</rb><dof>z</dof><value lc=\"1\">-0.1</value>"
                "</rigid_bc>"
                "<rigid_bc name=\"spin\" type=\"rigid_rotation\">"
                "<rb>2</rb><dof>Ru</dof><value lc=\"1\">0.5</value>"
                "</rigid_bc></Rigid>. "
                "Signal: tag \"dof\" (line N) : invalid value: Rz for "
                "the capitalised translation DOF, and tag \"dof\" "
                "(line N) : invalid value: u for the lowercase rotation "
                "DOF, both with `Reading file ...FAILED!` and exit 1. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] A rigid body and a deformable body that do "
                "NOT share nodes need a <Contact> section, or the rigid "
                "body passes straight through the deformable one and the "
                "run still reports N O R M A L   T E R M I N A T I O N "
                "with exit 0 and every load step completed. Measured on "
                "a rigid flat punch driven 0.10 length units into an "
                "8x8x4-hex8 block from a 0.05 gap: with no <Contact> "
                "section the block's top surface does not move at all "
                "and the punch ends up 0.10 BELOW it — the whole "
                "prescribed indentation becomes interpenetration; with "
                "the <Contact> section the block surface is driven down "
                "with the punch and the residual penetration is a few "
                "percent of the indentation depth. "
                "REQUIRED pieces, all four: (1) a <Surface> for each "
                "side declared inside <Mesh>, (2) a <SurfacePair> "
                "inside <Mesh> naming them, (3) a <Contact> section "
                "referencing that pair by name, (4) the two bodies must "
                "have separate node ids — shared nodes are already "
                "bonded and need no contact. "
                "WRONG: <Mesh>...</Mesh> with the rigid and deformable "
                "parts as two <Elements> blocks and no <Surface>, no "
                "<SurfacePair>, no <Contact>. "
                "RIGHT: <Mesh> ... "
                "<Surface name=\"BlockTop\"><quad4 id=\"1\">"
                "325,326,335,334</quad4> ... </Surface> "
                "<Surface name=\"PunchBottom\"><quad4 id=\"1\">"
                "406,411,412,407</quad4> ... </Surface> "
                "<SurfacePair name=\"PunchOnBlock\">"
                "<primary>BlockTop</primary>"
                "<secondary>PunchBottom</secondary></SurfacePair>"
                "</Mesh> ... "
                "<Contact><contact type=\"sliding-elastic\" "
                "surface_pair=\"PunchOnBlock\">"
                "<laugon>PENALTY</laugon><penalty>10</penalty>"
                "<auto_penalty>1</auto_penalty><two_pass>1</two_pass>"
                "<tolerance>0.1</tolerance>"
                "<search_radius>1.0</search_radius></contact></Contact>. "
                "Signal: there is no warning of any kind for the "
                "missing contact — the only detection is to read the "
                "geometry back out with "
                "<Output><logfile><node_data data=\"z\" "
                "file=\"pos.txt\"/></logfile></Output> and check that "
                "the deformable surface under the indenter has moved. "
                "Template rigid_contact_3d_indentation ships the "
                "working deck. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] A <Contact> section with the DEFAULT "
                "penalty is silently inactive: contact is declared, the "
                "run completes, and the bodies still interpenetrate "
                "almost completely. FEBio's <penalty> default is 1 and "
                "is NOT scaled by the material stiffness, so on a body "
                "with E = 1000 a penalty of 1 is roughly five orders of "
                "magnitude too soft. Swept penalty = 1 / 10 / 100 / 1000 "
                "/ 10000 with auto_penalty off, on a 4x4x2 and an 8x8x4 "
                "block: at penalty 1 and 10 more than 99% of the "
                "prescribed indentation is absorbed as interpenetration, "
                "at 100 more than 90%, and only once penalty reaches "
                "about ten times Young's modulus does the residual "
                "penetration fall to the percent level. Both meshes give "
                "the same picture, so this is not discretization noise. "
                "REQUIRED: set <auto_penalty>1</auto_penalty>, which "
                "turns <penalty> into a dimensionless multiplier of the "
                "element stiffness, and use a multiplier of about 10; "
                "auto_penalty with a multiplier of 1 still leaves "
                "several percent penetration, and a multiplier of 100 "
                "cost convergence (`------- failed to converge at time`) "
                "on the finer mesh. "
                "WRONG: <contact type=\"sliding-elastic\" "
                "surface_pair=\"PunchOnBlock\"><penalty>1</penalty>"
                "</contact>. "
                "RIGHT: <contact type=\"sliding-elastic\" "
                "surface_pair=\"PunchOnBlock\"><laugon>PENALTY</laugon>"
                "<penalty>10</penalty><auto_penalty>1</auto_penalty>"
                "<two_pass>1</two_pass><tolerance>0.1</tolerance>"
                "<search_radius>1.0</search_radius></contact>. "
                "Signal: every one of those ten runs ended "
                "`N O R M A L   T E R M I N A T I O N` with exit 0 and "
                "all time steps completed — there is no warning, no "
                "message, nothing in the log that separates working "
                "contact from inactive contact. Verify by reading node z "
                "back out of a <logfile>. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Discretization] Contact <Surface> facets must wind so "
                "that the surface normal points AWAY from the body they "
                "belong to and TOWARDS the other body; a reversed "
                "quad4 node order disables the contact without saying "
                "so. Verified by reversing the node order of every facet "
                "on one surface at a time, on the same 8x8x4 punch "
                "deck: with both surfaces wound correctly the punch "
                "leaves a few percent residual penetration; reversing "
                "EITHER the deformable top surface or the rigid punch "
                "bottom surface returns the full prescribed indentation "
                "as interpenetration, exactly as if no <Contact> section "
                "existed; reversing BOTH makes the surfaces attract and "
                "the solve diverges. "
                "WRONG (block top face at z = 0.5, normal must be +z, "
                "written clockwise seen from above): "
                "<quad4 id=\"1\">325,334,335,326</quad4>. "
                "RIGHT (counter-clockwise seen from +z): "
                "<quad4 id=\"1\">325,326,335,334</quad4>; and for the "
                "punch bottom face, whose normal must be -z, "
                "counter-clockwise seen from BELOW: "
                "<quad4 id=\"1\">406,411,412,407</quad4>. "
                "Signal: one reversed surface still ends "
                "`N O R M A L   T E R M I N A T I O N` with exit 0 and "
                "every time step completed, and NOTHING in its output "
                "flags the disabled contact. The two logs are not quite "
                "identical, and the difference points the WRONG "
                "WAY: the CORRECT run additionally warns `Problem is "
                "diverging. Stiffness matrix will now be reformed`, "
                "because it is actually resolving contact, so the "
                "broken deck is the QUIETER of the two. Nothing appears "
                "in the reversed run that is not also in the correct "
                "one. Two reversed surfaces give "
                "`------- failed to converge at time : <t>` and "
                "`E R R O R   T E R M I N A T I O N`. Generate facets "
                "from the parent hex face rather than by hand, and check "
                "penetration numerically out of a <logfile>. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] <SurfacePair> children are <primary> and "
                "<secondary> in FEBio 4.0. The FEBio 2.x <master> / "
                "<slave> spelling is gone, and so are the 2.x contact "
                "type strings. "
                "WRONG: <SurfacePair name=\"Pair1\">"
                "<master>BlockTop</master>"
                "<slave>PunchBottom</slave></SurfacePair>. "
                "WRONG: <contact type=\"facet-to-facet sliding\" ...>, "
                "<contact type=\"sliding_with_gaps\" ...>, "
                "<contact type=\"sliding elastic\" ...> (space instead "
                "of hyphen). "
                "RIGHT: <SurfacePair name=\"Pair1\">"
                "<primary>BlockTop</primary>"
                "<secondary>PunchBottom</secondary></SurfacePair> with "
                "<contact type=\"sliding-elastic\" "
                "surface_pair=\"Pair1\">. "
                "The <SurfacePair> belongs inside <Mesh>, next to the "
                "<Surface> blocks it names; the <contact> element points "
                "at it with the REQUIRED attribute surface_pair=. An "
                "EMPTY <contact> element is also rejected: write the "
                "parameters out. Self-closing it as "
                "<contact type=\"...\" surface_pair=\"...\"/> does parse, "
                "but leaves every parameter at its default and gives "
                "inactive contact — see the penalty pitfall above. "
                "Signal: tag \"master\" (line N) : `unrecognized tag` "
                "for the 2.x surface spelling; tag \"contact\" (line N) "
                ": `invalid value for attribute \"type\"` for every 2.x "
                "or mis-hyphenated type string; tag \"contact\" (line "
                "N) : `invalid value for attribute \"surface_pair\"` when "
                "the named pair does not exist in <Mesh>; tag "
                "\"contact\" (line N) : `unrecognized tag` for "
                "<contact ...></contact> with an empty body. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] <center_of_mass> is OPTIONAL on a 'rigid body' "
                "material and is auto-computed from the element mass "
                "distribution when omitted; when supplied it silently "
                "OVERRIDES the computed value and every rotation is then "
                "taken about the coordinates you wrote. The companion "
                "flag <override_com> is not needed to make it take "
                "effect. Measured on a unit cube rotated 0.5 rad about "
                "x: omitting the tag rotates the body about its "
                "geometric centre (0.5,0.5,0.5), which is correct; "
                "writing <center_of_mass>0,0,5</center_of_mass> rotates "
                "it about (0,0,5) and translates the whole body metres "
                "away, and adding <override_com>1</override_com> changes "
                "nothing — the two runs agree to every printed digit. "
                "WRONG: copying <center_of_mass> from another deck, or "
                "guessing it — for an impactor spanning z in [0.5, 1.0] "
                "the centre is 0.75, not 1.0. "
                "RIGHT: <material id=\"2\" name=\"Punch\" "
                "type=\"rigid body\"><density>1.0</density></material> — "
                "omit the tag and let FEBio compute it. "
                "Signal: none. Both the correct and the nonsense centre "
                "end `N O R M A L   T E R M I N A T I O N` with exit 0 "
                "and all steps completed; the only detection is to read "
                "node positions back out of a <logfile> and check where "
                "the body actually pivoted. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "rigid_body_3d_pushdown": _rigid_body_3d_pushdown,
    "rigid_contact_3d_indentation": _rigid_contact_3d_indentation,
}
