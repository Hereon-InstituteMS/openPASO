"""Cross-backend collation pitfalls.

When a simulation engineer ports the same problem between two backends —
e.g. validating a fenics result against a kratos run — the failure modes
are NOT in any single backend's catalog. They live in the *delta* between
two backends that both claim to solve the same problem.

This module collects those delta-pitfalls. Each entry follows the standard
[Category]+Signal format used elsewhere in the catalog, with the extra
constraint that the Signal clause MUST name at least two backends
side-by-side. A pitfall that's "just a fenics issue" or "just a kratos
issue" belongs in src/backends/<be>/, not here.

Surfaced via `knowledge(topic='cross_backend')` — see
src/tools/consolidated.py.
"""
from __future__ import annotations


_UNITS_DESC = (
    "All 8 backends are UNIT-AGNOSTIC: they perform arithmetic on numbers "
    "and assume the user has fed them a self-consistent set. None validates "
    "that your inputs form a coherent unit system. This is the single most "
    "common bug when porting a problem between backends."
)

_UNITS_PITFALLS = [
    "[Cross-Backend][Units] FEBio's documented user-manual default is "
    "mm-tonne-s (mass = tonne, length = mm, time = s, derived force = N, "
    "derived stress = MPa, derived density = tonne/mm^3 = 1e-9 SI density). "
    "Most fenics / kratos / dealii / ngsolve / skfem / dune / 4C examples "
    "use SI (m, kg, s, derived Pa, derived kg/m^3). Signal: same problem "
    "in two backends returns results differing by exactly 1e3, 1e6, or "
    "1e9 — that factor is the unit-system mismatch, not a numerical bug. "
    "Defense: pick ONE unit system upfront before touching any backend, "
    "put it in a single header constant (E_PA = 210e9 OR E_MPa = 210e3, "
    "never both), and convert ALL backend inputs to match. Verify with a "
    "dimensional sanity check on the FIRST timestep: max displacement "
    "should match analytical expectation to within 10%.",

    "[Cross-Backend][Units] Density conversion is the most-missed step. "
    "Steel in SI is 7850 kg/m^3; in mm-tonne-s it is 7.85e-9 tonne/mm^3 "
    "(NOT 7850, NOT 7.85e-6). Signal: explicit dynamics or modal analysis "
    "ported between SI fenics and mm-tonne-s FEBio gives natural "
    "frequencies off by a factor of sqrt(1e6) ~ 1000. Defense: when "
    "porting from SI to mm-tonne-s, divide density by 1e9 (NOT 1e6, NOT "
    "1e3); when porting from mm-tonne-s to SI, multiply density by 1e9.",

    "[Cross-Backend][Units] Material model parameters embedded in physics: "
    "e.g., a Neo-Hookean shear modulus mu=80 GPa is 8e10 Pa in SI but "
    "8e4 MPa in mm-tonne-s. Signal: hyperelastic problem ported between "
    "fenics (SI) and FEBio (mm-tonne-s) shows max stress off by 1e6 (Pa "
    "vs MPa) — easy to miss because both backends report numbers without "
    "units. Defense: write a units-comment block at the top of every "
    "input file that lists the unit of EACH material parameter, not just "
    "E and nu.",

    "[Cross-Backend][Units] Loads and BCs: an applied traction of 100 in "
    "SI (fenics / kratos / dealii defaults) means 100 Pa, in mm-tonne-s "
    "(FEBio defaults) means 100 MPa = 100e6 Pa — a 1e6 difference. "
    "Signal: contact / hyperelastic / plasticity problem produces "
    "unphysically large deformations in one backend (e.g. FEBio reading "
    "100 as 100 MPa), near-zero in another (e.g. fenics reading 100 as "
    "100 Pa), for the 'same' load value. Defense: tractions and "
    "pressures are where the mismatch bites hardest because every "
    "backend accepts a bare number. Always paste the unit-system "
    "constant next to the load definition in code review.",
]

_UNITS_SIGNAL = (
    "[Cross-Backend][Units] Aggregate: cross-backend numerical "
    "discrepancies that are CLEAN powers of 1000 (1e3, 1e6, 1e9, 1e12) "
    "are almost always unit-system mismatch, NOT a numerical bug. If you "
    "see exactly 1000x, 1e6x, or 1e9x between backends, fix the unit "
    "system before debugging anything else."
)


_NODE_ORDER_DESC = (
    "Mesh-format converters (Gmsh, VTK, ABAQUS .inp) and backend internal "
    "element orderings do NOT agree on the local numbering of element "
    "nodes. A quadratic tet has 10 nodes and the on-edge midpoint nodes "
    "appear in different positions in Gmsh order vs VTK order vs ABAQUS "
    "order vs each backend's internal order."
)

_NODE_ORDER_PITFALLS = [
    "[Cross-Backend][Mesh] Quadratic-tet (Tet10) midpoint-node ordering "
    "differs: Gmsh has midpoints in order (0,1)(1,2)(0,2)(0,3)(1,3)(2,3); "
    "ABAQUS uses (0,1)(1,2)(0,2)(0,3)(2,3)(1,3) — note swapped 5th/6th. "
    "VTK uses yet another ordering. fenics/dolfinx imports Gmsh order "
    "natively via dolfinx.io.gmshio.read_from_msh; dealii's GridIn "
    "re-orders to its own internal scheme; FEBio reads ABAQUS order. "
    "Signal: an MMS convergence study on a 10-node tet shows the right "
    "rate in one backend but a polluted rate (1.5 instead of 3) in "
    "another — caused by midpoints being mis-located after import. "
    "Defense: for Tet10 / Hex20 / Quad9, never trust a converter; verify "
    "with a single Gauss-point test: place a known polynomial f(x,y,z) "
    "on the nodes, evaluate at the cell centroid, compare across "
    "backends; mismatch reveals the ordering bug.",

    "[Cross-Backend][Mesh] Linear-quad face orientation: a 2D quad mesh "
    "has each face winding either CCW (mathematics convention, "
    "fenics/dolfinx default) or CW (some FEBio .feb files via legacy "
    "converters). Signal: a Stokes lid-driven cavity problem solved on "
    "the SAME .msh file by fenics and skfem gives velocity fields that "
    "look mirrored or flipped, because one backend interpreted the mesh "
    "winding opposite to the other. Defense: always pass meshes through "
    "one canonical converter (gmsh+meshio) before feeding any backend; "
    "never bridge ABAQUS->FEBio->fenics in one pipeline. Anchor with a "
    "single-element sanity test: apply a known body force, check that "
    "the centroid displacement matches analytical to 5 digits.",
]

_NODE_ORDER_SIGNAL = (
    "[Cross-Backend][Mesh] If a converged MMS rate in one backend "
    "becomes non-converged in another on the SAME mesh file, suspect "
    "node ordering BEFORE suspecting the discretisation. Validate with "
    "a sentinel: a single high-order element with a known analytic "
    "field, compared node-by-node."
)


_LE_SEMANTICS_DESC = (
    "The phrase 'linear elastic' is overloaded across backends. Some "
    "backends interpret it as small-strain (Cauchy stress = D * sym(grad "
    "u), Green strain tensor LINEARISED); some use the same template "
    "name but actually wire up Total Lagrangian Green-Lagrange / 2nd "
    "Piola-Kirchhoff. Both produce identical results in the small-strain "
    "limit but diverge when applied strain exceeds ~5%."
)

_LE_SEMANTICS_PITFALLS = [
    "[Cross-Backend][Physics] 'linear_elasticity' across backends: "
    "fenics/dolfinx + skfem + ngsolve + dune use small-strain (Cauchy "
    "stress = lambda*tr(eps)*I + 2*mu*eps with eps = sym(grad u), no F "
    "= I + grad u, no PK1 / PK2 machinery). FEBio's 'isotropic elastic' "
    "material (close name) uses the same small-strain form ONLY when "
    "the <analysis>STATIC</analysis> block is 'small-strain'; if the "
    "analysis is 'dynamic' or any kinematic_type other than "
    "small_strain, FEBio silently switches to a finite-strain PK1 "
    "evaluation of the SAME Hooke law. Kratos's LinearElasticIsotropic3D "
    "is always small-strain. 4C's MAT_ELAST_HOOKE in a NONLINEAR "
    "analysis is a geometrically-nonlinear St-Venant-Kirchhoff (finite "
    "strain). Signal: a uniaxial tension test stretched to 10% strain "
    "gives the SAME stress in fenics-LE and "
    "kratos-LinearElasticIsotropic3D (small-strain), but FEBio in "
    "dynamic mode and 4C in nonlinear mode give a slightly DIFFERENT "
    "stress because they silently switch to St-Venant-Kirchhoff. "
    "Defense: when validating a 'linear elastic' problem across "
    "backends, always check the analysis-type setting of EACH backend "
    "and force kinematics='small_strain' / 'linear' / KINEM linear in "
    "the input file even when you THINK it's the default.",
]

_LE_SEMANTICS_SIGNAL = (
    "[Cross-Backend][Physics] 'Linear elastic' is not the same material "
    "law in every backend at >5% strain. The small-strain / "
    "Green-Lagrange branch depends on the BACKEND's analysis-type "
    "metadata, not on the material name. Force the kinematics tag "
    "explicitly in every backend's input."
)


_DIRICHLET_DESC = (
    "Backends impose Dirichlet boundary conditions through different "
    "mechanisms: row-elimination (strong) vs penalty vs Nitsche vs "
    "Lagrange multiplier. Each affects matrix structure, condition "
    "number, and the spurious stress/reaction values at the boundary."
)

_DIRICHLET_PITFALLS = [
    "[Cross-Backend][BC] Dirichlet enforcement: dolfinx's "
    "fem.dirichletbc uses strong row-elimination (diagonal = 1, "
    "off-diagonal = 0, RHS = bc_value); skfem's condense() does the "
    "same. NGSolve's fes.Update() with Dirichlet='boundary' flags the "
    "DOFs and the linear-form assembler skips them. Kratos applies a "
    "PENALTY by default for AssignVectorByDirectionProcess at the "
    "modeler/constructor level; the penalty coefficient is 1e30 unless "
    "overridden. 4C uses strong elimination by default but switches to "
    "penalty/Nitsche for some contact / TSI use cases. FEBio uses "
    "penalty for prescribed displacements on contact surfaces. Signal: "
    "cross-backend reaction-force comparison on the SAME Dirichlet "
    "boundary shows agreement on interior fields to 1e-8 but "
    "reaction-force agreement only to 1e-3 — the discrepancy is the "
    "penalty residual in the backend that uses penalty. Defense: when "
    "porting a problem, force STRONG elimination where each backend "
    "supports it (penalty_coefficient=None or explicit elimination "
    "flag); accept that reaction forces computed from penalty methods "
    "carry an O(1/penalty) bias.",
]

_DIRICHLET_SIGNAL = (
    "[Cross-Backend][BC] If two backends agree on interior fields but "
    "disagree on Dirichlet-boundary reaction forces, check the BC "
    "enforcement mechanism (strong vs penalty) BEFORE re-meshing or "
    "refining."
)


_RESTART_DESC = (
    "Backend-internal checkpoint/restart files are NEVER cross-portable. "
    "Each backend writes its own binary format with backend-specific "
    "assumptions about mesh layout, DOF ordering, time-step metadata, "
    "and solver state."
)

_RESTART_PITFALLS = [
    "[Cross-Backend][Output] Restart files are PER-BACKEND. FEBio's "
    ".restart binary, fenics's dolfinx checkpoint .bp/.xdmf (h5 "
    "backend), 4C's .restart YAML+binary pair, ngsolve's .pickle dump, "
    "kratos's .post.restart all use incompatible layouts. There is no "
    "'open restart format'. Signal: trying to chain a 4C structural "
    "pre-stress computation into a fenics dynamic analysis via "
    "checkpoint+restart fails immediately with a binary-incompatible "
    "parse error, or — worse — succeeds at loading garbage and runs to "
    "NaN. Defense: always exchange FIELD DATA (displacement, velocity, "
    "stress) between backends via a NEUTRAL format: XDMF+HDF5 for "
    "time-dependent, .vtu for snapshots, .npz for raw arrays. Never "
    "exchange solver state. Re-initialise the solver from the field at "
    "the new backend's t0.",
]

_RESTART_SIGNAL = (
    "[Cross-Backend][Output] Cross-backend workflows must exchange "
    "FIELDS, not SOLVER STATE. The neutral exchange format is XDMF+HDF5 "
    "(time-dependent) or .vtu / .npz (snapshots). Reinitialise solver "
    "state on import."
)


_MPI_DESC = (
    "Each backend has its own MPI bootstrapping convention. Mixing "
    "them — or wrapping the wrong one in mpirun — produces silent "
    "serial runs masquerading as parallel."
)

_MPI_PITFALLS = [
    "[Cross-Backend][Performance] MPI launch: dolfinx / 4C use "
    "MPI_COMM_WORLD natively and require `mpirun -n N python script.py` "
    "(or the 4C binary). NGSolve uses TaskManager for shared-memory "
    "parallel (single-process threads) by default and ngs-petsc-mpi for "
    "MPI; mixing `mpirun -n N ngspy script.py` with TaskManager-using "
    "scripts produces N independent runs, each computing the FULL "
    "problem (wasted compute, results identical, no parallel speedup). "
    "Kratos uses MPI via its own KratosTrilinosApplication; ditto "
    "wrapping a non-Trilinos Kratos script in mpirun is a no-op. "
    "Signal: a 4-process mpirun gives speedup ~0.95x (slightly slower "
    "than serial) on a problem you expect to scale linearly — almost "
    "certain you wrapped a backend that doesn't natively use "
    "MPI_COMM_WORLD. Defense: verify MPI is actually used by adding a "
    "MPI.COMM_WORLD.Get_rank()/Get_size() print at the start of every "
    "backend-MPI script; if rank=0 and size=1 on all N processes, your "
    "mpirun did nothing useful.",
]

_MPI_SIGNAL = (
    "[Cross-Backend][Performance] A backend that doesn't natively use "
    "MPI_COMM_WORLD will not parallelise via `mpirun -n N`. Verify with "
    "a Get_rank()/Get_size() print before debugging scalability."
)


_ELEMENT_TYPE_DESC = (
    "Element-type names like 'hex8' / 'HEX8' / 'C3D8' / 'Element3D8N' / "
    "'ElementHex8' refer to nominally the same 8-node hexahedral "
    "element across all 8 backends — but each backend's API exposes a "
    "DIFFERENT string for the same shape, and a copy-pasted element "
    "name from one backend's example silently fails in another."
)

_ELEMENT_TYPE_PITFALLS = [
    "[Cross-Backend][API] Element-type naming: kratos uses "
    "Element3D8N (8-node hex), Element3D4N (4-node tet), "
    "Element3D6N (6-node wedge), Element3D10N (10-node tet). 4C uses "
    "SOLIDHEX8 / SOLIDTET4 / SOLIDTET10 / SOLIDPYRAMID5 / SOLIDWEDGE6, "
    "but EARLIER 4C versions used HEX8 / TET4 directly without the "
    "SOLID prefix — the prefix was added in the migration to the "
    "unified solid namespace. fenics/dolfinx uses cell type enums "
    "mesh.CellType.hexahedron / tetrahedron / prism (lowercase), "
    "passed as the cell_type kwarg to mesh.create_box. skfem uses "
    "Python classes ElementHex1 / ElementTetP1 / ElementHex2 / "
    "ElementTetP2 (with explicit polynomial order in the class name). "
    "FEBio's .feb XML uses elem type='hex8' / 'tet4' / 'tet10' / "
    "'penta6' (note: 'penta' not 'wedge' or 'prism'). dealii uses C++ "
    "enum ReferenceCells::Hexahedron / Tetrahedron / Pyramid / Wedge. "
    "Signal: copy-pasting an element name string from one backend's "
    "example into another's input file silently fails. Kratos throws "
    "KratosError 'element type 'HEX8' is not registered' (uses "
    "Element3D8N); 4C silently emits 'unrecognized element name HEX8' "
    "in older versions vs 'SOLIDHEX8 expected' in newer; FEBio's XML "
    "parser raises XMLSyntaxError on 'hexahedron'. Defense: never "
    "copy an element string between backends. Look up the target "
    "backend's exact spelling in its own catalog (knowledge tool, "
    "topic='overview') before writing the input.",

    "[Cross-Backend][API] Wedge / prism / pyramid element naming "
    "is the most-confused: 4C calls it SOLIDWEDGE6, kratos calls it "
    "Element3D6N (the dimensionality-and-node-count name; no shape "
    "indicator at all), fenics calls it mesh.CellType.prism, dealii "
    "uses ReferenceCells::Wedge, FEBio uses 'penta6'. Five different "
    "words for the same 6-node element. Signal: a Gmsh-generated mesh "
    "containing wedge cells imports correctly to fenics (prism) and "
    "dealii (Wedge) but fails to load in kratos unless the .mdpa "
    "element block names them Element3D6N, and fails in FEBio unless "
    "the .feb element type='penta6'. Defense: pre-process Gmsh "
    "output through meshio's per-backend writers (meshio.write with "
    "the target format), not through a single-format intermediate.",
]

_ELEMENT_TYPE_SIGNAL = (
    "[Cross-Backend][API] Element-type strings (hex8 / HEX8 / "
    "SOLIDHEX8 / Element3D8N / ElementHex1 / mesh.CellType.hexahedron) "
    "are backend-specific spellings of the same shape. Never copy "
    "between backends. Look up the target backend's exact spelling "
    "in its own catalog before writing input."
)


_TIME_INTEGRATION_DESC = (
    "Implicit-dynamics time integration scheme defaults differ across "
    "backends. The SAME problem (mass-spring oscillator, structural "
    "dynamics, transient heat) ported across backends gives different "
    "trajectories because the default beta / gamma / alpha parameters "
    "are not standardised."
)

_TIME_INTEGRATION_PITFALLS = [
    "[Cross-Backend][Numerical] Newmark-beta default parameters: "
    "the 'average acceleration' / 'undamped trapezoidal' scheme is "
    "beta=1/4, gamma=1/2 (unconditionally stable, no algorithmic "
    "damping). 4C's default for structural_dynamics is beta=0.25, "
    "gamma=0.5 (matches average-acceleration). FEBio's default is "
    "ALSO beta=0.25, gamma=0.5 (matches). NGSolve's "
    "TimeIntegrationNewmark and skfem helper default to "
    "beta=0.25, gamma=0.5. Kratos's StructuralMechanicsApplication "
    "ResidualBasedNewmarkDisplacementScheme uses beta=0.25, "
    "gamma=0.5 as DEFAULT but exposes alpha_m / alpha_f for "
    "generalised-alpha; if the user passes alpha_m != 0 it switches "
    "implicitly to the Hilber-Hughes-Taylor (HHT) family and beta / "
    "gamma get RECOMPUTED from alpha, silently. dealii's "
    "step-23/step-25 wave examples use a fixed Newmark variant with "
    "theta=0.5 (which is NOT the same as gamma=0.5 — theta there is "
    "the time-discretisation parameter for the velocity, not the "
    "Newmark gamma). Signal: a 4C linear oscillator with damping "
    "ported to a Kratos generalised-alpha scheme produces a "
    "displacement amplitude that decays faster than expected even "
    "though both backends report 'Newmark beta=0.25 gamma=0.5' — the "
    "Kratos implementation silently added alpha_m=0.05 algorithmic "
    "damping. Defense: when porting a transient structural problem, "
    "explicitly set beta, gamma, alpha_m, alpha_f in EVERY backend's "
    "input (do not rely on defaults); verify with a no-damping "
    "single-DOF reference: period error after 100 cycles should be "
    "< 1% if and only if the schemes match.",
]

_TIME_INTEGRATION_SIGNAL = (
    "[Cross-Backend][Numerical] If a transient solid/structural "
    "problem matches across backends at t=0 but the displacement "
    "envelope decays at different rates, the cause is almost always "
    "an algorithmic-damping parameter (alpha_m / alpha_f) silently "
    "added by one backend's default but not the other's. Always "
    "explicitly set ALL Newmark/generalised-alpha parameters in the "
    "input, do not trust 'default'."
)


_TOLERANCE_DESC = (
    "Newton and Krylov solver convergence-tolerance defaults differ "
    "by orders of magnitude across backends. The SAME problem can "
    "report 'converged in 4 iterations' in one backend and "
    "'diverged after 50 iterations' in another not because the "
    "physics differs but because the relative-tolerance default "
    "happened to land at a different power of 10."
)

_TOLERANCE_PITFALLS = [
    "[Cross-Backend][Numerical] Newton solver default tolerances: "
    "dolfinx's NewtonSolver defaults to rtol=1e-9, atol=1e-10, "
    "max_it=50 (very strict). NGSolve's solvers.Newton() defaults "
    "to maxerr=1e-11 absolute. Kratos's "
    "ResidualBasedNewtonRaphsonStrategy defaults to relative_tol="
    "1e-4, absolute_tol=1e-9 (LOOSER on relative). 4C's default "
    "via solver_params.yaml is rel_tol=1.0e-6, abs_tol=1.0e-12 "
    "(moderate). FEBio's default Newton control is rtol=0.001 "
    "(VERY loose by FEM standards — 0.1%). dealii has no global "
    "default; each step-XX tutorial sets it locally, usually "
    "1e-6 to 1e-8. Signal: ported nonlinear problem 'converges' "
    "in FEBio (rtol 1e-3 reached easily) but the SAME tolerance "
    "request fails in fenics where rtol must reach 1e-9 — and the "
    "FEBio solution is actually still 6 orders of magnitude away "
    "from the true Newton fixed point. Defense: when porting, "
    "ALWAYS set rtol AND atol explicitly in every backend's input. "
    "Use rtol=1e-8, atol=1e-10 as a portable cross-backend baseline "
    "for production problems; rtol=1e-4 for fast smoke tests.",

    "[Cross-Backend][Numerical] Krylov (CG / GMRES / BiCGStab) "
    "default tolerances similarly differ. PETSc-backed solvers "
    "(dolfinx, 4C via PETSc) default to ksp_rtol=1e-5, ksp_atol="
    "1e-50 (essentially relative-only). NGSolve's CGSolver defaults "
    "to tol=1e-12 absolute. scipy.sparse.linalg.cg (used by skfem) "
    "defaults to rtol=1e-5, atol=0 (also relative-only). Kratos's "
    "AMGCL/Trilinos defaults are application-specific; "
    "TrilinosLinearSolver typically tol=1e-6. Signal: ported linear "
    "problem solves to 'machine precision' in NGSolve but only to "
    "1e-5 relative in PETSc-backed dolfinx, and a downstream "
    "nonlinear Newton loop in dolfinx fails to converge because "
    "the linear solve residual seeds the Newton residual at 1e-5 "
    "instead of 1e-12. Defense: set ksp_rtol AND ksp_atol "
    "explicitly when porting; tighten ksp_rtol BELOW the outer "
    "Newton rtol by at least 2 orders of magnitude (Newton rtol="
    "1e-8 → ksp_rtol=1e-10 minimum).",
]

_TOLERANCE_SIGNAL = (
    "[Cross-Backend][Numerical] If a 'converged' nonlinear solution "
    "in one backend reports a different final residual than the "
    "same problem in another backend, check the default Newton/"
    "Krylov tolerances FIRST. Same-name convergence ('converged') "
    "carries different meanings: FEBio's rtol=1e-3 default is 10^6 "
    "times looser than dolfinx's rtol=1e-9."
)


_CONTACT_FORMULATION_DESC = (
    "Contact-mechanics enforcement methods differ across backends "
    "in defaults AND in available options. The SAME contact problem "
    "(Hertzian sphere-on-plane, frictional slide, multi-body "
    "assembly) solved 'with default settings' in two backends "
    "produces non-comparable results because the constraint "
    "formulation is silently different."
)

_CONTACT_FORMULATION_PITFALLS = [
    "[Cross-Backend][Physics] Contact constraint enforcement: "
    "Kratos's ContactStructuralMechanicsApplication defaults to "
    "PENALTY (penalty_factor adaptive, starts ~E*1e3); explicit "
    "alternative is augmented_lagrange via the AugmentedLagrange "
    "process. 4C's CONTACT block defaults to STRATEGY 'Lagrange' "
    "(true Lagrange multipliers via mortar segmentation), with "
    "PENALTY available via STRATEGY 'Penalty'. FEBio uses "
    "AUGMENTED LAGRANGE by default for sliding interface "
    "(augmented_lagrangian='1' is the .feb XML attribute); "
    "switches to PENALTY when augmented_lagrangian='0'. dolfinx "
    "has no built-in contact; users build it via custom Nitsche "
    "or external library (Mirco, Conmech). NGSolve has hp-FEM "
    "Nitsche contact via the contact-mechanics tutorial. Signal: "
    "Hertzian sphere-on-plane test ported between Kratos (penalty) "
    "and 4C (Lagrange) gives matching peak pressure but the "
    "Kratos solution shows ~1e-3 penetration at the contact patch "
    "(penalty residual) while 4C shows ~1e-12 (true Lagrange "
    "satisfies zero-penetration to solver tolerance). Defense: "
    "when validating cross-backend, ALWAYS use Lagrange or "
    "augmented Lagrange — penalty's accuracy depends on a "
    "user-set penalty factor that has no universal default. State "
    "the formulation EXPLICITLY in the input.",
]

_CONTACT_FORMULATION_SIGNAL = (
    "[Cross-Backend][Physics] If two backends agree on bulk "
    "stress fields in a contact problem but disagree on the "
    "contact-patch penetration depth by orders of magnitude, "
    "the cause is penalty (one backend) vs Lagrange/augmented "
    "Lagrange (the other). Force Lagrange or augmented Lagrange "
    "explicitly when validating across backends."
)


_OUTPUT_FORMAT_DESC = (
    "Output file conventions — VTK / XDMF / ADIOS2 / .post.bin — "
    "differ in (a) what each backend can write, (b) where it puts "
    "DOFs (point-data on mesh nodes vs cell-data vs DG-style "
    "per-element data), and (c) how high-order polynomial "
    "discretisations are represented. ParaView opens all of them "
    "but interprets the same field as different things."
)

_OUTPUT_FORMAT_PITFALLS = [
    "[Cross-Backend][Output] Point-data vs cell-data: dolfinx's "
    "VTXWriter and dolfinx.io.XDMFFile.write_function write "
    "Lagrange P1 / P2 / etc. DOFs as POINT-DATA (one value per "
    "mesh vertex, with high-order DOFs interpolated to vertices). "
    "Kratos's VtkOutput writes NODAL_RESULTS as POINT-DATA too. "
    "skfem's skfem.io.json or skfem-to-meshio writes element-by-"
    "element output as CELL-DATA for DG fields. dealii's DataOut "
    "with VTK output writes high-order polynomials by SUBDIVIDING "
    "each element into a refined sub-mesh of P1 patches (so a P2 "
    "field on 100 cells produces 400 sub-cells in the .vtu file) "
    "— ParaView shows the smooth field correctly, but cell-count "
    "differs from the simulation mesh. NGSolve's vtk_output "
    "subdivides similarly (controlled by `subdivision=N` kwarg). "
    "FEBio's .xplt is its own binary format and only ParaView via "
    "the FEBio plugin reads it natively. Signal: a P2 stress field "
    "ported to ParaView via fenics-XDMF (interpolated to vertices) "
    "and via dealii-VTK (subdivided) shows the SAME peak value but "
    "the dealii output reports 4-9x more cells, and a downstream "
    "ParaView Calculator filter that iterates over cells produces "
    "different integrated quantities. Defense: when cross-backend "
    "post-processing in ParaView, force ALL outputs to a single "
    "convention (subdivision=1 in dealii/NGSolve, P1-projection in "
    "fenics) before any Filter / Calculator operation; or do the "
    "post-processing on raw arrays via meshio/numpy, not in "
    "ParaView.",

    "[Cross-Backend][Output] XDMF vs .bp (ADIOS2) vs .vtu: dolfinx "
    "0.10+ recommends VTXWriter -> .bp (ADIOS2 directory) for "
    "high-order Lagrange Functions; XDMFFile.write_function "
    "rejects P>1 with 'RuntimeError: Degree of output Function "
    "must be same as mesh degree' (the standard P1-only XDMF "
    "format). 4C writes XDMF natively for post-processing. Kratos "
    "default GidOutput is GID-format binary (not XDMF, not VTU); "
    "VtkOutput writes .vtu. NGSolve writes .vtu via vtk_output. "
    "ParaView 5.10+ reads ALL of .bp/.xdmf/.vtu; ParaView <5.10 "
    "reads .xdmf/.vtu only. Signal: a cross-backend workflow "
    "where one stage writes .bp (dolfinx VTXWriter, the only "
    "P>1-capable format) and another stage reads .xdmf via "
    "meshio raises meshio.ReadError because meshio < 5 doesn't "
    "speak .bp. Defense: pick ONE neutral exchange format for "
    "cross-backend pipelines: .vtu (linearised, ParaView 5+ "
    "reads), or interpolated-to-P1 XDMF. Reserve .bp for "
    "dolfinx-only chains.",
]

_OUTPUT_FORMAT_SIGNAL = (
    "[Cross-Backend][Output] If the same field-export pipeline "
    "produces different integrated quantities in ParaView Filters "
    "(min/max, integrate, surface-cell-count), the cause is "
    "almost always cell-subdivision (dealii/NGSolve subdivide; "
    "fenics/Kratos interpolate to vertices). Force "
    "subdivision=1 / P1-projection on ALL backends before any "
    "post-processing Filter."
)


_INTEGRATION_ORDER_DESC = (
    "Gauss-quadrature order selection differs across backends. "
    "The SAME polynomial-order element (P2 / Q2 / Hex8) may use "
    "different default numbers of integration points in each "
    "backend, leading to slightly different stress / strain-"
    "energy values for the same physical problem."
)

_INTEGRATION_ORDER_PITFALLS = [
    "[Cross-Backend][Numerical] Default Gauss integration order: "
    "dolfinx selects quadrature degree automatically based on UFL "
    "form analysis (estimate_total_polynomial_degree) UNLESS the "
    "user passes form_compiler_options={'quadrature_degree': N} "
    "or sets it on the dx measure (dx(metadata={'quadrature_"
    "degree': N})); the auto-estimate may be CONSERVATIVE (over-"
    "integrate) or INSUFFICIENT (under-integrate) depending on "
    "nonlinearity. skfem's @BilinearForm uses Basis(... intorder="
    "2*p) by default where p is the element order. NGSolve "
    "Integrate() uses order = 2*deg(form) by default but accepts "
    "order=N kwarg. dealii's QGauss<dim>(n) is explicit — n = "
    "ceil((2*p+1)/2) is the standard textbook formula but every "
    "tutorial sets it manually. Kratos's elements have HARD-CODED "
    "integration rules per element type (Element3D8N uses 2x2x2 "
    "= 8 points for full integration; reduced is via "
    "Element3D8NReduced with 1 point). 4C uses GAUSSRULE keyword "
    "in input: '2x2x2' / '3x3x3' / 'reduced'. FEBio uses solid "
    "element 'gp_order' attribute. Signal: a hyperelastic P2 "
    "compression test ported between fenics (auto quadrature) "
    "and Kratos (fixed 2x2x2 for Element3D8N) reports slightly "
    "different peak strain-energy because fenics auto-picked "
    "quadrature degree 4 (over-integrated for cubic strain field) "
    "while Kratos's 2x2x2 under-integrates the strain energy by "
    "~0.1% (within engineering tolerance but visible in MMS "
    "convergence studies as a stalled rate at fine mesh). "
    "Defense: when validating cross-backend, explicitly set the "
    "quadrature order to the same value in every backend; for "
    "nonlinear elements use 2*p+1 minimum (where p is the shape-"
    "function order), or 2*p+2 for hyperelastic to capture the "
    "nonlinear strain-energy density.",
]

_INTEGRATION_ORDER_SIGNAL = (
    "[Cross-Backend][Numerical] If an MMS convergence rate "
    "stalls at the predicted theoretical order in one backend "
    "but achieves super-convergence in another, suspect "
    "integration order BEFORE suspecting the discretisation. "
    "Fenics auto-quadrature can over-integrate (free super-"
    "convergence) while Kratos's fixed Element3D8N rule "
    "under-integrates."
)


_BOUNDARY_TAG_DESC = (
    "Gmsh physical-group tags / boundary IDs are mesh metadata "
    "that backends interpret differently. A `gmsh model.add_physical_"
    "group(2, [10], tag=5)` in Python becomes a SURFACE id 5 in "
    "the .msh file, but each backend then maps that to its own "
    "internal boundary-id namespace with different fall-through "
    "rules when a tag is missing."
)

_BOUNDARY_TAG_PITFALLS = [
    "[Cross-Backend][Mesh] Boundary-id default fallthrough: "
    "dolfinx's mesh.locate_entities_boundary(mesh, fdim, marker) "
    "uses a USER-supplied lambda (no tag mapping at all — the "
    "user must reconstruct facet IDs from coordinates); the "
    "alternative read_from_msh path preserves Gmsh physical-"
    "group tags on a mesh.MeshTags object that the user passes "
    "to fem.dirichletbc(value, dofs, V). dealii's GridIn::read_"
    "msh imports Gmsh tags as boundary_id() attributes on the "
    "Triangulation's faces — directly usable by "
    "VectorTools::interpolate_boundary_values. Kratos's "
    "ModelPart loads sub-model-parts named after Gmsh physical "
    "groups but the .mdpa user must HAND-EDIT the sub-model-"
    "part section to expose them — Gmsh→Kratos converters don't "
    "always preserve the physical-group name. 4C's --DESIGN "
    "SURF DIRICH CONDITIONS block uses E (=design surface) "
    "indices that must MATCH the .dat mesh's NSURF / DSURF "
    "tags — the indices are 1-indexed in 4C but 0-indexed in "
    "Gmsh, so a Gmsh-exported mesh needs a +1 shift on every "
    "boundary id before 4C will accept it. FEBio .feb XML uses "
    "<Surface name=...> blocks identified by NAME not by "
    "numeric ID. Signal: SAME .msh file feeds dolfinx and 4C; "
    "dolfinx applies Dirichlet on facets with tag=5 correctly; "
    "4C applies Dirichlet on the WRONG surface because the 4C "
    "input still references E=5 but the Gmsh→.dat converter "
    "shifted the tag to E=6. Defense: when porting a Gmsh "
    "mesh between backends, always print the per-facet tag "
    "histogram in EACH backend's loader (count facets per "
    "boundary_id) and compare; mismatch reveals the off-by-one "
    "or naming bug.",
]

_BOUNDARY_TAG_SIGNAL = (
    "[Cross-Backend][Mesh] If a Dirichlet boundary 'works' in "
    "one backend and quietly applies to the wrong face in "
    "another with the SAME mesh, the cause is 1-indexed vs "
    "0-indexed boundary tag conventions (Gmsh 0 vs 4C 1) or a "
    "sub-model-part name not preserved in the .mdpa converter. "
    "Print per-facet tag histograms before applying any BC."
)


_PLASTICITY_RETURN_MAP_DESC = (
    "Plasticity return-mapping algorithms — radial return / "
    "cutting-plane / closest-point projection / Newton-on-yield "
    "— differ across backends in (a) algorithm choice and (b) "
    "default convergence criteria. Same J2 / Mohr-Coulomb / "
    "Drucker-Prager material with default solver settings "
    "produces slightly different stress paths in different "
    "backends."
)

_PLASTICITY_RETURN_MAP_PITFALLS = [
    "[Cross-Backend][Numerical] Return-mapping defaults for J2 "
    "plasticity: 4C's MAT_PLASTIC_VONMISES uses radial-return "
    "(closed-form for J2 + isotropic linear hardening, exact in "
    "one step). Kratos's SmallStrainJ2Plasticity uses an "
    "implicit Newton-on-yield-surface with tol=1e-6 by default. "
    "FEBio's plastic-fluid and elastoplastic-isotropic-hardening "
    "use radial-return but the convergence tol is rtol=1e-3 "
    "(consistent with FEBio's looser Newton defaults). dolfinx "
    "has no built-in plasticity — users implement custom return "
    "maps via UFL+conditional or external libraries (dolfiny). "
    "Signal: SAME uniaxial cyclic tension-compression test on "
    "J2 + linear hardening, loaded past yield, produces matching "
    "elastic branches but different residual plastic strain "
    "after the first half-cycle: 4C and FEBio with radial-return "
    "give the analytic value to 1e-12; Kratos with default "
    "Newton-on-yield gives 1e-6 (its internal rtol). Defense: "
    "use radial-return where the material law supports it "
    "(linear hardening, isotropic only); for nonlinear "
    "hardening or anisotropic yield, tighten the internal "
    "return-map tol to 1e-10 explicitly in every backend.",

    "[Cross-Backend][Numerical] Mohr-Coulomb tension cut-off: "
    "kratos's MohrCoulombPlasticity has an OPTIONAL tension "
    "cut-off enabled by setting `tension_cutoff_factor` > 0; "
    "default is 0 (no cut-off, allowing negative principal "
    "stresses past the apex). 4C's MAT_MohrCoulomb has tension "
    "cut-off CONTROLLED via apex_smoothing_factor (default 0.1, "
    "always active). FEBio's mohr-coulomb material uses an "
    "exact apex return (no smoothing). dolfinx custom "
    "implementations vary; the most common Mohr-Coulomb-with-"
    "tension-cutoff template (e.g. dolfiny's geomechanics) has "
    "no default and the user must set it. Signal: a triaxial "
    "extension test at low confining pressure ported between "
    "Kratos (default no cut-off, allows negative apex stresses) "
    "and 4C (default smoothed cut-off at 0.1) gives the SAME "
    "peak deviatoric stress on the compression side but "
    "completely different unloading behaviour on the extension "
    "side (Kratos returns to the apex along the hydrostatic "
    "axis; 4C returns to a smoothed cone surface). Defense: "
    "when validating cross-backend Mohr-Coulomb, ALWAYS specify "
    "the tension cut-off and apex-smoothing factor explicitly "
    "in every backend.",
]

_PLASTICITY_RETURN_MAP_SIGNAL = (
    "[Cross-Backend][Numerical] If a cyclic plasticity test "
    "matches on the elastic branch but residual plastic strains "
    "differ across backends, the cause is return-map convergence "
    "tolerance (Kratos default 1e-6 vs FEBio rtol=1e-3 vs 4C "
    "exact radial-return). For Mohr-Coulomb / Drucker-Prager "
    "explicitly set the apex / tension-cut-off treatment in "
    "every backend's input."
)


_TURBULENCE_DESC = (
    "Reynolds-Averaged Navier-Stokes (RANS) turbulence model "
    "defaults differ in (a) wall treatment (wall-function vs "
    "low-Re damping), (b) inlet turbulence-intensity defaults, "
    "and (c) model-constant choices. The SAME k-epsilon "
    "channel-flow problem can converge to qualitatively "
    "different velocity profiles in two backends because the "
    "wall-treatment default differs."
)

_TURBULENCE_PITFALLS = [
    "[Cross-Backend][Physics] Turbulence wall treatment: 4C's "
    "fluid_turbulence with TURBMODEL k-epsilon defaults to "
    "STANDARD WALL FUNCTIONS (assumes y+ > 30 at the first "
    "wall-adjacent cell centroid); a finer near-wall mesh "
    "violates the wall-function assumption silently. Kratos's "
    "FluidDynamicsApplication k-epsilon process defaults to "
    "no-slip + ad-hoc damping (allows fine wall meshes, y+ < 1 "
    "OK) but the damping constants differ from standard "
    "Launder-Spalding. NGSolve and dolfinx have no built-in "
    "RANS — users write custom UFL forms; OPENPASO (a fenics-"
    "based RANS solver) defaults to standard wall functions. "
    "FEBio has no fluid-turbulence support. Signal: SAME k-eps "
    "channel-flow problem on the SAME mesh produces different "
    "centerline velocity in 4C (wall-function, mesh too fine, "
    "wall shear over-predicted by ~15%) vs Kratos (low-Re "
    "damping, correct centerline) — the user assumed 4C's "
    "default would 'just work' on a refined mesh. Defense: "
    "report y+_max from each backend's first wall-cell on the "
    "convergence print; if y+_max < 30, switch 4C from "
    "WALL-FUNCTION to LOW-REYNOLDS (standard Launder-Sharma "
    "k-eps), or coarsen the wall mesh.",

    "[Cross-Backend][Physics] Inlet turbulence-intensity (TI) "
    "defaults: 4C's INFLOW BC for k-epsilon defaults to TI=5% "
    "and turbulent length scale Lt = 0.07*D_hydraulic. "
    "Kratos's KEpsilonHighReProcess defaults to TI=1% and "
    "Lt=0.01 (much smaller). dolfinx user-written RANS: no "
    "default, user must set. Signal: SAME pipe-flow problem "
    "with 'default turbulence inlet' shows turbulent kinetic "
    "energy at the centerline ~25x higher in 4C vs Kratos at "
    "x/D=10 — the 4C input never specified TI so 4C used 5%, "
    "Kratos used 1%. Defense: always specify TI and Lt "
    "EXPLICITLY in the inlet BC; never rely on the backend's "
    "default."
]

_TURBULENCE_SIGNAL = (
    "[Cross-Backend][Physics] RANS k-eps/k-omega/SST turbulence "
    "results that disagree across backends on the SAME mesh + "
    "SAME inlet usually fail on (1) wall treatment default "
    "(wall-function vs low-Re) and (2) inlet TI default. State "
    "both explicitly in every backend's input — defaults vary "
    "by 5-25x."
)


_MATERIAL_ORIENTATION_DESC = (
    "Anisotropic material orientation — fiber direction in "
    "transverse isotropy, layup angle in laminated shells, "
    "principal-axis frame in orthotropic elasticity — is "
    "specified through different mechanisms in each backend "
    "and the default fallback (when not specified) varies."
)

_MATERIAL_ORIENTATION_PITFALLS = [
    "[Cross-Backend][Physics] Fiber-direction specification in "
    "transversely-isotropic / orthotropic materials: 4C's "
    "MAT_AAA_FIBER specifies the fiber direction via a "
    "constant vector or a fiber-field input file (.fib) per "
    "element. FEBio's <transversely_isotropic>...<fiber type='"
    "vector'>1,0,0</fiber> specifies fibers either as element-"
    "wise vectors or by referencing a coordinate-system node "
    "set. Kratos's FiberReinforcedMaterial uses a "
    "FIBER_DIRECTION_1 nodal/elemental variable set via a "
    "process. dolfinx custom UFL: user defines a vector-valued "
    "Function f and writes the constitutive law explicitly "
    "(no default direction). dealii similarly user-supplied. "
    "DEFAULT FALLBACK when fiber direction is unspecified: 4C "
    "uses (1, 0, 0) global x; FEBio uses (1, 0, 0) global x; "
    "Kratos returns an error 'FIBER_DIRECTION_1 not initialised "
    "on element'. Signal: ported fiber-reinforced beam-bending "
    "problem where the fiber should be along the beam axis but "
    "the .feb / .dat input file forgot to specify it: 4C and "
    "FEBio silently use global x (which COULDN'T match the "
    "beam axis if the beam is oriented along y), giving "
    "qualitatively wrong stiffness. Defense: ALWAYS specify "
    "fiber direction explicitly (no defaults); verify with a "
    "simple uniaxial tension along the fiber axis — the "
    "stiffness should match the fiber-direction Young's "
    "modulus, not the matrix modulus.",
]

_MATERIAL_ORIENTATION_SIGNAL = (
    "[Cross-Backend][Physics] If an anisotropic-material "
    "problem (transverse isotropy, orthotropy, laminated shell) "
    "gives qualitatively wrong stiffness on a SAME-mesh "
    "cross-backend port, the cause is almost always an "
    "unspecified fiber/principal-axis direction. 4C/FEBio "
    "silently fall back to global x; Kratos errors out. "
    "Specify direction explicitly in every backend."
)


_FREQUENCY_DESC = (
    "Eigenvalue / modal analysis backends differ in whether "
    "they report eigenvalues as angular frequency squared "
    "(omega^2 = (2*pi*f)^2 in rad^2/s^2), as angular frequency "
    "(omega in rad/s), or as ordinary frequency (f in Hz). "
    "The same modal problem solved by two backends produces "
    "numerical results that look unrelated until you account "
    "for factors of 2*pi and squaring."
)

_FREQUENCY_PITFALLS = [
    "[Cross-Backend][Numerical] Eigenvalue return convention "
    "for modal analysis: SLEPc (used by dolfinx via "
    "dolfinx.fem.petsc + slepc4py) and PETSc returns "
    "eigenvalues as the literal numerical eigenvalues of the "
    "(K, M) generalised problem K phi = lambda M phi — i.e. "
    "lambda = omega^2 in rad^2/s^2 (NOT Hz, NOT rad/s). NGSolve's "
    "ArnoldiSolver / PINVIT returns lambda = omega^2 in the same "
    "convention. skfem.utils.solver_eigen.scipy.LinearEig wraps "
    "scipy.sparse.linalg.eigsh which also returns omega^2. "
    "Kratos's EigensolverStrategy in StructuralMechanics returns "
    "lambda = omega^2 by default but the FrequencyResponse "
    "post-processor REPORTS them as Hz after sqrt(lambda)/(2*pi). "
    "4C's MODAL ANALYSIS section reports frequencies in Hz "
    "directly (post-processed from lambda internally). FEBio's "
    "modal analysis reports omega in rad/s (not omega^2, not "
    "Hz). dealii has no built-in modal solver; tutorial step-36 "
    "uses SLEPc and reports the raw lambda. Signal: a 1D "
    "cantilever-beam-bending modal analysis on the same mesh "
    "gives 'first natural frequency = 247' in 4C (Hz, correct) "
    "vs 'first eigenvalue = 2.41e6' in dolfinx-SLEPc (omega^2 in "
    "rad^2/s^2, also correct, just unit-converted) vs '1553' in "
    "FEBio (omega in rad/s, correct). All three are the same "
    "physical frequency; users porting cross-backend often miss "
    "the omega^2 / omega / Hz distinction and conclude one "
    "backend is wrong by a factor of (2*pi)^2 ~ 39.5 or 2*pi ~ "
    "6.28. Defense: always print BOTH omega^2 AND sqrt(omega^2)/"
    "(2*pi) [Hz] in every backend's modal output; compare Hz "
    "values, never raw eigenvalues."
]

_FREQUENCY_SIGNAL = (
    "[Cross-Backend][Numerical] Cross-backend modal-analysis "
    "discrepancies of (2*pi)^2 ~ 39.5 or 2*pi ~ 6.28 are NOT "
    "physical errors — they are unit conventions (omega^2 in "
    "rad^2/s^2 vs omega in rad/s vs f in Hz). Always convert to "
    "Hz before comparing. SLEPc/PETSc/NGSolve/scipy return "
    "omega^2; FEBio returns omega; 4C returns Hz."
)


_MESH_QUALITY_DESC = (
    "Mesh-quality acceptance thresholds — aspect ratio, "
    "skewness, Jacobian positivity, minimum dihedral angle — "
    "differ across backends in both the default rejection "
    "threshold AND in whether a bad element causes a hard "
    "error vs a silent slow-down via solver ill-conditioning."
)

_MESH_QUALITY_PITFALLS = [
    "[Cross-Backend][Mesh] Negative-Jacobian element rejection: "
    "4C's DiscretizationReader explicitly rejects elements with "
    "Jacobian determinant <= 0 at any Gauss point — abort with "
    "'element X has negative Jacobian'. FEBio's element checker "
    "also rejects negative Jacobian and refuses to proceed past "
    "input parsing. Kratos's elements compute Jacobian on the "
    "fly per assembly call and silently produce NaN in the "
    "stiffness matrix when J<=0 (no upfront check); the LINEAR "
    "SOLVER then reports 'matrix singular' or runs and returns "
    "garbage. dolfinx's dolfinx.mesh.create_mesh validates "
    "cell orientation at construction; a negative-Jacobian cell "
    "raises RuntimeError 'Cell orientation is invalid'. NGSolve's "
    "Netgen mesher generates only positive-Jacobian elements by "
    "construction but READING an external bad mesh via "
    "Mesh(filename) silently accepts negatives. dealii's GridIn "
    "imports without geometric validation; the first assembly "
    "fails. Signal: a Gmsh-exported mesh with one inverted "
    "tetrahedron (common after CSG boolean operations near sharp "
    "features) is rejected upfront by 4C/FEBio/dolfinx (clear "
    "error message with element id) but loaded silently by "
    "Kratos and NGSolve-from-file; the user discovers it only "
    "when the SECOND solver run starts giving NaN residuals. "
    "Defense: always run a mesh-validation pass (gmsh --check or "
    "pyvista's mesh.compute_cell_quality()) BEFORE feeding any "
    "backend; a single inverted element wastes hours when only "
    "caught by 'matrix singular'.",

    "[Cross-Backend][Mesh] Aspect-ratio / sliver-element "
    "tolerance: 4C has no hard aspect-ratio rejection (silently "
    "accepts slivers; solver may diverge on ill-conditioned "
    "system). FEBio warns on aspect ratio > 100 but proceeds. "
    "Kratos has no built-in check. dolfinx delegates to PETSc "
    "linear-solver tolerance which silently struggles. NGSolve's "
    "Netgen mesher avoids slivers by construction with default "
    "quality threshold 0.3; users overriding `quad_dominated=True` "
    "or `optsteps=0` can produce slivers. Signal: a 'good' "
    "tetrahedral mesh from Gmsh with optimisation disabled "
    "contains O(10) slivers; converged in 4C in 1000 GMRES "
    "iterations (slow but works); diverged in Kratos AMGCL with "
    "default settings; clean exit in NGSolve only because the "
    "Netgen optimiser silently improved it on import. Defense: "
    "always check max element aspect ratio (gmsh GUI tools menu) "
    "before solving; if > 50, refine the mesh-generation "
    "parameters BEFORE the solver inevitably has trouble."
]

_MESH_QUALITY_SIGNAL = (
    "[Cross-Backend][Mesh] If a mesh 'works' in one backend but "
    "produces 'matrix singular' / NaN residuals / silent "
    "divergence in another, suspect (1) negative-Jacobian "
    "elements silently accepted by Kratos/NGSolve-from-file but "
    "rejected by 4C/FEBio/dolfinx, or (2) sliver elements with "
    "aspect ratio > 50. Run gmsh --check before feeding any "
    "backend."
)


_STRESS_MEASURE_DESC = (
    "Output stress fields can be reported as Cauchy (true) "
    "stress sigma, 1st Piola-Kirchhoff PK1, 2nd Piola-Kirchhoff "
    "PK2, or as engineering stress sigma_eng. All four agree "
    "in the infinitesimal-strain limit but diverge at finite "
    "strain. Backends differ in WHICH stress measure they "
    "default to in output files."
)

_STRESS_MEASURE_PITFALLS = [
    "[Cross-Backend][Output] Default stress-measure in output: "
    "dolfinx writes whatever the user-projected stress UFL "
    "expression evaluates to — typically the user writes "
    "sigma = mu*(F*F.T - I) + lam*ln(J)*I (Cauchy in current "
    "configuration). FEBio's <stress> output element writes "
    "Cauchy stress sigma in the deformed configuration. 4C's "
    "STRUCTURAL_OUTPUT element 'stress' option 'cauchy' writes "
    "Cauchy; option 'pk2' writes PK2 in the reference config; "
    "default is determined by KINEM (linear -> infinitesimal, "
    "nonlinear -> PK2 unless overridden). Kratos's "
    "ConstitutiveLaw::CalculateMaterialResponseCauchy returns "
    "Cauchy by default but VonMisesStress nodal-value "
    "calculation uses whatever the constitutive law's "
    "STRESS_VECTOR contains (varies). NGSolve and skfem custom "
    "UFL: user-controlled. dealii's data_out.add_data_vector "
    "with the user's stress lambda — convention is whatever the "
    "user computed. Signal: hyperelastic uniaxial tension at "
    "10% stretch — Cauchy reports sigma = mu*(lambda^2 - 1/"
    "lambda) ~ 0.21 mu; PK1 reports P11 = mu*(lambda - 1/"
    "lambda^2) ~ 0.18 mu; PK2 reports S11 = mu*(1 - 1/"
    "lambda^3) ~ 0.16 mu. All correct in their respective "
    "configurations, all DIFFERENT numerical values for the "
    "same physical state. Defense: explicitly name the stress "
    "measure in every backend's output config; convert to "
    "Cauchy for cross-backend comparison via sigma = (1/J)*F*"
    "PK1 = (1/J)*F*PK2*F.T."
]

_STRESS_MEASURE_SIGNAL = (
    "[Cross-Backend][Output] Cross-backend hyperelastic stress "
    "values differing by ~10-20% at >5% strain are almost "
    "always Cauchy-vs-PK1-vs-PK2 reporting differences, not "
    "physical errors. The three measures coincide only at "
    "infinitesimal strain. Always specify the stress measure "
    "in every backend's output config."
)


_PERIODIC_BC_DESC = (
    "Periodic boundary conditions — used for representative-"
    "volume-element (RVE) homogenisation, periodic crystal "
    "structures, and infinite-domain approximations — are "
    "implemented via different constraint mechanisms across "
    "backends with different accuracy and matrix-structure "
    "implications."
)

_PERIODIC_BC_PITFALLS = [
    "[Cross-Backend][BC] Periodic-BC implementation: dolfinx "
    "uses dolfinx_mpc (a separate package, not core dolfinx) "
    "via mpc.create_periodic_constraint_geometrical or "
    "create_periodic_constraint_topological — true "
    "MasterSlaveConstraint at the DOF level, exact periodicity "
    "to solver precision. NGSolve uses the Periodic() wrapper "
    "around any FESpace, which builds master-slave pairs at "
    "construction time — exact periodicity. 4C uses --PERIODIC "
    "BOUNDARY CONDITIONS block referencing pairs of design-"
    "surfaces (master + slave); enforcement is via Lagrange "
    "multipliers (exact). Kratos's "
    "ApplyPeriodicConditionProcess uses a penalty-style "
    "approach with penalty factor 1e30 default — approximate, "
    "may show O(1/penalty) periodicity error in displacement "
    "field. FEBio has NO built-in periodic BC — users "
    "approximate via tied-pair contact (which is NOT periodic "
    "in the constitutive sense). dealii's "
    "DoFTools::make_periodicity_constraints builds an "
    "AffineConstraints object with exact periodicity. Signal: "
    "RVE-homogenisation problem on the SAME mesh: dolfinx-mpc / "
    "4C / NGSolve give matching effective stiffness tensor; "
    "Kratos's penalty-based periodic gives an effective "
    "stiffness tensor with O(1e-3) symmetry violation (the "
    "penalty residual breaks the major symmetry of C_ijkl); "
    "FEBio's tied-pair approximation diverges because the "
    "constraint is local-displacement-equality, not "
    "displacement-equality-plus-strain-periodicity. Defense: "
    "for production RVE work, use only dolfinx-mpc / 4C / "
    "NGSolve / dealii. Avoid Kratos penalty-periodic unless "
    "penalty_factor is tuned per problem AND symmetry is "
    "post-symmetrised. FEBio is not suitable for periodic-RVE "
    "homogenisation."
]

_PERIODIC_BC_SIGNAL = (
    "[Cross-Backend][BC] If an RVE-homogenisation effective "
    "stiffness tensor C_ijkl loses major symmetry (C_1122 != "
    "C_2211) by O(1e-3) across cross-backend runs, the cause "
    "is Kratos's penalty-based periodic enforcement. Switch "
    "to dolfinx-mpc, 4C Lagrange periodic, or NGSolve Periodic "
    "FESpace for true periodicity."
)


_DAMPING_DESC = (
    "Damping in transient analyses is specified through "
    "fundamentally different mathematical forms — Rayleigh "
    "(C = alpha*M + beta*K), structural / hysteretic damping "
    "ratio xi, viscous coefficient c per element — and "
    "backends differ in which form they use as 'damping' in "
    "their input file. The SAME 'damping ratio = 0.05' "
    "request can yield wildly different actual energy "
    "dissipation across backends."
)

_DAMPING_PITFALLS = [
    "[Cross-Backend][Physics] Damping specification: 4C's "
    "structural_dynamics block accepts alpha (mass-proportional) "
    "+ beta (stiffness-proportional) Rayleigh coefficients "
    "directly: C = alpha*M + beta*K. NO default — must specify "
    "both. Kratos's StructuralMechanicsApplication accepts "
    "RAYLEIGH_ALPHA + RAYLEIGH_BETA on each element's material "
    "parameters; if missing, treated as 0 (UNDAMPED silently). "
    "FEBio's <rigid_body><damping> takes a viscous coefficient "
    "per RB (not Rayleigh, not structural). NGSolve's "
    "TimeIntegrationNewmark accepts gamma (algorithmic damping "
    "via the gamma > 0.5 trick) but NOT physical Rayleigh; "
    "physical damping must be added by the user as a separate "
    "M-term in the bilinear form. dolfinx has no built-in "
    "damping; user writes the Rayleigh expression in UFL. "
    "dealii varies per tutorial. Signal: same modal analysis "
    "with 'damping ratio xi = 0.05' specified in 4C "
    "(translates to alpha = 2*xi*omega_1, beta = 2*xi/"
    "omega_max for a mode pair) vs Kratos (interpreted as "
    "Rayleigh alpha = 0.05 directly, beta = 0) gives different "
    "decay envelopes — 4C correctly damps modes between "
    "omega_1 and omega_max; Kratos's alpha=0.05 alone damps "
    "ALL modes uniformly via the mass term, over-damping "
    "high-frequency modes by orders of magnitude. Defense: "
    "always specify damping as Rayleigh alpha + beta "
    "explicitly; do NOT pass 'damping ratio' to a backend "
    "without first converting to alpha + beta using the "
    "two-mode formula alpha = 2*xi*om1*om2/(om1+om2), "
    "beta = 2*xi/(om1+om2) for the frequency range of "
    "interest.",
]

_DAMPING_SIGNAL = (
    "[Cross-Backend][Physics] If a damped transient analysis "
    "matches across backends at low frequency but diverges "
    "wildly at high frequency, the cause is mass-proportional "
    "Rayleigh damping (alpha*M) being applied uniformly to all "
    "modes. Compute alpha + beta from a two-mode formula for "
    "your target frequency range; never pass 'damping ratio' "
    "as a raw alpha."
)


_TIMESTAMP_DESC = (
    "Time-series output files differ across backends in (a) "
    "whether the t=0 frame is written, (b) whether output is "
    "captured at start-of-step or end-of-step (matters for "
    "implicit methods where field is unknown at start), and "
    "(c) the time-stamp metadata precision (single vs double "
    "in some XDMF/VTK writers)."
)

_TIMESTAMP_PITFALLS = [
    "[Cross-Backend][Output] t=0 frame inclusion: 4C's "
    "STRUCTURAL_OUTPUT writes the initial-condition frame at "
    "t=0 by default (controls via RESTART option). FEBio's "
    "plotfile-must-include t=0 default writes t=0. Kratos's "
    "VtkOutput skips the initial frame unless "
    "output_initial_conditions=True is explicit. dolfinx's "
    "VTXWriter writes only when the user calls vtx.write(t); "
    "user-controlled — typical scripts forget to write t=0 "
    "BEFORE the time loop. NGSolve vtk_output: same as "
    "dolfinx — user-controlled. dealii data_out_stack: "
    "user-controlled but every tutorial includes t=0. Signal: "
    "ParaView animation of a 4C output (frame 0 = t=0, frames "
    "1..N = computed steps) shows the initial geometry; same "
    "problem in Kratos VtkOutput (frame 0 = t=dt, no t=0) "
    "shows the post-first-step state as 'initial' — a "
    "visualisation that misleads users into thinking the "
    "initial condition was different. Defense: in Kratos / "
    "dolfinx / NGSolve, explicitly write the t=0 frame before "
    "the time loop starts.",

    "[Cross-Backend][Output] Start-of-step vs end-of-step "
    "field capture: implicit time integration computes the "
    "field at t_{n+1} from the field at t_n. dolfinx, NGSolve, "
    "skfem, dealii: the user calls write(t_{n+1}, u_{n+1}) "
    "AFTER the solve — output is unambiguously at the "
    "computed time. 4C's STRUCTURAL_OUTPUT defaults to "
    "end-of-step (t_{n+1}, u_{n+1}). Kratos's process at the "
    "default <ExecutionPoint>OnEndSolutionStep</ExecutionPoint>: "
    "end-of-step. BUT Kratos's OnBeginSolutionStep writes "
    "BEFORE the solve (t_{n+1}, u_n) — same time-stamp, "
    "previous-step displacement. FEBio's logfile vs plotfile "
    "differ: plotfile end-of-step, logfile may be configured "
    "to either. Signal: a velocity / acceleration field "
    "exported alongside displacement: in dolfinx these are "
    "all evaluated at t_{n+1} using the Newmark update of u; "
    "in a misconfigured Kratos pipeline (OnBeginSolutionStep) "
    "the displacement is stale by one timestep relative to "
    "the velocity, producing a phase lag artifact in ParaView "
    "velocity-vs-displacement plots. Defense: in Kratos, "
    "always use OnEndSolutionStep for field outputs; never "
    "mix Begin and End on the same .vtu series.",
]

_TIMESTAMP_SIGNAL = (
    "[Cross-Backend][Output] Cross-backend ParaView animations "
    "where frame 0 shows the post-first-step state in one "
    "backend but the initial condition in another are caused "
    "by Kratos/dolfinx/NGSolve skipping the t=0 frame "
    "(user-controlled). For phase-lag bugs in velocity-vs-"
    "displacement plots, check Kratos's "
    "OnBeginSolutionStep vs OnEndSolutionStep configuration."
)


_INITIAL_CONDITION_DESC = (
    "Initial-condition specification for transient analyses — "
    "whether u(t=0) is supplied as a per-DOF array, a "
    "Function/CoefficientFunction expression, or an analytic "
    "lambda interpolated onto the mesh — differs across "
    "backends in (a) the API surface, (b) the DEFAULT when "
    "unspecified, and (c) the interpolation quality for "
    "high-order spaces."
)

_INITIAL_CONDITION_PITFALLS = [
    "[Cross-Backend][Physics] Initial-condition default: "
    "dolfinx's u = fem.Function(V); u.x.array[:] = 0 is the "
    "explicit zero-IC; an UN-INITIALISED Function returned "
    "from fem.Function(V) has whatever garbage the PETSc Vec "
    "allocator gave it (typically zeros but NOT GUARANTEED — "
    "depends on PETSc build). NGSolve's GridFunction defaults "
    "to zero (numerically zeroed in C++ ctor). Kratos's "
    "ProcessInfo / nodal variables default to zero. 4C's "
    "transient analyses default ICs to zero unless --INITIAL "
    "CONDITIONS block is present. FEBio's <Initial> XML block "
    "defaults to zero. dealii's Vector<double> ctor zero-"
    "initialises. dolfinx Function INTERPOLATION of an "
    "analytic IC: u.interpolate(lambda x: x[0]**2) for a P2 "
    "FunctionSpace SAMPLES the lambda at the P2 nodal points "
    "(degree-2 interpolation, exact for polynomials up to "
    "deg 2). Kratos's process-based IC application sets only "
    "nodal values (P1-equivalent, even when the element "
    "supports higher order — internal DOFs default to zero). "
    "Signal: ported transient hyperelastic problem with "
    "u(t=0) = sin(pi*x)*sin(pi*y) on a P2 mesh: in dolfinx "
    "the IC is captured to interpolation accuracy O(h^3); in "
    "Kratos (when ported via a process that sets only "
    "nodal values, leaving P2 internal DOFs at zero) the IC "
    "is captured only to O(h^2) AND the internal DOFs that "
    "should hold the polynomial peak are zero, producing a "
    "kink-shaped IC field instead of the smooth sine. "
    "Defense: when porting a non-trivial IC to a P>1 backend, "
    "either (a) use the backend's true interpolation operator "
    "(dolfinx u.interpolate; NGSolve gfu.Set; dealii "
    "VectorTools::interpolate), or (b) confirm the resulting "
    "IC matches the analytic expression at the cell centroid "
    "(not just at vertices) before stepping in time.",
]

_INITIAL_CONDITION_SIGNAL = (
    "[Cross-Backend][Physics] If a transient ported between "
    "backends matches at t=dt but diverges from t=2*dt onward "
    "even with identical solver settings, the cause is "
    "almost always the initial condition was P1-interpolated "
    "in one backend (Kratos process-based IC) vs truly "
    "P-interpolated in another (dolfinx u.interpolate). "
    "Validate IC at cell centroids, not just nodes."
)


_FRAME_OF_REFERENCE_DESC = (
    "Boundary conditions and material orientations specified "
    "in local frames (surface-normal, beam-tangent, "
    "anisotropic principal axes) vs global Cartesian frames "
    "have backend-specific transformation conventions. The "
    "same 'roller constraint normal to the surface' produces "
    "different physical BC in different backends if the "
    "normal-direction computation differs."
)

_FRAME_OF_REFERENCE_PITFALLS = [
    "[Cross-Backend][BC] Local-frame BC computation: 4C's "
    "DBC condition with type='Normal' projects the prescribed "
    "displacement onto the OUTWARD surface normal computed "
    "from the element-face geometry at the START of the "
    "analysis (reference configuration). FEBio's <Surface "
    "load> normal pressure uses the CURRENT-configuration "
    "normal (updated each Newton iteration for finite "
    "deformation). Kratos's RollerConstraintProcess "
    "constrains the DOF along a USER-supplied direction "
    "vector (not auto-normal); if the user supplies (0,0,1) "
    "for a curved surface, it constrains the global z-DOF "
    "everywhere, NOT the local normal — wrong on curved "
    "surfaces. dolfinx: no built-in roller; users compose "
    "via locate_entities_boundary + a custom projection. "
    "Signal: a curved-surface roller-supported plate ported "
    "between 4C (auto reference normal) and Kratos (user "
    "supplies (0,0,1)) gives matching results when the plate "
    "is flat (global z IS the surface normal) but "
    "diverges as the plate curves — Kratos still constrains "
    "global z (wrong physics), 4C constrains true normal. "
    "Defense: for curved-surface rollers in Kratos, "
    "manually compute the per-node normal via "
    "ComputeNodalNormalDivergenceProcess BEFORE applying "
    "RollerConstraintProcess; never assume the user-supplied "
    "direction is the local normal."
]

_FRAME_OF_REFERENCE_SIGNAL = (
    "[Cross-Backend][BC] If a 'roller' or 'normal pressure' "
    "BC produces matching results on flat geometry but "
    "diverges on curved geometry across backends, the cause "
    "is Kratos's user-supplied global-direction roller vs "
    "4C's auto-computed reference-normal roller vs FEBio's "
    "current-config normal pressure. Always verify the BC "
    "by printing the per-node constraint direction on a "
    "test point at the apex of the curvature."
)


_LINEAR_SOLVER_DESC = (
    "Default linear-solver choice, tolerance, iteration cap, "
    "GMRES restart parameter, and direct-solver dispatch path "
    "all differ across backends. Same matrix, same RHS, same "
    "user-stated target precision — different wall-time, "
    "convergence behaviour, or silent fallback solver "
    "depending on which backend's default fires."
)
_LINEAR_SOLVER_PITFALLS = [
    "[Cross-Backend][Solver] Default linear solver dispatch: "
    "fenics/dolfinx's LinearProblem(...).solve() routes "
    "through PETSc KSP with `ksp_type=preonly` + "
    "`pc_type=lu` (MUMPS) for small problems but flips to a "
    "GMRES iterative path with no preconditioner if you "
    "construct the PETSc.Mat by hand and call KSPSolve "
    "directly. skfem's solve(K, b) hits scipy.sparse.linalg."
    "spsolve (SuperLU direct, single-threaded). For a "
    "50k-DOF Poisson problem fenics LinearProblem takes ~3 "
    "s and skfem solve(K,b) takes ~5 s; for 5M-DOF fenics "
    "needs PETScOptions ksp_type=cg + pc_type=hypre to "
    "stay competitive, while skfem requires an explicit "
    "pyamg / petsc4py dispatch (scipy spsolve runs out of "
    "memory). Never trust 'the default linear solver' — "
    "print PETSc.Options() in fenics and inspect "
    "type(K) / scipy version in skfem before benchmarking.",
    "[Cross-Backend][Solver] GMRES restart parameter: PETSc "
    "(used by fenics/dolfinx, dealii, kratos for some "
    "configs) defaults to restart=30 — the Krylov subspace "
    "is rebuilt every 30 iterations. NGSolve's CGSolver / "
    "GMRESSolver does NOT auto-restart; its 'maxit' "
    "parameter is the total iteration cap and divergence on "
    "indefinite systems is silent until the cap fires. The "
    "same matrix can converge in 12 iters under PETSc / "
    "GMRES(30) but stall at 199/200 under NGSolve's non-"
    "restarted GMRES because the Krylov basis becomes "
    "linearly dependent — symptom: 'iter residual' "
    "plateaus around iteration 50 and never drops. Fix: "
    "pass restart=30 explicitly to NGSolve's GMRES, or "
    "wrap NGSolve's iterative loop with manual restart.",
    "[Cross-Backend][Solver] Direct-solver dispatch path: "
    "dealii's SolverDirect(SolverControl) wraps UMFPACK by "
    "default, while 4C's <SOLVER ID=\"1\"> with "
    "TYPE_OF_SOLVER \"UMFPACK\" requires the LSEAux library "
    "to be linked at build time — if it isn't, 4C silently "
    "falls back to a sequential direct solver (Spooles or "
    "Pardiso depending on build flags). For a 100k-DOF "
    "problem dealii's UMFPACK runs single-threaded at ~30 s "
    "while 4C's Pardiso fallback hits 6 s on the same "
    "hardware. Don't compare solver wall-times across "
    "these two without checking each binary's linked-"
    "solver list (`ldd <fourc_binary> | grep -E "
    "'pardiso|spooles|mumps|umfpack'`, and read dealii's "
    "DEAL_II_WITH_UMFPACK / DEAL_II_WITH_MUMPS cmake "
    "summary).",
    "[Cross-Backend][Solver] Tolerance scaling convention: "
    "fenics/dolfinx's PETSc KSP rtol defaults to 1e-5 (the "
    "PETSc default), while NGSolve's bf.Assemble + "
    "Inverse(inverse=\"...\") for iterative paths uses "
    "tol=1e-12 for direct paths and 1e-6 for CG. Kratos's "
    "LinearSolversApplication GMRES has 'tolerance: 1e-7' "
    "in the typical SettingsConstructor. A user copying "
    "solver settings from a fenics tutorial (rtol=1e-5) "
    "into a dealii SolverControl(...) call gets 5x faster "
    "but lower-precision solves than the dealii tutorial "
    "defaults assume. Always set tolerance explicitly in "
    "the script; never inherit it silently.",
    "[Cross-Backend][Solver] Iterative-solver iteration "
    "cap default: dealii's SolverControl(max_iters, "
    "tolerance) requires explicit max_iters with no "
    "default — omitting it raises a constructor error. "
    "PETSc-based (fenics, kratos default) uses 10000 as the "
    "KSP maxits default. skfem's pyamg.solve has "
    "maxiter=300 by default. For an ill-conditioned 1M-DOF "
    "problem fenics spins for 10000 iterations (~minutes) "
    "before reporting failure, while skfem fails fast at "
    "300 — same problem, opposite failure modes. Symptom "
    "in fenics: 'PETSc KSP DIVERGED_ITS' after a long "
    "wall-time; symptom in skfem: pyamg.solve returns with "
    "residual ~1e-3 and no exception (you have to inspect "
    "the residual histories yourself).",
]
_LINEAR_SOLVER_SIGNAL = (
    "[Cross-Backend][Solver] Same matrix, same RHS, same "
    "target precision — wildly different wall-time, "
    "convergence behaviour, or silent fallback to a different "
    "direct solver (e.g. fenics MUMPS vs dealii UMFPACK vs "
    "4C Spooles/Pardiso). Diagnose at runtime by printing "
    "the actual KSP type, preconditioner, restart, "
    "tolerance, maxits, AND linked direct-solver library "
    "(`ksp_view` in fenics, `solver.print()` in dealii, "
    "`ldd` on the fourc binary) BEFORE drawing conclusions "
    "about which backend is 'faster' or 'more accurate' on "
    "the same problem."
)


_NONLINEAR_CONVERGENCE_DESC = (
    "Newton/Picard nonlinear iterations use different "
    "convergence criteria across backends: relative residual "
    "vs absolute residual vs energy norm vs displacement "
    "increment, with or without line search, with different "
    "failure-reporting paths (exception vs tuple-flag vs "
    "exit-code). Same physics, same mesh, the word "
    "'converged' means different things."
)
_NONLINEAR_CONVERGENCE_PITFALLS = [
    "[Cross-Backend][Solver] Newton convergence criterion: "
    "fenics/dolfinx's NewtonSolver checks ||F(u)|| absolute "
    "(residual_norm < tol, default 1e-10) AND ||du|| / "
    "||u|| relative (default 1e-9) — both must hold. "
    "dealii's hand-coded Newton (step-15 pattern) typically "
    "uses ||F(u)|| / ||F(u_0)|| relative residual only. 4C's "
    "NLNSOL_TYPE = 'Full Newton' with NORM_INC = 'Abs' or "
    "'Rel' toggles between the two for the increment norm, "
    "with a separate NORM_RES for the residual. A 'converged' "
    "model in fenics (both criteria at 1e-10) may report "
    "'iter 30: residual=1e-4, not converged' in 4C if "
    "NORM_RES is 'Rel' and TOLINC = 1e-12 — different "
    "criterion entirely.",
    "[Cross-Backend][Solver] Energy-norm convergence: "
    "NGSolve's Newton (ngsolve.solvers.Newton) checks "
    "'energy norm' du.dot(K * du) below tol_energy as one "
    "of its convergence criteria, whereas FEBio's BFGS / "
    "Quasi-Newton uses 'E_tol' = ratio of incremental "
    "energy to initial energy. For a plasticity problem "
    "where the residual is hard to reduce but the energy "
    "stagnates, NGSolve declares convergence early and "
    "FEBio iterates further — same problem, different "
    "'converged' answer. Workaround: always print residual, "
    "increment, AND energy per iteration, and pick the "
    "criterion explicitly rather than relying on the "
    "'default'.",
    "[Cross-Backend][Solver] Initial-residual zero-load "
    "trap: dealii Newton steppers commonly compute the "
    "'initial residual' at iteration 0 with Dirichlet BCs "
    "applied but ZERO loading — yielding ||F_0|| near "
    "machine epsilon. The relative criterion ||F_k|| / "
    "||F_0|| then needs to drop by 10 orders of magnitude "
    "to reach rtol=1e-9 — impossible. fenics's NewtonSolver "
    "guards this with an absolute_tolerance fallback; "
    "dealii's hand-coded steppers usually don't. Symptom: "
    "Newton stalls at iter 3-5 with residual ~1e-8 in "
    "dealii, while fenics's NewtonSolver reports converged "
    "at the same residual.",
    "[Cross-Backend][Solver] Line-search defaults: "
    "fenics/dolfinx's PETSc SNES has line-search='basic' "
    "default (no actual line search, just a damped step). "
    "kratos's ResidualBasedNewtonRaphsonStrategy with "
    "line_search: true uses a backtracking line search. "
    "dealii's SolverNewton::solve has no line search built "
    "in — you implement it. For a buckling problem near "
    "the bifurcation point, fenics's 'basic' SNES can jump "
    "past the bifurcation and Newton diverges in one step; "
    "kratos's backtracking carries through smoothly. Fix: "
    "set line_search='bt' in PETSc SNES options, or set "
    "fenics's SNES to bt explicitly.",
    "[Cross-Backend][Solver] Convergence-failure signal "
    "path: NGSolve's Newton raises a Python exception on "
    "max-iter-exceeded by default; FEBio writes 'Solution "
    "failed to converge' to the .log and exits with code 1 "
    "(no Python-level signal in any wrapper); "
    "fenics/dolfinx's NewtonSolver returns a tuple "
    "(n_iters, converged_bool) without raising. A script "
    "looping over load steps in fenics will silently "
    "continue with stale displacement after a failed step "
    "unless the user checks converged_bool; the same "
    "script crashes immediately in NGSolve; in FEBio there "
    "is no Python-level signal at all (the binary process "
    "exits before any Python wrapper sees it).",
]
_NONLINEAR_CONVERGENCE_SIGNAL = (
    "[Cross-Backend][Solver] 'Converged' means different "
    "things across backends — print all of (||F||, ||du||, "
    "||du||/||u||, energy) per iter, set tolerances "
    "absolutely (not relative-only) to avoid the zero-load "
    "trap, and on convergence FAILURE be sure your script "
    "handles each backend's specific signal: Python "
    "exception (NGSolve), tuple-flag (fenics), process "
    "exit-code (FEBio), KratosError (kratos), or dealii's "
    "ExcSolverFailure. A missing exception does NOT mean "
    "convergence."
)


CROSS_BACKEND_PITFALLS = {
    "units": {
        "description": _UNITS_DESC,
        "pitfalls": _UNITS_PITFALLS,
        "Signal": _UNITS_SIGNAL,
    },
    "element_node_ordering": {
        "description": _NODE_ORDER_DESC,
        "pitfalls": _NODE_ORDER_PITFALLS,
        "Signal": _NODE_ORDER_SIGNAL,
    },
    "linear_elastic_semantics": {
        "description": _LE_SEMANTICS_DESC,
        "pitfalls": _LE_SEMANTICS_PITFALLS,
        "Signal": _LE_SEMANTICS_SIGNAL,
    },
    "dirichlet_bc_enforcement": {
        "description": _DIRICHLET_DESC,
        "pitfalls": _DIRICHLET_PITFALLS,
        "Signal": _DIRICHLET_SIGNAL,
    },
    "restart_checkpoint_compatibility": {
        "description": _RESTART_DESC,
        "pitfalls": _RESTART_PITFALLS,
        "Signal": _RESTART_SIGNAL,
    },
    "mpi_launch_idioms": {
        "description": _MPI_DESC,
        "pitfalls": _MPI_PITFALLS,
        "Signal": _MPI_SIGNAL,
    },
    "element_type_naming": {
        "description": _ELEMENT_TYPE_DESC,
        "pitfalls": _ELEMENT_TYPE_PITFALLS,
        "Signal": _ELEMENT_TYPE_SIGNAL,
    },
    "time_integration_defaults": {
        "description": _TIME_INTEGRATION_DESC,
        "pitfalls": _TIME_INTEGRATION_PITFALLS,
        "Signal": _TIME_INTEGRATION_SIGNAL,
    },
    "solver_tolerance_defaults": {
        "description": _TOLERANCE_DESC,
        "pitfalls": _TOLERANCE_PITFALLS,
        "Signal": _TOLERANCE_SIGNAL,
    },
    "contact_formulation_defaults": {
        "description": _CONTACT_FORMULATION_DESC,
        "pitfalls": _CONTACT_FORMULATION_PITFALLS,
        "Signal": _CONTACT_FORMULATION_SIGNAL,
    },
    "output_format_conventions": {
        "description": _OUTPUT_FORMAT_DESC,
        "pitfalls": _OUTPUT_FORMAT_PITFALLS,
        "Signal": _OUTPUT_FORMAT_SIGNAL,
    },
    "integration_order_defaults": {
        "description": _INTEGRATION_ORDER_DESC,
        "pitfalls": _INTEGRATION_ORDER_PITFALLS,
        "Signal": _INTEGRATION_ORDER_SIGNAL,
    },
    "boundary_tag_semantics": {
        "description": _BOUNDARY_TAG_DESC,
        "pitfalls": _BOUNDARY_TAG_PITFALLS,
        "Signal": _BOUNDARY_TAG_SIGNAL,
    },
    "plasticity_return_mapping": {
        "description": _PLASTICITY_RETURN_MAP_DESC,
        "pitfalls": _PLASTICITY_RETURN_MAP_PITFALLS,
        "Signal": _PLASTICITY_RETURN_MAP_SIGNAL,
    },
    "turbulence_model_defaults": {
        "description": _TURBULENCE_DESC,
        "pitfalls": _TURBULENCE_PITFALLS,
        "Signal": _TURBULENCE_SIGNAL,
    },
    "material_orientation_defaults": {
        "description": _MATERIAL_ORIENTATION_DESC,
        "pitfalls": _MATERIAL_ORIENTATION_PITFALLS,
        "Signal": _MATERIAL_ORIENTATION_SIGNAL,
    },
    "frequency_unit_conventions": {
        "description": _FREQUENCY_DESC,
        "pitfalls": _FREQUENCY_PITFALLS,
        "Signal": _FREQUENCY_SIGNAL,
    },
    "mesh_quality_thresholds": {
        "description": _MESH_QUALITY_DESC,
        "pitfalls": _MESH_QUALITY_PITFALLS,
        "Signal": _MESH_QUALITY_SIGNAL,
    },
    "stress_measure_conventions": {
        "description": _STRESS_MEASURE_DESC,
        "pitfalls": _STRESS_MEASURE_PITFALLS,
        "Signal": _STRESS_MEASURE_SIGNAL,
    },
    "periodic_bc_implementation": {
        "description": _PERIODIC_BC_DESC,
        "pitfalls": _PERIODIC_BC_PITFALLS,
        "Signal": _PERIODIC_BC_SIGNAL,
    },
    "damping_convention_defaults": {
        "description": _DAMPING_DESC,
        "pitfalls": _DAMPING_PITFALLS,
        "Signal": _DAMPING_SIGNAL,
    },
    "timestamp_output_conventions": {
        "description": _TIMESTAMP_DESC,
        "pitfalls": _TIMESTAMP_PITFALLS,
        "Signal": _TIMESTAMP_SIGNAL,
    },
    "initial_condition_interpolation": {
        "description": _INITIAL_CONDITION_DESC,
        "pitfalls": _INITIAL_CONDITION_PITFALLS,
        "Signal": _INITIAL_CONDITION_SIGNAL,
    },
    "frame_of_reference_bc": {
        "description": _FRAME_OF_REFERENCE_DESC,
        "pitfalls": _FRAME_OF_REFERENCE_PITFALLS,
        "Signal": _FRAME_OF_REFERENCE_SIGNAL,
    },
    "linear_solver_defaults": {
        "description": _LINEAR_SOLVER_DESC,
        "pitfalls": _LINEAR_SOLVER_PITFALLS,
        "Signal": _LINEAR_SOLVER_SIGNAL,
    },
    "nonlinear_convergence_criteria": {
        "description": _NONLINEAR_CONVERGENCE_DESC,
        "pitfalls": _NONLINEAR_CONVERGENCE_PITFALLS,
        "Signal": _NONLINEAR_CONVERGENCE_SIGNAL,
    },
}


def get_cross_backend_pitfalls(topic: str | None = None) -> dict:
    """Return the cross-backend pitfalls structure.

    If `topic` is provided (e.g. 'units', 'mesh', 'bc'), filter to
    matching entries. Topic matching is a case-insensitive substring
    against the entry key or description.
    """
    if not topic:
        return CROSS_BACKEND_PITFALLS
    t = topic.lower()
    matched = {}
    for key, entry in CROSS_BACKEND_PITFALLS.items():
        if t in key.lower() or t in entry.get("description", "").lower():
            matched[key] = entry
    return matched if matched else CROSS_BACKEND_PITFALLS
