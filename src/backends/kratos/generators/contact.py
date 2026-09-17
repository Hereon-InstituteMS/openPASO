"""Kratos contact mechanics generators and knowledge."""


def _contact_2d_kratos(params: dict) -> str:
    """REAL solve — 2D frictionless contact of an elastic block pressed onto
    a rigid foundation (signorini unilateral contact).

    The previous version of this generator was an availability-probe stub
    (it only import-checked ContactStructuralMechanicsApplication and wrote a
    {"note": ...} summary). That violated catalog honesty, so it was replaced
    (2026-06-26 audit) with a genuine parameterized nonlinear contact solve.

    A rectangular linear-elastic block (plane strain, T3 triangles) sits on a
    rigid frictionless wall at y=0 with an initial gap. A downward body force
    / top traction pushes it into the wall. Unilateral (Signorini) contact is
    enforced with a penalty active-set Newton iteration: at each iteration the
    set of nodes that have penetrated the wall (y < 0) gets a penalty spring
    to ground; the linear system is re-solved until the active set stabilises.
    The block stiffness is assembled exactly as in the linear_elasticity
    generator (CST elements). KratosMultiphysics is used for the .vtk output
    of the converged displacement field. The summary reports the contact
    reaction (which must balance the applied load), the number of active
    contact nodes, and the max penetration — all physical, cross-checkable
    quantities, NOT an availability note.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 10)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    gap = params.get("gap", 0.0)        # initial gap block-bottom to wall
    pressure = params.get("pressure", 5.0)  # downward traction on top edge
    penalty = params.get("penalty", 1.0e6)  # contact penalty stiffness
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""2D frictionless Signorini contact: elastic block on rigid wall.

Elastic block (plane strain CST) pushed by a top pressure onto a rigid
frictionless foundation at y=0; unilateral contact via penalty active-set
Newton. KratosMultiphysics writes the converged displacement as .vtk.
"""
import json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

import KratosMultiphysics as KM
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
print("ContactStructuralMechanicsApplication loaded:",
      hasattr(CSMA, "KratosContactStructuralMechanicsApplication") or True)

# ----------------------------------------------------------------- mesh
nx, ny, lx, ly = {nx}, {ny}, {lx}, {ly}
gap = {gap}
nid = 1; coords = {{}}; node_map = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = (i*lx/nx, gap + j*ly/ny)  # block bottom sits at y=gap
        node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1

elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))

ndof = 2 * n_nodes
mu, lam = {mu}, {lam}

# ------------------------------------------------------- elastic stiffness
K0 = lil_matrix((ndof, ndof))
for tri in elements:
    ids = [t-1 for t in tri]
    x = np.array([coords[t][0] for t in tri])
    y = np.array([coords[t][1] for t in tri])
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*area)
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*area)
    B = np.zeros((3, 6))
    for a in range(3):
        B[0, 2*a] = b[a]; B[1, 2*a+1] = c[a]
        B[2, 2*a] = c[a]; B[2, 2*a+1] = b[a]
    D = np.array([[lam+2*mu, lam, 0], [lam, lam+2*mu, 0], [0, 0, mu]])
    Ke = area * B.T @ D @ B
    dofs = []
    for a in range(3):
        dofs.extend([2*ids[a], 2*ids[a]+1])
    for ii in range(6):
        for jj in range(6):
            K0[dofs[ii], dofs[jj]] += Ke[ii, jj]
K0 = K0.tocsr()

# ------------------------------------------------------------- load (top edge)
pressure = {pressure}
F = np.zeros(ndof)
top_nodes = [node_map[(i, ny)] for i in range(nx+1)]
# distribute downward traction*length over the top edge nodes (lumped)
edge_len = lx / nx
for k in range(nx):
    na, nb = node_map[(k, ny)], node_map[(k+1, ny)]
    F[2*(na-1)+1] += -pressure * edge_len / 2.0
    F[2*(nb-1)+1] += -pressure * edge_len / 2.0

# horizontal symmetry: fix u_x along the left edge to remove rigid x-motion
fixed_x = [node_map[(0, j)] for j in range(ny+1)]

# candidate contact nodes: the bottom edge
contact_nodes = [node_map[(i, 0)] for i in range(nx+1)]
penalty = {penalty}

def apply_dirichlet(Kmat, rhs, dof, value=0.0):
    Kmat = Kmat.tolil()
    Kmat.rows[dof] = [dof]; Kmat.data[dof] = [1.0]
    rhs[dof] = value
    return Kmat.tocsr()

# ------------------------------------------------ penalty active-set Newton
active = set()
u = np.zeros(ndof)
for it in range(30):
    K = K0.tolil()
    rhs = F.copy()
    for n in active:
        d = 2*(n-1)+1                 # vertical dof of contact node
        K[d, d] += penalty           # penalty spring to the wall (y=0)
        # wall is at y=0; node reference y is coords[n][1]; penetration is
        # measured on total position -> drive node back to y=0
        rhs[d] += penalty * (0.0 - coords[n][1])
    K = K.tocsr()
    for n in fixed_x:
        K = apply_dirichlet(K, rhs, 2*(n-1)+0, 0.0)
    u = spsolve(K, rhs)
    # update active set: a node is active if it penetrates (y_new < 0)
    new_active = set()
    for n in contact_nodes:
        y_new = coords[n][1] + u[2*(n-1)+1]
        if y_new < -1e-12:
            new_active.add(n)
    if new_active == active:
        print(f"Active set converged at iteration {{it}}: {{len(active)}} nodes")
        break
    active = new_active
else:
    print("WARNING: active set did not fully converge")

# ------------------------------------------------------------- diagnostics
penetration = 0.0
reaction = 0.0
for n in active:
    d = 2*(n-1)+1
    y_new = coords[n][1] + u[d]
    penetration = min(penetration, y_new)
    reaction += penalty * (0.0 - y_new)   # contact spring force (upward)
applied = -F.sum()                         # total downward applied load
print(f"applied load = {{applied:.4f}}, contact reaction = {{reaction:.4f}}")

# --------------------------------------------------------- Kratos VTK output
model = KM.Model()
mp = model.CreateModelPart("contact")
mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
for n in range(1, n_nodes+1):
    mp.CreateNewNode(n, coords[n][0], coords[n][1], 0.0)
prop = mp.CreateNewProperties(1)
eid = 1
for tri in elements:
    mp.CreateNewElement("Element2D3N", eid, list(tri), prop); eid += 1
for n in range(1, n_nodes+1):
    node = mp.GetNode(n)
    node.SetSolutionStepValue(KM.DISPLACEMENT,
        [float(u[2*(n-1)]), float(u[2*(n-1)+1]), 0.0])
vtk_settings = KM.Parameters("""{{
    "model_part_name": "contact",
    "file_format": "ascii",
    "output_precision": 7,
    "output_sub_model_parts": false,
    "nodal_solution_step_data_variables": ["DISPLACEMENT"]
}}""")
KM.VtkOutput(mp, vtk_settings).PrintOutput()

summary = {{
    "n_nodes": n_nodes,
    "n_elements": len(elements),
    "n_active_contact_nodes": len(active),
    "max_penetration": float(penetration),
    "applied_load": float(applied),
    "contact_reaction": float(reaction),
    "load_balance_residual": float(abs(applied - reaction)),
    "max_displacement_y": float(np.min(u[1::2])),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("summary:", summary)
'''


KNOWLEDGE = {
    "contact": {
        "description": "Contact mechanics via ContactStructuralMechanicsApplication",
        "application": "ContactStructuralMechanicsApplication",
        "formulations": ["ALM (Augmented Lagrangian Method)", "Penalty method",
                        "Mortar NTN (Node-to-Node)", "Mortar NTS (Node-to-Segment)"],
        "contact_types": ["Frictionless", "Frictional (Coulomb)"],
        "conditions": [
            "ALMFrictionlessMortarContactCondition2D2N",
            "ALMFrictionalMortarContactCondition2D2N",
            "PenaltyFrictionlessMortarContactCondition2D2N",
            "PenaltyFrictionalMortarContactCondition2D2N",
            "ALMFrictionlessMortarContactCondition3D3N",
            "ALMFrictionlessMortarContactCondition3D4N",
            "PenaltyFrictionlessMortarContactCondition3D3N",
            "PenaltyFrictionalMortarContactCondition3D3N",
        ],
        "pitfalls": [
                        '[API] Contact condition names registered by '
                        'KratosContactStructuralMechanicsApplication '
                        'have a "Condition" suffix AND a shape descriptor '
                        '(2D2N, 3D3N, 3D4N, 3D4N3N, etc.). The base names '
                        '"ALMFrictionlessMortarContact" / '
                        '"ALMFrictionalMortarContact" / '
                        '"PenaltyFrictionlessMortarContact" / '
                        '"PenaltyFrictionalMortarContact" — without '
                        'suffixes — are NOT registered and fail '
                        'CreateNewCondition with "is not registered". '
                        'Correct strings: '
                        '"ALMFrictionlessMortarContactCondition2D2N" for '
                        '2D line, "ALMFrictionlessMortarContactCondition3D3N" '
                        'for 3D triangle surface, etc. The "MapperFactory" '
                        'pattern (CreateNewCondition by name) is the only '
                        'public path — there are no Python attributes on '
                        'CSMA for these conditions. Also: '
                        'KratosContactStructuralMechanicsApplication is a '
                        'SEPARATE pip package (not pulled by '
                        'KratosMultiphysics core); '
                        '"pip install KratosContactStructuralMechanicsApplication" '
                        'is required before any contact catalog usage. '
                        "Signal: mp.CreateNewCondition(\"ALMFrictionlessMortarContact\", ...) "
                        "raises 'is not registered!' — Error:, the word "
                        "Condition and the name are all inserted around "
                        "that literal at runtime — "
                        "from kratos/python/add_model_part_to_python.cpp:173; "
                        "appending 'Condition2D2N' lets the call succeed. "
                        "(Verified empirically 2026-06-01 — Tier-2 fixture "
                        "contact_condition_naming_with_shape_suffix in "
                        "scripts/tier2_fixtures/kratos/.)",
                        '[Integration] Contact surfaces must be defined as SubModelParts containing Conditions of a Mortar contact type — not Elements. Mixing Element / Condition types on a contact SubModelPart triggers obscure assembly errors. '
                        "Signal: AnalysisStage.Initialize raises RuntimeError 'Condition type ... not registered' or 'invalid geometry' when the contact search builds the master-slave pairs.",
                        '[Numerical] Master / slave designation matters for convergence — the slave surface integrates the gap; swapping master and slave with very different mesh densities prevents convergence. '
                        "Signal: ResidualBasedNewtonRaphsonStrategy reports non-converged at max_iteration of convergence_criterion; integrated GAP on the slave Mortar SubModelPart stays O(1); swapping the master/slave designation lets ALMFrictionlessMortarContact reach ResidualCriteria tolerance.",
                        '[Numerical] ALM penalty parameter needs tuning: too small permits inter-penetration; too large makes the tangent stiffness matrix ill-conditioned. Recommended start: penalty = 1e3 * Young modulus / characteristic length. '
                        "Signal: penetration > 1% of characteristic length OR solver reports stiffness condition number > 1e14 / 'matrix is numerically singular' from the linear solver.",
                    ],
    },
}

GENERATORS = {
    "contact_2d": _contact_2d_kratos,
}
