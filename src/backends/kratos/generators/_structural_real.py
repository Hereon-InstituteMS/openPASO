"""The REAL Kratos StructuralMechanics route, verified by execution.

Same defect, same fix as _convdiff_real: linear_elasticity.py,
structural_dynamics.py, cosimulation.py and shape_optimization.py all emitted a
numpy/scipy assembly and never imported KratosMultiphysics — their own first
lines said "(manual assembly)" or "(standalone)". An agent asked to solve with
Kratos got an artefact that cannot be attributed to Kratos.

Probed on this install (Kratos 10.3.0), all present:
    SmallDisplacementElement2D3N / 2D4N, TotalLagrangianElement2D3N,
    UpdatedLagrangianElement2D3N, LinearTrussElement2D2N, ShellThinElement3D3N
    LineLoadCondition2D2N, PointLoadCondition2D1N, SurfaceLoadCondition3D3N
    LinearElasticPlaneStrain2DLaw / PlaneStress2DLaw / LinearElastic3DLaw
and the load variables are on the APPLICATION module, not the core one:
SMA.LINE_LOAD / SMA.POINT_LOAD / SMA.SURFACE_LOAD, while
KM.VOLUME_ACCELERATION / KM.THICKNESS / KM.CONSTITUTIVE_LAW are on KM.
"""
from __future__ import annotations

_REAL = '''\
"""{title}

Solved by Kratos Multiphysics StructuralMechanicsApplication. The solver is
Kratos; nothing here assembles the system. Run it with `run_simulation`.
"""
import json

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
# ^ the import REGISTERS SmallDisplacementElement2D*N; without it
#   CreateNewElement raises 'The Element ... is not registered!'

NX, NY = {nx}, {ny}
LX, LY = {lx}, {ly}
E, NU, RHO = {young}, {nu}, {rho}
TRACTION = {traction}          # per unit length on the right edge, [tx, ty]
PLANE = "{plane}"              # "strain" or "stress"

xs = np.linspace(0.0, LX, NX + 1)
ys = np.linspace(0.0, LY, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731  0-based
quads = [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
         for j in range(NY) for i in range(NX)]

model = KM.Model()
mp = model.CreateModelPart("structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
# LINE_LOAD / POINT_LOAD / SURFACE_LOAD live on SMA, NOT on KM (measured:
# KM.LINE_LOAD raises "Module KratosMultiphysics has no attribute LINE_LOAD").
# VOLUME_ACCELERATION, THICKNESS and CONSTITUTIVE_LAW are on KM.
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION,
          SMA.LINE_LOAD, SMA.POINT_LOAD):
    mp.AddNodalSolutionStepVariable(v)

prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, NU)
prop.SetValue(KM.DENSITY, RHO)
prop.SetValue(KM.THICKNESS, 1.0)
# THE CONSTITUTIVE LAW IS NOT OPTIONAL and it is not implied by the element:
# without CONSTITUTIVE_LAW on the Properties the element has no material and
# Initialize() fails. Plane strain and plane stress are DIFFERENT laws, not a
# flag -- picking the wrong one is a silent stiffness error.
law = (SMA.LinearElasticPlaneStrain2DLaw() if PLANE == "strain"
       else SMA.LinearElasticPlaneStress2DLaw())
prop.SetValue(KM.CONSTITUTIVE_LAW, law)

for i, (px, py) in enumerate(nodes):
    mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
for e, el in enumerate(quads):
    mp.CreateNewElement("SmallDisplacementElement2D4N", e + 1,
                        [int(v) + 1 for v in el], prop)

# the loaded edge, as line-load conditions carrying LINE_LOAD on their nodes
tol = 1e-12
right = [i for i in range(len(nodes)) if abs(nodes[i, 0] - LX) < tol]
right.sort(key=lambda i: nodes[i, 1])
for c in range(len(right) - 1):
    mp.CreateNewCondition("LineLoadCondition2D2N", c + 1,
                          [int(right[c]) + 1, int(right[c + 1]) + 1], prop)
for i in right:
    mp.GetNode(int(i) + 1).SetSolutionStepValue(
        SMA.LINE_LOAD, KM.Vector([float(TRACTION[0]), float(TRACTION[1]), 0.0]))

KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)

# clamp the left edge: BOTH components, or the body is under-constrained and
# the solve returns a rigid-body mode on top of the answer
for i in range(len(nodes)):
    if abs(nodes[i, 0]) < tol:
        nd = mp.GetNode(i + 1)
        nd.Fix(KM.DISPLACEMENT_X)
        nd.Fix(KM.DISPLACEMENT_Y)

lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
bas = KM.ResidualBasedBlockBuilderAndSolver(lin)
strat = KM.ResidualBasedLinearStrategy(mp, scheme, bas, True, False, False,
                                       False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = 1.0
mp.CloneTimeStep(1.0)
strat.Initialize()
strat.Solve()

u = np.array([[mp.GetNode(i + 1).GetSolutionStepValue(KM.DISPLACEMENT_X),
               mp.GetNode(i + 1).GetSolutionStepValue(KM.DISPLACEMENT_Y)]
              for i in range(len(nodes))])
print(f"Kratos SmallDisplacementElement2D4N: nodes={{len(nodes)}} "
      f"elements={{len(quads)}} max|u|={{np.abs(u).max():.10e}}")
if np.abs(u).max() == 0.0 and any(abs(t) > 0 for t in TRACTION):
    raise SystemExit("Kratos returned zero displacement under a nonzero "
                     "traction: the load set is empty. Check that every "
                     "loaded-edge node carries LINE_LOAD and that the "
                     "conditions were created on those node pairs.")

np.savetxt("solution.csv",
           np.column_stack([nodes[:, 0], nodes[:, 1], u[:, 0], u[:, 1]]),
           delimiter=", ", header="x, y, ux, uy", comments="", fmt="%.15e")
json.dump({{"max_abs_displacement": float(np.abs(u).max()),
           "n_nodes": int(len(nodes)), "n_elements": int(len(quads)),
           "element": "SmallDisplacementElement2D4N",
           "law": f"LinearElasticPlane{{PLANE.capitalize()}}2DLaw",
           "solver": "Kratos StructuralMechanicsApplication"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos structural solve complete.")
'''


def real_structural_script(title: str, nx: int, ny: int, lx: float, ly: float,
                           young: float, nu: float, rho: float = 0.0,
                           traction=(0.0, -1.0),
                           plane: str = "strain") -> str:
    """A runnable Kratos small-displacement elasticity script."""
    return _REAL.format(title=title, nx=nx, ny=ny, lx=lx, ly=ly, young=young,
                        nu=nu, rho=rho, traction=list(traction), plane=plane)


_DYN = '''\
"""{title}

Transient structural dynamics solved by Kratos Multiphysics
(StructuralMechanicsApplication + ResidualBasedBossakDisplacementScheme). The
time integration is Kratos's; nothing here assembles or steps the system.
"""
import json

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

NX, NY = {nx}, {ny}
LX, LY = {lx}, {ly}
E, NU, RHO = {young}, {nu}, {rho}
DT, T_END = {dt}, {t_end}
TRACTION = {traction}
PLANE = "{plane}"

xs = np.linspace(0.0, LX, NX + 1)
ys = np.linspace(0.0, LY, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731
quads = [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
         for j in range(NY) for i in range(NX)]

model = KM.Model()
mp = model.CreateModelPart("structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
# A DYNAMIC SCHEME NEEDS VELOCITY AND ACCELERATION IN THE VARIABLES LIST.
# Registering only DISPLACEMENT lets the model build and then fails inside the
# scheme with "the variables list doesn't have this variable" -- in every
# worker thread, so the traceback arrives many times and names no line of
# yours.
for v in (KM.DISPLACEMENT, KM.VELOCITY, KM.ACCELERATION, KM.REACTION,
          KM.VOLUME_ACCELERATION, SMA.LINE_LOAD, SMA.POINT_LOAD):
    mp.AddNodalSolutionStepVariable(v)

prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, NU)
prop.SetValue(KM.DENSITY, RHO)        # RHO must be NONZERO or there is no mass
prop.SetValue(KM.THICKNESS, 1.0)
prop.SetValue(KM.CONSTITUTIVE_LAW,
              SMA.LinearElasticPlaneStrain2DLaw() if PLANE == "strain"
              else SMA.LinearElasticPlaneStress2DLaw())

for i, (px, py) in enumerate(nodes):
    mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
for e, el in enumerate(quads):
    mp.CreateNewElement("SmallDisplacementElement2D4N", e + 1,
                        [int(v) + 1 for v in el], prop)

tol = 1e-12
right = sorted([i for i in range(len(nodes)) if abs(nodes[i, 0] - LX) < tol],
               key=lambda i: nodes[i, 1])
for c in range(len(right) - 1):
    mp.CreateNewCondition("LineLoadCondition2D2N", c + 1,
                          [int(right[c]) + 1, int(right[c + 1]) + 1], prop)
for i in right:
    mp.GetNode(int(i) + 1).SetSolutionStepValue(
        SMA.LINE_LOAD, KM.Vector([float(TRACTION[0]), float(TRACTION[1]), 0.0]))

KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
for i in range(len(nodes)):
    if abs(nodes[i, 0]) < tol:
        nd = mp.GetNode(i + 1)
        nd.Fix(KM.DISPLACEMENT_X)
        nd.Fix(KM.DISPLACEMENT_Y)

# THE BUFFER MUST HOLD THE SCHEME'S HISTORY. Bossak reads step n-1; the default
# buffer of 1 gives it only the current step and the response comes out
# without any inertia in it.
mp.SetBufferSize(3)
lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
scheme = KM.ResidualBasedBossakDisplacementScheme(-0.3)
conv = KM.DisplacementCriteria(1e-10, 1e-12)
conv.SetEchoLevel(0)
strat = KM.ResidualBasedNewtonRaphsonStrategy(mp, scheme, conv,
                                              KM.ResidualBasedBlockBuilderAndSolver(lin),
                                              20, True, False, False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = DT
strat.Initialize()

t, tip, hist = 0.0, int(right[len(right) // 2]), []
while t < T_END - 1e-12:
    t += DT
    mp.CloneTimeStep(t)
    strat.Solve()
    uy = mp.GetNode(tip + 1).GetSolutionStepValue(KM.DISPLACEMENT_Y)
    hist.append((t, float(uy)))
print(f"Kratos Bossak, {{len(hist)}} steps: tip uy first={{hist[0][1]:.6e}} "
      f"last={{hist[-1][1]:.6e}} max|uy|={{max(abs(v) for _, v in hist):.6e}}")
# A DYNAMIC RUN WHOSE HISTORY NEVER MOVES IS A STATIC SOLVE REPEATED.
if len(hist) > 2 and max(abs(hist[k][1] - hist[0][1]) for k in range(len(hist))) == 0.0:
    raise SystemExit("the tip displacement is identical at every step: the "
                     "mass or the scheme is not in play. Check DENSITY != 0, "
                     "that VELOCITY and ACCELERATION are in the variables "
                     "list, and that SetBufferSize(3) was called.")

np.savetxt("history.csv", np.array(hist), delimiter=", ",
           header="t, tip_uy", comments="", fmt="%.15e")
json.dump({{"n_steps": len(hist), "dt": DT, "t_end": T_END,
           "max_abs_tip_uy": float(max(abs(v) for _, v in hist)),
           "scheme": "ResidualBasedBossakDisplacementScheme(-0.3)",
           "element": "SmallDisplacementElement2D4N",
           "solver": "Kratos StructuralMechanicsApplication"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos transient solve complete.")
'''


def real_dynamics_script(title: str, nx: int, ny: int, lx: float, ly: float,
                         young: float, nu: float, rho: float, dt: float,
                         t_end: float, traction=(0.0, -1.0),
                         plane: str = "strain") -> str:
    """A runnable Kratos transient structural-dynamics script."""
    return _DYN.format(title=title, nx=nx, ny=ny, lx=lx, ly=ly, young=young,
                       nu=nu, rho=rho, dt=dt, t_end=t_end,
                       traction=list(traction), plane=plane)


_NONLIN = '''\
"""{title}

Geometrically nonlinear elasticity solved by Kratos Multiphysics:
TotalLagrangianElement2D3N with a KirchhoffSaintVenantPlaneStrain2DLaw, driven
by ResidualBasedNewtonRaphsonStrategy over load steps. The nonlinear solve is
Kratos's; nothing here assembles or iterates.
"""
import json

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
# THE ST-VENANT-KIRCHHOFF AND HYPERELASTIC LAWS ARE IN A THIRD APPLICATION.
# Measured on this install: SMA carries only the LINEAR laws
# (LinearElastic{{3D,PlaneStrain2D,PlaneStress2D,Axisym2D}}Law plus beam/truss).
# KirchhoffSaintVenant*, HyperElastic* and every plasticity/damage law live in
# ConstitutiveLawsApplication, so `SMA.KirchhoffSaintVenantPlaneStrain2DLaw`
# raises AttributeError and a nonlinear run built on a LINEAR law is a silent
# small-strain answer wearing a finite-strain element.
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

NX, NY = {nx}, {ny}
LX, LY = {lx}, {ly}
E, NU = {young}, {nu}
TRACTION = {traction}
N_LOAD_STEPS = {n_steps}

xs = np.linspace(0.0, LX, NX + 1)
ys = np.linspace(0.0, LY, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731
tris = []
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
        tris += [[a, b, c], [a, c, d]]

model = KM.Model()
mp = model.CreateModelPart("structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION,
          SMA.LINE_LOAD, SMA.POINT_LOAD):
    mp.AddNodalSolutionStepVariable(v)

prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, NU)
prop.SetValue(KM.DENSITY, 0.0)
prop.SetValue(KM.THICKNESS, 1.0)
prop.SetValue(KM.CONSTITUTIVE_LAW, CLA.KirchhoffSaintVenantPlaneStrain2DLaw())

for i, (px, py) in enumerate(nodes):
    mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
for e, el in enumerate(tris):
    mp.CreateNewElement("TotalLagrangianElement2D3N", e + 1,
                        [int(v) + 1 for v in el], prop)

tol = 1e-12
right = sorted([i for i in range(len(nodes)) if abs(nodes[i, 0] - LX) < tol],
               key=lambda i: nodes[i, 1])
for c in range(len(right) - 1):
    mp.CreateNewCondition("LineLoadCondition2D2N", c + 1,
                          [int(right[c]) + 1, int(right[c + 1]) + 1], prop)

KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
for i in range(len(nodes)):
    if abs(nodes[i, 0]) < tol:
        nd = mp.GetNode(i + 1)
        nd.Fix(KM.DISPLACEMENT_X)
        nd.Fix(KM.DISPLACEMENT_Y)

lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
conv = KM.DisplacementCriteria(1e-9, 1e-12)
conv.SetEchoLevel(0)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, KM.ResidualBasedIncrementalUpdateStaticScheme(), conv,
    KM.ResidualBasedBlockBuilderAndSolver(lin), 30, True, False, False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = 1.0
strat.Initialize()

# RAMP THE LOAD. A single jump to the full finite-strain load stalls the
# Newton iteration; the load steps are what make it converge, and they are
# also what lets the run show the nonlinearity as a non-proportional
# displacement-versus-load curve.
tip = int(right[len(right) // 2])
hist = []
for step in range(1, N_LOAD_STEPS + 1):
    lam = step / N_LOAD_STEPS
    for i in right:
        mp.GetNode(int(i) + 1).SetSolutionStepValue(
            SMA.LINE_LOAD, KM.Vector([float(TRACTION[0]) * lam,
                                      float(TRACTION[1]) * lam, 0.0]))
    mp.CloneTimeStep(float(step))
    strat.Solve()
    uy = mp.GetNode(tip + 1).GetSolutionStepValue(KM.DISPLACEMENT_Y)
    hist.append((lam, float(uy)))
    print(f"load factor {{lam:.3f}}: tip uy = {{uy:.8e}}")

lin_pred = hist[0][1] / hist[0][0] * hist[-1][0]
print(f"Kratos TotalLagrangianElement2D3N: {{len(hist)}} load steps, "
      f"tip uy = {{hist[-1][1]:.8e}}, a linear extrapolation of the first step "
      f"would give {{lin_pred:.8e}}")
if len(hist) > 2 and abs(hist[-1][1] - lin_pred) <= 0.0:
    raise SystemExit("the response is exactly proportional to the load: the "
                     "run is linear. Check that the element is "
                     "TotalLagrangian and the law comes from "
                     "ConstitutiveLawsApplication, not a LinearElastic law.")

np.savetxt("load_history.csv", np.array(hist), delimiter=", ",
           header="load_factor, tip_uy", comments="", fmt="%.15e")
json.dump({{"n_load_steps": len(hist), "tip_uy_final": hist[-1][1],
           "linear_extrapolation_of_first_step": float(lin_pred),
           "element": "TotalLagrangianElement2D3N",
           "law": "KirchhoffSaintVenantPlaneStrain2DLaw "
                  "(ConstitutiveLawsApplication)",
           "solver": "Kratos StructuralMechanicsApplication"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos nonlinear solve complete.")
'''


def real_nonlinear_script(title: str, nx: int, ny: int, lx: float, ly: float,
                          young: float, nu: float, traction=(0.0, -1.0),
                          n_steps: int = 5) -> str:
    """A runnable Kratos geometrically-nonlinear elasticity script."""
    return _NONLIN.format(title=title, nx=nx, ny=ny, lx=lx, ly=ly,
                          young=young, nu=nu, traction=list(traction),
                          n_steps=n_steps)
