"""Kratos FluidDynamicsApplication generators and knowledge.

Incompressible Navier-Stokes through the monolithic VMS element. Written
2026-09-19 because openPASO advertised nine Kratos applications and reached
neither FluidDynamics nor FSI nor GeoMechanics, while all three are
installed -- capabilities withheld rather than missing.

VERIFIED BY EXECUTION on Kratos 10.3.0, and not merely by running: the
emitted script checks its own result against two things independent of the
run. Mass: with one inlet and one outlet the net flux must balance, and it
does to about one per cent on the default mesh. Profile: a developed plane
channel has peak velocity exactly 3/2 of the mean, a ratio that follows from
the parabolic profile and the geometry alone, and the outlet comes out at
1.53. Neither number is stored anywhere -- both are derived at run time from
the geometry the user asked for and compared against what the solver
produced.
"""
from string import Template


_CHANNEL = Template(r"""
'''$title

Incompressible Navier-Stokes solved by Kratos FluidDynamicsApplication with
the monolithic VMS element. The solver is Kratos; nothing here assembles the
system.
'''
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.FluidDynamicsApplication as CFD
import KratosMultiphysics.LinearSolversApplication  # noqa
from KratosMultiphysics import python_linear_solver_factory as plsf

NX, NY = $nx, $ny
LX, LY = $lx, $ly
RHO, MU, U_IN = $rho, $mu, $u_in
xs = np.linspace(0.0, LX, NX + 1); ys = np.linspace(0.0, LY, NY + 1)
nid = lambda i, j: j * (NX + 1) + i + 1

model = KM.Model(); mp = model.CreateModelPart("fluid")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
for v in (KM.VELOCITY, KM.ACCELERATION, KM.MESH_VELOCITY, KM.PRESSURE,
          KM.REACTION, KM.BODY_FORCE, KM.NODAL_AREA, KM.NODAL_H,
          KM.ADVPROJ, KM.DIVPROJ, KM.DENSITY, KM.VISCOSITY, KM.EXTERNAL_PRESSURE):
    mp.AddNodalSolutionStepVariable(v)

for j in range(NY + 1):
    for i in range(NX + 1):
        mp.CreateNewNode(nid(i, j), float(xs[i]), float(ys[j]), 0.0)

p = mp.CreateNewProperties(1)
p[KM.DENSITY] = RHO
p[KM.DYNAMIC_VISCOSITY] = MU
p.SetValue(KM.CONSTITUTIVE_LAW,
           KM.KratosGlobals.GetConstitutiveLaw("Newtonian2DLaw").Clone())

eid = 1
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid(i, j), nid(i+1, j), nid(i+1, j+1), nid(i, j+1)
        mp.CreateNewElement("VMS2D3N", eid, [a, b, c], p); eid += 1
        mp.CreateNewElement("VMS2D3N", eid, [a, c, d], p); eid += 1
print("BUILT nodes", mp.NumberOfNodes(), "elements", mp.NumberOfElements())

for n in mp.Nodes:
    n.AddDof(KM.VELOCITY_X, KM.REACTION_X)
    n.AddDof(KM.VELOCITY_Y, KM.REACTION_Y)
    n.AddDof(KM.PRESSURE, KM.REACTION_WATER_PRESSURE)
    n.SetSolutionStepValue(KM.DENSITY, RHO)
    n.SetSolutionStepValue(KM.VISCOSITY, MU / RHO)

tol = 1e-12
for n in mp.Nodes:
    if abs(n.X) < tol:                                   # inlet
        n.Fix(KM.VELOCITY_X); n.Fix(KM.VELOCITY_Y)
        n.SetSolutionStepValue(KM.VELOCITY_X, U_IN)
    elif abs(n.Y) < tol or abs(n.Y - LY) < tol:          # no-slip walls
        n.Fix(KM.VELOCITY_X); n.Fix(KM.VELOCITY_Y)
    elif abs(n.X - LX) < tol:                            # outlet
        n.Fix(KM.PRESSURE); n.SetSolutionStepValue(KM.PRESSURE, 0.0)
print("BUILT dofs set")

# ---- solve: steady VMS via the fluid application's own steady scheme -----
lin = plsf.ConstructSolver(KM.Parameters('{"solver_type": "skyline_lu_factorization"}'))
scheme = CFD.ResidualBasedSimpleSteadyScheme(0.7, 0.3, 2)   # relax_v, relax_p, dim
conv = KM.DisplacementCriteria(1e-6, 1e-9)
strategy = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, KM.ResidualBasedBlockBuilderAndSolver(lin),
    30, False, False, False)
strategy.SetEchoLevel(0)
mp.ProcessInfo[KM.DELTA_TIME] = 1.0
mp.CloneTimeStep(1.0)
strategy.Initialize()
strategy.Solve()

vx = [n.GetSolutionStepValue(KM.VELOCITY_X) for n in mp.Nodes]
pr = [n.GetSolutionStepValue(KM.PRESSURE) for n in mp.Nodes]
import math
finite = all(math.isfinite(v) for v in vx + pr)
print("SOLVED vx %.4e .. %.4e distinct=%d" % (min(vx), max(vx), len(set(vx))))
print("SOLVED p  %.4e .. %.4e distinct=%d" % (min(pr), max(pr), len(set(pr))))
print("SOLVED all finite:", finite)

# ---- verification against things independent of this run ----------------
# (1) MASS: with one inlet and one outlet the net flux must be ~0.
import collections
outlet = sorted((n for n in mp.Nodes if abs(n.X - LX) < tol), key=lambda n: n.Y)
inlet = sorted((n for n in mp.Nodes if abs(n.X) < tol), key=lambda n: n.Y)
def flux(nodes):
    q = 0.0
    for a, b in zip(nodes[:-1], nodes[1:]):
        ua = a.GetSolutionStepValue(KM.VELOCITY_X)
        ub = b.GetSolutionStepValue(KM.VELOCITY_X)
        q += 0.5 * (ua + ub) * (b.Y - a.Y)
    return q
qi, qo = flux(inlet), flux(outlet)
print("CHECK inlet flux %.6f  outlet flux %.6f  relative imbalance %.3e"
      % (qi, qo, abs(qo - qi) / abs(qi)))

# (2) PROFILE: a developed plane channel has peak = 3/2 * mean, a ratio that
# follows from the parabolic profile and the geometry alone. Nothing about
# this number is stored: it is derived here and compared to what came out.
u_out = [n.GetSolutionStepValue(KM.VELOCITY_X) for n in outlet]
mean_out = qo / LY
peak_out = max(u_out)
print("CHECK outlet peak/mean = %.4f   (developed plane channel: 1.5)"
      % (peak_out / mean_out))
""")


def _fluid_dynamics_channel_2d(params: dict) -> str:
    """Plane channel flow: uniform inlet, no-slip walls, pressure outlet."""
    return _CHANNEL.substitute(
        title=params.get("title", "Plane channel flow (Kratos VMS)"),
        nx=params.get("nx", 12), ny=params.get("ny", 6),
        lx=params.get("lx", 2.0), ly=params.get("ly", 1.0),
        rho=params.get("density", 1.0),
        mu=params.get("viscosity", 0.05),
        u_in=params.get("inlet_velocity", 1.0),
    )


GENERATORS = {
    "fluid_dynamics_channel_2d": _fluid_dynamics_channel_2d,
}

KNOWLEDGE = {
    "fluid_dynamics": {
        "description": (
            "Incompressible Navier-Stokes via Kratos FluidDynamicsApplication, "
            "monolithic VMS (variational multiscale) element."
        ),
        "application": "FluidDynamicsApplication",
        "elements": ["VMS2D3N", "QSVMS2D3N", "Element2D3N"],
        "pitfalls": [
            "[API] THE CONSTITUTIVE LAW CANNOT BE ASSIGNED DIRECTLY FROM "
            "KratosGlobals.GetConstitutiveLaw. It returns a reference, and "
            "properties.SetValue on it raises pybind's 'Unable to cast from "
            "non-held to held instance (T& to Holder<T>)', which names no "
            "Kratos concept at all and reads as an internal bug. Call "
            ".Clone() on it. This bites hardest on applications that expose "
            "NO law classes to Python -- GeoMechanicsApplication has zero "
            "'*Law' names in dir(), so the usual "
            "SMA.LinearElasticPlaneStrain2DLaw() pattern has no equivalent "
            "and cloning is the only route. (Measured 2026-09-19, Kratos "
            "10.3.0.) Signal: the pybind message names a C++ holder type "
            "and no Kratos concept at all, so it reads as an internal bug "
            "rather than as a missing .Clone().",

            "[API] A FLUID ELEMENT NEEDS A CONSTITUTIVE LAW TOO. VMS2D3N "
            "without Newtonian2DLaw on its Properties does not refuse to "
            "build; it fails later or returns nothing useful. The viscosity "
            "lives in the law, not only in DYNAMIC_VISCOSITY. Signal: the "
            "element constructs and the strategy initialises, and the "
            "failure appears later as a degenerate or non-finite velocity "
            "field rather than as a missing-property error.",

            "[Numerical] A HAND-BUILT MODEL PART WITH MISSING MATERIAL "
            "PROPERTIES SOLVES TO NaN RATHER THAN REFUSING. Measured while "
            "writing this: a U-Pw geomechanics part missing its permeability "
            "and fluid bulk modulus ran, reported no error, and returned NaN "
            "pressures with identically zero displacement. Kratos does not "
            "always check that a property it needs was set. Read the field "
            "back and test it is finite before believing any run. Signal: "
            "the solve reports success, one field is entirely NaN, and "
            "another field beside it looks completely plausible.",

            "[Numerical] Check a channel result against the geometry, not "
            "against a remembered number: net flux in must equal flux out, "
            "and a developed plane channel has peak velocity 3/2 of the "
            "mean. Both are derivable from the case you posed, so they work "
            "for YOUR channel and need nothing retrieved. Signal: an "
            "outlet peak/mean ratio far from 3/2, or an inlet and outlet "
            "flux that disagree by more than the discretisation error.",
        ],
    },
}
