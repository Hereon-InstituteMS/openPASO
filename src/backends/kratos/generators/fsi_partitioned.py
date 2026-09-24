"""Kratos FSIApplication: partitioned coupling with a convergence accelerator.

FSIApplication's substance is its CONVERGENCE ACCELERATORS -- Aitken, MVQN,
IBQN-MVQN, constant relaxation -- for the fixed-point iteration of a
partitioned coupling. This template uses one for exactly that, driving a
Dirichlet-Neumann exchange between two real Kratos ConvectionDiffusion
solves.

VERIFIED BY EXECUTION on Kratos 10.3.0 against two things independent of the
run. The converged interface temperature must equal the value derived from
the two conductivities, the two lengths and the two boundary temperatures --
it does, to six figures, at conductivity ratios of 1, 10 and 100. And the
accelerator must not move that fixed point, only reach it sooner -- at a
ratio of 10 Aitken takes 3 iterations where constant relaxation takes 25,
and both land on the same value.

THE FOUR DEFECTS IT TOOK TO GET THERE are the knowledge below. Three of them
produce a confident wrong answer rather than an error, and the fourth
produces no error message at all.
"""
from string import Template


_PARTITIONED = Template(r"""
'''$title

Partitioned Dirichlet-Neumann coupling of two REAL Kratos solves, with the
fixed-point iteration driven by a KratosFSIApplication convergence
accelerator. Both sides are ConvectionDiffusion solves; nothing here
assembles a system by hand.

It checks itself against two things independent of the run: the converged
interface temperature must equal the value derived from the two
conductivities, the two lengths and the two boundary temperatures, and the
accelerated iteration must not change that fixed point.
'''
import numpy as np, math
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication as CDA
import KratosMultiphysics.FSIApplication as FSI
import KratosMultiphysics.LinearSolversApplication  # noqa
from KratosMultiphysics import python_linear_solver_factory as plsf

import os
MUTATE = os.environ.get("SURVEY_MUTATE", "") == "1"   # planted failure, constant-relaxation branch only
NX, NY = $nx, $ny
L1, L2, H = $l1, $l2, $height
K1, K2 = $k1, $k2
T_LEFT, T_RIGHT = $t_left, $t_right

def make_strip(model, name, x0, x1, k):
    mp = model.CreateModelPart(name)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    for v in (KM.TEMPERATURE, KM.REACTION_FLUX, KM.HEAT_FLUX,
              KM.FACE_HEAT_FLUX, KM.CONDUCTIVITY, KM.SPECIFIC_HEAT, KM.DENSITY):
        mp.AddNodalSolutionStepVariable(v)
    xs = np.linspace(x0, x1, NX + 1); ys = np.linspace(0.0, H, NY + 1)
    nid = lambda i, j: j * (NX + 1) + i + 1
    for j in range(NY + 1):
        for i in range(NX + 1):
            mp.CreateNewNode(nid(i, j), float(xs[i]), float(ys[j]), 0.0)
    # CONVECTION_DIFFUSION_SETTINGS tells LaplacianElement WHICH variables
    # are the unknown, the conductivity and the source. Without it the
    # element dereferences an unset pointer and the process SEGFAULTS -- no
    # exception, no message, just exit 139.
    st = KM.ConvectionDiffusionSettings()
    st.SetUnknownVariable(KM.TEMPERATURE)
    st.SetDiffusionVariable(KM.CONDUCTIVITY)
    st.SetVolumeSourceVariable(KM.HEAT_FLUX)
    st.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    st.SetDensityVariable(KM.DENSITY)
    st.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, st)
    p = mp.CreateNewProperties(1)
    p[KM.CONDUCTIVITY] = k; p[KM.DENSITY] = 1.0; p[KM.SPECIFIC_HEAT] = 1.0
    eid = 1
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a,b,c], p); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a,c,d], p); eid += 1
    for n in mp.Nodes:
        n.AddDof(KM.TEMPERATURE, KM.REACTION_FLUX)
        n.SetSolutionStepValue(KM.CONDUCTIVITY, k)
    return mp

def solver_for(mp):
    lin = plsf.ConstructSolver(KM.Parameters('{"solver_type": "skyline_lu_factorization"}'))
    # THE FIRST FLAG IS CalculateReactionsFlag AND A PARTITIONED COUPLING
    # DIES WITHOUT IT. With it False the solve is correct and REACTION_FLUX
    # stays identically zero, so a Dirichlet-Neumann exchange transfers
    # nothing, the residual is zero on the first pass, and the coupling
    # reports CONVERGED IN ONE ITERATION to the wrong answer.
    s = KM.ResidualBasedLinearStrategy(
        mp, KM.ResidualBasedIncrementalUpdateStaticScheme(), lin,
        True, False, False, False)
    s.SetEchoLevel(0); return s

model = KM.Model()
left = make_strip(model, "left", 0.0, L1, K1)
right = make_strip(model, "right", L1, L1 + L2, K2)
# A NODAL FLUX IS NOT A LOAD. FACE_HEAT_FLUX set on nodes contributes
# nothing unless FACE CONDITIONS integrate it over the boundary. Without
# them the receiving side sees no source at all and stays at its initial
# value, while every variable you set looks correctly assigned.
iface_r_nodes = sorted((n for n in right.Nodes if abs(n.X - L1) < 1e-12),
                       key=lambda n: n.Y)
pr = right.GetProperties()[1]
for ci, (a, b) in enumerate(zip(iface_r_nodes[:-1], iface_r_nodes[1:]), start=1):
    right.CreateNewCondition("ThermalFace2D2N", 10000 + ci, [a.Id, b.Id], pr)

sl, sr = solver_for(left), solver_for(right)
for mp in (left, right):
    mp.ProcessInfo[KM.DELTA_TIME] = 1.0; mp.CloneTimeStep(1.0)
tol = 1e-12
iface_l = [n for n in left.Nodes if abs(n.X - L1) < tol]
iface_r = [n for n in right.Nodes if abs(n.X - L1) < tol]
for n in left.Nodes:
    if abs(n.X) < tol:
        n.Fix(KM.TEMPERATURE); n.SetSolutionStepValue(KM.TEMPERATURE, T_LEFT)
for n in right.Nodes:
    if abs(n.X - (L1 + L2)) < tol:
        n.Fix(KM.TEMPERATURE); n.SetSolutionStepValue(KM.TEMPERATURE, T_RIGHT)
for n in iface_l:
    n.Fix(KM.TEMPERATURE)

def coupled(acc, label):
    for n in iface_l: n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    acc.Initialize(); acc.InitializeSolutionStep()
    x = KM.Vector(len(iface_l))
    for i in range(len(iface_l)): x[i] = 0.0
    for it in range(1, 60):
        for i, n in enumerate(iface_l): n.SetSolutionStepValue(KM.TEMPERATURE, x[i])
        sl.Solve()
        # A NODAL REACTION IS AN INTEGRATED QUANTITY; FACE_HEAT_FLUX IS A
        # DENSITY THAT THE CONDITION INTEGRATES AGAIN. Passing the reaction
        # straight across double-integrates it, and the coupling converges
        # -- cleanly, with the accelerator working -- to the WRONG fixed
        # point. Divide by the node's tributary length: h for an interior
        # node, h/2 at each end.
        h = H / NY
        q = []
        for i, n in enumerate(iface_l):
            trib = h if 0 < i < len(iface_l) - 1 else 0.5 * h
            if MUTATE and label == "constant":
                trib = 1.0    # the planted wrong answer: the reaction passed across as if it were a density
            q.append(n.GetSolutionStepValue(KM.REACTION_FLUX) / trib)
        for i, n in enumerate(iface_r):
            n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, -q[i])
        sr.Solve()
        t_new = [n.GetSolutionStepValue(KM.TEMPERATURE) for n in iface_r]
        r = KM.Vector(len(iface_l))
        for i in range(len(iface_l)): r[i] = t_new[i] - x[i]
        res = max(abs(r[i]) for i in range(len(iface_l)))
        if res < 1e-9: break
        acc.InitializeNonLinearIteration(); acc.UpdateSolution(r, x)
        acc.FinalizeNonLinearIteration()
    return it, [x[i] for i in range(len(iface_l))]

it_a, t_a = coupled(FSI.AitkenConvergenceAccelerator(0.5), "aitken")
it_c, t_c = coupled(FSI.ConstantRelaxationConvergenceAccelerator(0.5), "constant")
t_exact = (T_LEFT*K1/L1 + T_RIGHT*K2/L2) / (K1/L1 + K2/L2)
print("FSI aitken   %2d iterations, interface T = %.6f" % (it_a, sum(t_a)/len(t_a)))
print("FSI constant %2d iterations, interface T = %.6f" % (it_c, sum(t_c)/len(t_c)))
print("FSI analytic interface T = %.6f  (derived from k, L and the two BCs)" % t_exact)
# One VERDICT line per accelerator, in the coverage harness's grammar
# (scripts/coverage_harness/definitions.py). The reference is the closed-form
# interface temperature of two conducting strips in series; P1 elements on
# these meshes are nodally exact for that piecewise-linear profile, so the only
# distance left is the coupling's stopping residual (1e-9 on the interface
# temperature), and the tolerance is a thousand times that.
for _label, _t in (("aitken", sum(t_a)/len(t_a)), ("constant", sum(t_c)/len(t_c))):
    _ok = abs(_t - t_exact) <= 1e-6
    print("VERDICT kratos fsi_partitioned closed_form ref=%.6e got=%.6e tol=%.6e "
          "tol_from=coupling_stop_residual_1e-9_x1000;P1_nodally_exact_for_a_piecewise_linear_profile %s"
          % (t_exact, _t, 1e-6, "PASS" if _ok else "FAIL"))

ok = abs(sum(t_a)/len(t_a) - t_exact) < 1e-6 and abs(sum(t_c)/len(t_c) - t_exact) < 1e-6
print("FSI interface matches the derived value:", ok)
print("FSI accelerator saved %d iterations" % (it_c - it_a))
""")


def _fsi_partitioned_2d(params: dict) -> str:
    """Two conducting strips coupled Dirichlet-Neumann across an interface."""
    return _PARTITIONED.substitute(
        title=params.get("title", "Partitioned coupling with an FSI accelerator"),
        nx=params.get("nx", 8), ny=params.get("ny", 4),
        l1=params.get("length_left", 1.0), l2=params.get("length_right", 1.0),
        height=params.get("height", 1.0),
        k1=params.get("conductivity_left", 1.0),
        k2=params.get("conductivity_right", 10.0),
        t_left=params.get("temperature_left", 1.0),
        t_right=params.get("temperature_right", 0.0),
    )


GENERATORS = {
    "fsi_partitioned_2d": _fsi_partitioned_2d,
}

KNOWLEDGE = {
    "fsi_partitioned": {
        "description": (
            "Partitioned Dirichlet-Neumann coupling driven by a "
            "KratosFSIApplication convergence accelerator (Aitken, MVQN, "
            "IBQN-MVQN, constant relaxation)."
        ),
        "application": "FSIApplication",
        "pitfalls": [
            "[Numerical] A NODAL REACTION IS AN INTEGRATED QUANTITY AND A "
            "FACE FLUX IS A DENSITY. Passing REACTION_FLUX straight across "
            "an interface double-integrates it: the coupling still "
            "converges, the accelerator still works, and it lands on the "
            "WRONG fixed point. Measured: an interface that should sit at "
            "0.5 converged cleanly to 0.177 instead. Divide by the node's "
            "tributary length -- h for an interior node, h/2 at each end -- "
            "and the same run lands on 0.500000. Nothing about the failure "
            "looks like a failure. Signal: a coupling that converges cleanly to a "
            "value you can also derive analytically, and misses it by a "
            "factor of roughly the tributary length -- 0.177 where 0.5 was "
            "derivable. Derive the interface value and compare; a clean "
            "convergence to the wrong number has no other symptom.",

            "[API] REACTION_FLUX IS IDENTICALLY ZERO UNLESS THE STRATEGY WAS "
            "BUILT TO COMPUTE IT. The first flag of ResidualBasedLinearStrategy "
            "is CalculateReactionsFlag; with it False the solve is entirely "
            "correct and the reactions are all zero. A Dirichlet-Neumann "
            "exchange then transfers nothing, the residual is zero on the "
            "first pass, and the coupling reports CONVERGED IN ONE ITERATION. "
            "A coupling that converges immediately is exchanging nothing "
            "until proven otherwise. Signal: the iteration count is 1, and every "
            "transferred value reads back as exactly zero while each side 's"
            "own solve is correct.",

            "[API] A NODAL FLUX IS NOT A LOAD. Setting FACE_HEAT_FLUX on the "
            "receiving nodes contributes nothing unless FACE CONDITIONS "
            "integrate it over the boundary -- ThermalFace2D2N for a 2-D "
            "line. Without them the receiving side sees no source at all and "
            "sits at its initial value, while every variable you set reads "
            "back exactly as assigned. Signal: the receiving field is uniformly "
            "its initial value while the variable you set reads back "
            "correctly on every node -- assignment succeeded, assembly "
            "never happened.",

            "[Setup] A LaplacianElement WITHOUT CONVECTION_DIFFUSION_SETTINGS "
            "SEGFAULTS. The settings object on ProcessInfo names which "
            "variable is the unknown, which is the diffusivity and which are "
            "the sources; without it the element dereferences an unset "
            "pointer. There is no exception and no message -- the process "
            "dies with exit 139. Signal: the process terminates with SIGSEGV and no "
            "Python traceback, immediately at the first solve rather than "
            "during setup. Set the settings object before creating any "
            "element.",

            "[Numerical] THE ACCELERATOR EARNS ITS KEEP ONLY WHEN THE "
            "COUPLING IS STIFF. With matched conductivities both Aitken and "
            "constant relaxation converge in 2 iterations, so a comparison "
            "on an easy case proves nothing about the accelerator. Raise the "
            "conductivity ratio and the gap opens: measured 3 against 25 at "
            "a ratio of 10. Judge an accelerator on the case that needs one. "
            "Signal: two accelerators reporting the SAME iteration count "
            "on your test case -- that is the case being too easy, not the "
            "accelerators being equivalent.",
        ],
    },
}
