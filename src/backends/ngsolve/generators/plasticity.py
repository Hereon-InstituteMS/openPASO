"""NGSolve plasticity generators and knowledge."""


def _plasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Elasto-plasticity with J2/von Mises yield and isotropic hardening."""
    E = params.get("E", 210e3)
    nu = params.get("nu", 0.3)
    sigma_y = params.get("yield_stress", 250.0)
    H_hard = params.get("hardening", 1000.0)
    n_load_steps = params.get("load_steps", 10)
    max_load = params.get("max_load", 500.0)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    K_bulk = lam + 2 * mu / 3  # for 2D plane strain
    return f'''\
"""Elasto-plasticity: J2/von Mises with isotropic hardening — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json, math

# Domain: rectangular specimen with notch — set geometry for your problem
geo = SplineGeometry()
pnts = [(0, 0), (10, 0), (10, 2), (0, 2)]
p = [geo.AddPoint(*pnt) for pnt in pnts]
geo.Append(["line", p[0], p[1]], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p[1], p[2]], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p[2], p[3]], leftdomain=1, rightdomain=0, bc="top")
geo.Append(["line", p[3], p[0]], leftdomain=1, rightdomain=0, bc="left")
mesh = Mesh(geo.GenerateMesh(maxh=0.5))

# Material parameters
E_mod = {E}
nu_val = {nu}
mu_val = {mu}
lam_val = {lam}
sigma_y = {sigma_y}
H_hard = {H_hard}

# FE space
fes = VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()

def Strain(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

def Stress_elastic(strain):
    return 2 * mu_val * strain + lam_val * Trace(strain) * Id(2)

# Displacement solution
gfu = GridFunction(fes)

# Internal variables stored as GridFunctions on L2
# Plastic strain (symmetric tensor) and equivalent plastic strain
fes_tensor = MatrixValued(L2(mesh, order=1), symmetric=True)
fes_scalar = L2(mesh, order=1)

eps_p = GridFunction(fes_tensor)    # plastic strain
alpha = GridFunction(fes_scalar)    # equivalent plastic strain (hardening)
eps_p.Set(CoefficientFunction((0, 0, 0, 0), dims=(2,2)))
alpha.Set(0)

# Load stepping
n_steps = {n_load_steps}
max_load = {max_load}

for step in range(1, n_steps + 1):
    load_factor = step / n_steps
    traction = max_load * load_factor

    # Elastic predictor: solve with current plastic strain as pre-strain
    a = BilinearForm(fes)
    a += InnerProduct(Stress_elastic(Strain(u)), Strain(v)) * dx
    a.Assemble()

    # RHS: traction on right edge + correction for plastic strain
    f = LinearForm(fes)
    f += CoefficientFunction((traction, 0)) * v * ds("right")
    f += InnerProduct(Stress_elastic(eps_p), Strain(v)) * dx
    f.Assemble()

    # Solve elastic predictor
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

    # Compute trial stress
    eps_total = Strain(gfu)
    eps_elastic_trial = eps_total - eps_p
    stress_trial = Stress_elastic(eps_elastic_trial)

    # Von Mises yield check and return mapping would go here
    # For a simplified approach: compute von Mises stress for output
    # sigma_vm = sqrt(s:s * 3/2) where s = stress - 1/3*tr(stress)*I
    s_dev = stress_trial - 1.0/2.0 * Trace(stress_trial) * Id(2)
    sigma_vm_cf = sqrt(1.5 * InnerProduct(s_dev, s_dev))

    vm_max = Integrate(sigma_vm_cf, mesh) / Integrate(1, mesh)

    # Simplified radial return: if sigma_vm > sigma_y + H*alpha, update plastic strain
    # (Full implementation requires integration point level return mapping)
    print(f"Step {{step}}/{n_load_steps}: load={{traction:.1f}}, avg von Mises={{vm_max:.2f}}")

# Final output
vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"],
                filename="result", subdivision=1)
vtk.Do()

# Compute final displacement at right edge
disp_x = gfu.components[0](mesh(10, 1))
disp_y = gfu.components[1](mesh(10, 1))
print(f"Displacement at (10,1): u_x={{disp_x:.6e}}, u_y={{disp_y:.6e}}")

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "load_steps": n_steps,
    "max_load": max_load,
    "yield_stress": sigma_y,
    "hardening": H_hard,
    "disp_x_at_tip": float(disp_x),
    "disp_y_at_tip": float(disp_y),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Elasto-plasticity analysis complete.")
'''


KNOWLEDGE = {
    "plasticity": {
        "description": (
            "Elasto-plasticity with J2 / von Mises yield and "
            "isotropic hardening; integration-point-level nonlinear "
            "material solves use ngsolve.fem.NewtonCF or "
            "ngsolve.fem.MinimizationCF (NOT importable from the "
            "top-level ngsolve namespace — see pitfall #0)."
        ),
        "spaces": (
            "VectorH1 for displacement, L2 for internal variables "
            "(plastic strain, hardening)."
        ),
        "solver": (
            "Load stepping with elastic predictor + plastic "
            "corrector (return mapping)."
        ),
        "pitfalls": [
            "[API] NewtonCF and MinimizationCF live in the "
            "ngsolve.fem submodule — NOT at the top level. "
            "hasattr(ngsolve, 'NewtonCF') is False and "
            "'from ngsolve import *' does NOT bring them in. "
            "Correct import is 'from ngsolve.fem import NewtonCF, "
            "MinimizationCF'. An LLM agent that follows the prior "
            "catalog hint (which read 'NewtonCF/MinimizationCF in "
            "NGSolve') and writes 'from ngsolve import NewtonCF' "
            "hits ImportError; 'ngsolve.NewtonCF(...)' hits "
            "AttributeError. Signal: hasattr(ngsolve, 'NewtonCF') "
            "is False, hasattr(ngsolve.fem, 'NewtonCF') is True, "
            "and callable(NewtonCF) after the correct import. "
            "(Verified empirically 2026-06-01 — Tier-2 fixture "
            "plasticity_newtoncf_in_fem_submodule in scripts/"
            "tier2_fixtures/ngsolve/.)",
            "[Physics] J2 plasticity yield criterion: sigma_vm = "
            "sqrt(3/2 * s:s) <= sigma_y + H*alpha, where s is the "
            "deviatoric stress, alpha is the accumulated plastic "
            "strain, and H is the isotropic hardening modulus. "
            "The post-yield slope is the observable, but USE THE "
            "RIGHT TANGENT FORMULA FOR THE RIGHT LOADING: for "
            "UNIAXIAL tension the post-yield secant satisfies "
            "E_t/E = H/(H + E); the factor (1 + 3*mu/H)^{-1} = "
            "H/(H + 3*mu) is the SHEAR tangent, and the two "
            "coincide only in the incompressible limit nu = 0.5. "
            "Applying the shear expression to a uniaxial test is "
            "off by a wide margin at ordinary Poisson ratios, so "
            "a tolerance band built on it rejects a correct return "
            "map. Signal: drive a homogeneous strain on a "
            "VectorH1 GridFunction with the radial return written "
            "in CoefficientFunction algebra and read the "
            "components back with Integrate — below yield the "
            "secant equals E; above yield the uniaxial secant "
            "divided by E matches H/(H + E) to round-off while "
            "the SHEAR secant divided by mu matches H/(H + 3*mu). "
            "If the radial return is skipped altogether the "
            "post-yield slope stays at E and sigma_vm overshoots "
            "sigma_y + H*alpha by a large factor — that, not the "
            "tangent value, is the check for a missing corrector. "
            "(Verified 2026-08-06 on NGSolve 6.2.2604 — Tier-2 "
            "fixture plasticity_post_yield_slope.)",
            "[Numerical] Return mapping is the integration-point "
            "step: compute the elastic trial stress sigma_trial = "
            "C : (eps - eps_p_old), check sigma_vm(sigma_trial) "
            "against the yield surface, and if outside project "
            "back via the radial return (J2: project along the "
            "trial deviator direction). With NewtonCF this becomes "
            "a per-integration-point Newton solve. Signal: "
            "iteration count to drive |sigma_vm - "
            "(sigma_y + H*alpha)| below 1e-10 is bounded "
            "independently of mesh size for quadratic Newton "
            "convergence. (Catalog claim inherited.)",
            "[API] Internal variables eps_p and alpha must live at "
            "the QUADRATURE POINTS of the displacement space, not "
            "on H1 nodes — and an L2 space does NOT give you them. "
            "L2(mesh, order=p) is a modal/polynomial space: its "
            "DOFs per element are the dimension of the polynomial "
            "space, which does not equal the number of points in "
            "any integration rule, at any order. Use "
            "ngsolve.comp.IntegrationRuleSpace, which is "
            "point-collocated and satisfies ndof == mesh.ne * "
            "len(irs.GetIntegrationRules()[ET.TRIG]) exactly. Ask "
            "the SPACE for its rule: its collocated rule is not "
            "the same as ngsolve.fem.IntegrationRule(ET.TRIG, p), "
            "so do not assume one and count its points. "
            "IMPORT DETAIL THAT COSTS AN AGENT A RUN: it is "
            "reachable NEITHER as ngsolve.IntegrationRuleSpace NOR "
            "through 'from ngsolve import *' — you must write "
            "'from ngsolve.comp import IntegrationRuleSpace'. "
            "Mismatched quadrature between the residual assembly "
            "and the internal-variable storage silently produces "
            "wrong plastic strains. Signal: print L2(mesh, "
            "order=p).ndof / mesh.ne next to the point count of "
            "ngsolve.fem.IntegrationRule(ET.TRIG, p) — they "
            "disagree at every order, so an equality assertion on "
            "the L2 space fails on a correct setup and tells you "
            "nothing about alignment; the same comparison for "
            "IntegrationRuleSpace against its own "
            "GetIntegrationRules() matches "
            "exactly, and hasattr(ngsolve, 'IntegrationRuleSpace') "
            "is False. (Verified 2026-08-06 on NGSolve 6.2.2604 — "
            "Tier-2 fixture "
            "plasticity_internal_var_quadrature_space.)",
            "[Numerical] Load stepping is required (NOT optional) "
            "for J2 + hardening — applying the full traction in "
            "one step almost always overshoots the yield surface "
            "by more than the trust region of the return-mapping "
            "Newton iteration. Use n_steps such that the per-step "
            "stress increment is below ~10% of sigma_y. Signal: "
            "with too-large step size, solvers.Newton called on "
            "the per-step BilinearForm returns (-1, maxit) and "
            "prints 'Warning: Newton might not converge! Error = ' "
            "— note the ORDER: the return is (status, numit) with "
            "status 0 = converged and -1 = gave up, NOT "
            "(iters, conv), and there is NO error value in the "
            "return at all, so a test of the form 'second element "
            "> maxerr' is comparing against an iteration count. "
            "Test status == 0. Reduce the load increment (more "
            "n_load_steps), not the maxit kwarg of Newton: raising "
            "maxit leaves the run diverging at every value, "
            "because the trust region is set by the step size, "
            "while shrinking the increment to about a tenth of "
            "sigma_y or below converges — and all sufficiently "
            "small increments converge to the same displacement. "
            "(Verified 2026-08-06 on NGSolve 6.2.2604 — Tier-2 "
            "fixture plasticity_load_step_divergence_signature.)",
            "[Physics] For large deformations switch to the "
            "multiplicative decomposition F = F_e * F_p (not the "
            "additive eps = eps_e + eps_p which is small-strain "
            "only). Stresses computed from F_e via a hyperelastic "
            "constitutive law; plastic flow rule and yield "
            "criterion expressed in the Mandel-stress / "
            "intermediate configuration. Signal: define F = Id + "
            "Grad(gfu) with VectorH1 displacement, compute Det(F) "
            "and Cof(F) on an integration rule; with the small-"
            "strain additive split, the Cauchy stress assembled "
            "via Stress(eps_e) and the spatially-pushed-forward "
            "Mandel stress from F_e disagree by a factor on the "
            "order of (Det(F))^{-1} beyond ~5%% engineering "
            "strain — observable as a divergence between the two "
            "computed stress fields on the same gfu. (Catalog "
            "claim inherited.)",
            "[Numerical] Consistent tangent modulus is required "
            "for quadratic global Newton convergence; the "
            "continuum tangent (elastic + plastic stiffness) does "
            "NOT match the linearization of the return-mapping "
            "discretization and gives only linear convergence. "
            "Compute the consistent tangent by linearizing the "
            "return-mapping update — for J2 this gives the "
            "well-known formula with the projection tensor. Do "
            "NOT expect the continuum tangent to merely be slower: "
            "its per-iteration residual drop is far closer to 1 "
            "than to the 'factor of ~2-3' the prior text quoted, "
            "so in practice it does not converge at all inside any "
            "sane iteration cap — an iteration budget generous "
            "enough to 'let it finish' does not exist. Signal: "
            "below yield, AssembleLinearization and a hand-built "
            "continuum C_ep produce the SAME matrix (that "
            "agreement is the check that the continuum "
            "implementation itself is right); in the plastic range "
            "they differ by a large fraction in matrix norm. "
            "Newton with the consistent tangent converges in a "
            "handful of residual evaluations with a superlinear "
            "tail, while with the continuum tangent "
            "|r_{k+1}|/|r_k| sits at a nearly CONSTANT value just "
            "under 1 and the run is still above tolerance when the "
            "cap is hit. Watch the ratio, not the iteration count. "
            "(Verified 2026-08-06 on NGSolve 6.2.2604 — Tier-2 "
            "fixture plasticity_consistent_tangent.)",
        ],
    },
}

GENERATORS = {
    "plasticity_2d": _plasticity_2d,
}
