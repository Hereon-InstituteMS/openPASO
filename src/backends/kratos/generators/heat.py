"""Kratos heat conduction generators and knowledge."""


from ._convdiff_real import (CROSS_CHECK_NOTE, real_convdiff_script,
                             real_transient_script)


def _heat_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE - values are defaults, determine appropriate values for your specific problem.

    Steady heat conduction -div(k grad T) = f, solved BY KRATOS
    (ConvectionDiffusionApplication, LaplacianElement2D3N).

    The previous body of this function emitted a numpy/scipy assembly whose own
    first line read "Heat conduction - Kratos (manual assembly)" and which
    never imported KratosMultiphysics. See _convdiff_real for the API facts and
    for what that cost.
    """
    nx = params.get("nx", 32)
    return real_convdiff_script(
        title="Steady heat conduction -div(k grad T) = f, Kratos",
        nx=nx, ny=params.get("ny", nx), k=params.get("k", 1.0),
        f_expr=str(params.get("f", 0.0)),
        g_expr=(f"{params.get('T_left', 100.0)} if x <= X0 + 1e-12 "
                f"else {params.get('T_right', 0.0)}"),
        x0=params.get("x0", 0.0), x1=params.get("x1", 1.0),
        y0=params.get("y0", 0.0), y1=params.get("y1", 1.0))


def _heat_transient_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE - values are defaults, determine appropriate values for your specific problem.

    TRANSIENT conduction solved BY KRATOS with EulerianDiffusion2D3N, which is
    the element that carries the capacity term. LaplacianElement2D3N does not:
    a "transient" run built on it returns the steady answer at every step and
    the history is flat.

    The previous body emitted a numpy/scipy assembly with its own time loop and
    never imported KratosMultiphysics.

    Verified by execution: an 8x8 unit square, k=1, rho=cp=1, a 100 -> 0 ramp
    across x, dt=0.01, gives midpoint T = 0.0835 -> 11.86 -> 35.99 -> 46.88 at
    steps 1/5/15/30, approaching the steady-state 50 from below.
    """
    nx = params.get("nx", 32)
    return real_transient_script(
        title="Transient conduction, Kratos EulerianDiffusion2D3N",
        nx=nx, ny=params.get("ny", nx),
        k=params.get("k", 1.0), rho=params.get("rho", 1.0),
        cp=params.get("cp", 1.0),
        dt=params.get("dt", 0.01), t_end=params.get("T_end", 0.3),
        f_expr=str(params.get("f", 0.0)),
        dirichlet_body=(
            f"if x <= X0 + 1e-12:\n"
            f"        return {params.get('T_left', 100.0)}\n"
            f"    if x >= X1 - 1e-12:\n"
            f"        return {params.get('T_right', 0.0)}\n"
            f"    return None"),
        t_init=params.get("T_init", 0.0))


KNOWLEDGE = {
    "heat": {
        "description": "Thermal analysis via ConvectionDiffusionApplication",
        "application": "ConvectionDiffusionApplication",
        "solver_types": ["stationary", "transient"],
        "pitfalls": [
                        '[API] Same field equation as Poisson but with TEMPERATURE as the unknown — TEMPERATURE must be added to ModelPart variables before any Node is created. '
                        "Signal: RuntimeError 'This container only can store the variables specified in its variables list. The variables list doesn't have this variable: TEMPERATURE' from kratos/containers/variables_list_data_value_container at the first GetSolutionStepValue / SetSolutionStepValue on the node. (Verified empirically 2026-06-01 — same wording as the VELOCITY case in fluid#0; prior catalog text 'not found in variables list of ModelPart' + 'from ConvectionDiffusion InitializeSolutionStep' is rearranged and points at the wrong call site.)",
                        '[Syntax] Non-homogeneous Dirichlet: use AssignScalarVariableProcess with constrained=True. Setting constrained=False applies the value but does NOT fix the DOF, so the solver overwrites it. '
                        'Signal: boundary temperatures drift away from the prescribed values during the solve; T_boundary - T_imposed is O(1) instead of O(eps).',
                        '[Syntax] Neumann (heat flux): use ApplyConstantScalarValueProcess on FACE_HEAT_FLUX (not TEMPERATURE). Targeting TEMPERATURE applies a Dirichlet pseudo-flux. '
                        'Signal: the steady-state interior TEMPERATURE field from the VtkOutput .vtu is wrong by a multiplicative factor; the FACE_HEAT_FLUX integral on the boundary does not match the applied value.',
                    ],
    },
    "heat_transient": {
        "description": "Transient heat conduction via ConvectionDiffusionApplication",
        "application": "ConvectionDiffusionApplication",
        "solver_types": ["transient (theta scheme: 0=FE, 0.5=CN, 1=BE)"],
        "time_integration": {
            "backward_euler": "theta=1.0, unconditionally stable, first-order",
            "crank_nicolson": "theta=0.5, second-order but may oscillate",
            "forward_euler": "theta=0.0, conditionally stable (dt < h^2/(2*kappa))",
        },
        "pitfalls": [
            "[Numerical] Backward Euler: factor (M + dt*K) once and reuse each "
            "step — and subtract the Dirichlet columns at the NEW step, not the "
            "old one. The template shipped under this physics assembles K and M "
            "with numpy/scipy (its own first line says 'Kratos (manual "
            "assembly)'), so no Kratos element, scheme or Check() ever sees the "
            "time loop and nothing inside Kratos can catch this: the loop "
            "carries the Dirichlet columns on the old step only, which leaves a "
            "per-step increment that does NOT vanish as dt shrinks. "
            "Signal: run it as pure diffusion — zero source, Dirichlet walls "
            "only — and read max_value out of the results_summary.json the "
            "script writes; it is the same number as the max(T)= field on the "
            "'Transient heat:' line it prints and as the top of the "
            "'temperature' point-data range in result.vtu. For pure diffusion "
            "max_value can never exceed the largest prescribed wall value, and "
            "here it does; worse, REFINING dt raises max_value further instead "
            "of converging, while a correct backward Euler holds it at the wall "
            "maximum for every dt. That makes this checkable with nothing "
            "external — no reference solution, no mesh study — because the run "
            "breaks a bound its own boundary data sets. Fix it by subtracting "
            "the new-step Dirichlet contribution each step, or by not "
            "hand-rolling the loop at all and letting "
            "ConvectionDiffusionApplication own the time stepping with "
            "TEMPERATURE on the ModelPart.",
            "[Numerical] Crank-Nicolson: (M + 0.5*dt*K)*T_new = (M - 0.5*dt*K)*T_old Signal: implemented correctly the scheme is second order in time \u2014 halving dt cuts the error by about a factor of four. A first-order rate on the same mesh means the theta weighting or the Dirichlet elimination is wrong, not that the mesh is too coarse.",
            "[Numerical] Consistent mass matrix gives better accuracy than lumped Signal: the consistent element mass carries non-zero off-diagonal entries while its row-sum lumping is exactly diagonal with identical row sums; swapping in the lumped form raises the time-discretisation error at fixed dt without changing the total heat capacity.",
        ],
        "guidance": [
            "[Physics] For varying BCs in time: update Dirichlet values each step",
        ]
    },
}

GENERATORS = {
    "heat_2d": _heat_2d_kratos,
    "heat_transient_2d": _heat_transient_2d_kratos,
}
