"""Transient thermal MMS generator for 4C — temporal convergence measurement.

Unsteady heat conduction (PROBLEMTYPE "Thermo", One-Step-Theta) with a
manufactured time-and-space-dependent solution u*(x,t), a matching
volumetric heat source q = rho*c*du*/dt - kappa*lap(u*) as a
SYMBOLIC_FUNCTION_OF_SPACE_TIME body load, time-dependent Dirichlet u*
on the whole boundary, and initial field u*(x,0). On a fixed fine mesh
a dt-halving study measures the temporal order of the time integrator:
theoretical order 2 for theta = 0.5 (Crank-Nicolson), 1 for theta = 1
(backward Euler). The order a given study attains is an OUTPUT of that
study — compute it from the emitted errors, do not assume it.

The 4C-behaviour claims below (which sections are honoured, what is
silently dropped, what the parser accepts) were checked live against a
built 4C binary rather than read off the documentation.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ThermoTransientMMSGenerator(BaseGenerator):
    """Generator for transient thermo MMS temporal-convergence decks."""

    module_key = "thermo_transient_mms"
    display_name = "Transient Thermal MMS (Temporal Convergence)"
    problem_type = "Thermo"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Method-of-manufactured-solutions transient heat "
                "conduction on [0,lx]x[0,ly]: u* = temp_offset + "
                "(amp*sin(kx x)*sin(ky y) + grad_amp*x/lx)*f(t) with "
                "f(t) = cos(omega t) or exp(-omega t). The exact "
                "volumetric source rho*c*du*/dt - kappa*lap(u*) is "
                "applied as a space-time function body load, u* as "
                "time-dependent Dirichlet on the whole boundary and "
                "as the initial field. Fixed mesh + dt-halving measures "
                "the TEMPORAL order of One-Step-Theta: ~2 for "
                "theta=0.5, ~1 for theta=1."
            ),
            "yaml_section": "THERMAL DYNAMIC",
            "elements": {"2D": ["THERMO QUAD4"]},
            "time_integration": {
                "OneStepTheta": (
                    "THERMAL DYNAMIC/ONESTEPTHETA THETA in (0,1] "
                    "(4C throws 'theta out of range' otherwise, "
                    "4C_thermo_timint_ost.cpp:20). The OST residual "
                    "theta-weights internal AND external forces "
                    "between t_n and t_n+1 (trapezoid quadrature of "
                    "the source) and consistent initial temperature "
                    "rates are computed at t=0, so theta=0.5 is "
                    "genuinely 2nd-order in time and theta=1.0 is "
                    "1st-order. Confirm on YOUR case with a dt-halving "
                    "study; see the spatial-floor pitfall for how to "
                    "measure it correctly."
                ),
            },
            "materials": {
                "MAT_Fourier": {
                    "parameters": {
                        "CAPA": "VOLUMETRIC heat capacity rho*c "
                                "[J/(m^3 K)] — the generator computes "
                                "it as rho*c from the two inputs",
                        "CONDUCT": "tensor-typed: wrap as "
                                   "'constant: [kappa]'",
                    },
                },
            },
            "output_extraction": (
                "THERMAL DYNAMIC/RUNTIME VTK OUTPUT (OUTPUT_THERMO: "
                "true, TEMPERATURE: true) plus IO/RUNTIME VTK OUTPUT "
                "(INTERVAL_STEPS: NUMSTEP) writes "
                "<prefix>-vtk-files/thermo-<step>-<proc>.vtu with "
                "nodal point_data key 'temperature' (plus a "
                "<prefix>-thermo.pvd collection). Step 0 and the "
                "final step are written; meshio reads the .vtu "
                "directly. Verified: step-0 field reproduces "
                "u*(x,0) exactly (initial field by function)."
            ),
            "pitfalls": [
                "[Syntax] A volumetric heat source in standalone "
                "Thermo must use the PLAIN Neumann sections "
                "('DESIGN SURF NEUMANN CONDITIONS' in 2D, "
                "'DESIGN VOL NEUMANN CONDITIONS' in 3D). The "
                "THERMO-prefixed 'DESIGN SURF/VOL THERMO NEUMANN "
                "CONDITIONS' are registered under condition names "
                "'ThermoSurfaceNeumann'/'ThermoVolumeNeumann' "
                "(4C_global_legacy_module_validconditions.cpp:112-"
                "124) and are matched NEITHER by Core::FE::"
                "Discretization::evaluate_neumann (loops only "
                "'PointNeumann/LineNeumann/SurfaceNeumann/"
                "VolumeNeumann') NOR by the thermo body-load lookup "
                "TemperImpl::radiation() (find_element_conditions "
                "with the plain names, 4C_thermo_ele_impl.cpp:"
                "2516-2525). The run finishes rc=0 with NO warning "
                "and the source is silently dropped. Signal: the run "
                "finishes rc=0 with no warning while the field "
                "converges (in dt) to the SOURCE-FREE solution — the "
                "error against the MMS solution is O(1) relative and "
                "refuses to shrink under dt refinement, whereas the "
                "same deck with the plain section converges "
                "normally. The THERMO-prefixed sections are for the "
                "TSI condition-transfer machinery, not standalone "
                "Thermo. (Verified 2026-08-01.)",
                "[Syntax] In a 2D Thermo discretization the plain "
                "'DESIGN SURF NEUMANN CONDITIONS' on a DSURF "
                "covering all nodes IS the body load: the condition "
                "geometry is the QUAD4 elements themselves and "
                "TemperImpl::radiation() integrates ONOFF*VAL*"
                "FUNCT(x,t) over each element (FunctionOfSpaceTime "
                "at Gauss points, reference coordinates). "
                "Constraints from the code: exactly ONE such "
                "condition may touch an element — a real "
                "FOUR_C_THROW, raised during element evaluation "
                "after the discretisation is already built, not at "
                "input read. The 'FUNCT must have exactly one "
                "entry' rule next to it is a FOUR_C_ASSERT and is "
                "therefore compiled out of a Release build: a "
                "longer FUNCT array is silently ignored there and "
                "gives bit-identical results, so do not rely on it "
                "being caught. Signal: overlapping conditions abort "
                "with 'more than one VolumeNeumann cond on one "
                "node' from 4C_thermo_ele_impl.cpp; a MISSING body "
                "load, by contrast, draws no diagnostic whatsoever "
                "and shows up only as a wrong field. (Verified by "
                "execution 2026-08-06.)",
                "[Syntax] Initial condition by function: THERMAL "
                "DYNAMIC INITIALFIELD 'field_by_function' + "
                "INITFUNCNO <id>; the referenced "
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME is evaluated at "
                "t=0 at the nodes. Signal: the step-0 VTK file "
                "equals u*(x,0) to machine precision when the "
                "initial field is applied; a "
                "zero_field default shows up as an O(amplitude) "
                "error at step 0 that decays over the first steps.",
                "[Syntax] Time-dependent Dirichlet: plain 'DESIGN "
                "LINE DIRICH CONDITIONS' with VAL [1.0] and FUNCT "
                "[<id>] evaluates the space-time function at node "
                "coordinates and CURRENT time each step "
                "(effective value = VAL * f(x,t)). Works on the "
                "whole boundary of a pure Thermo problem; the "
                "THERMO-prefixed DIRICH sections are not needed. "
                "Signal: boundary nodes track "
                "u*(x,t) exactly at every step when wired correctly "
                "(verified 2026-08-01). If FUNCT is omitted the "
                "boundary is frozen at VAL ITSELF — a single "
                "constant, the same at every boundary node and at "
                "every step — NOT at the t=0 trace of the intended "
                "solution. Check against VAL, not against u*(x,0): "
                "an earlier version of this entry said the frozen "
                "value equals the t=0 trace, and a reader checking "
                "for that would have concluded the boundary "
                "condition was correctly wired. (Corrected by "
                "execution 2026-08-06.)",
                "[Numerics] On a fixed mesh the error vs the exact "
                "MMS solution saturates at the SPATIAL (Q1) floor, so "
                "the small-dt entries of an error-vs-exact table "
                "flatten out and a naive fit reports a temporal order "
                "far below the theoretical one. Grade the temporal "
                "order from Richardson differences "
                "||u_dt - u_dt/2|| of consecutive-dt solutions on "
                "the SAME mesh instead — the spatial floor cancels "
                "exactly. Signal: an error-vs-exact dt series whose "
                "successive ratios decay toward 1 while the "
                "Richardson ratios stay near 2^p is the spatial "
                "floor, not a solver defect; refine the mesh (or use "
                "Richardson) before concluding anything about the "
                "time integrator.",
                "[Numerics] The symbolic expression parser knows "
                "'pi' (4C_utils_symbolic_expression.cpp:479); the "
                "generator nevertheless bakes kx, ky, omega and all "
                "coefficients in as numeric literals (:.16g) to "
                "avoid precedence surprises. Signal: a malformed "
                "symbolic expression fails at input read with a "
                "4C parser error naming the expression; a silently "
                "mis-parsed one shows as an initial field that does "
                "not match u*(x,0) in the step-0 VTK.",
                "[Physics] Choose omega and T_end so the temporal "
                "error dominates: temporal error ~ (dt*omega)^p * "
                "amplitude. If MAXTIME is not an integer multiple "
                "of TIMESTEP the generator rounds NUMSTEP = "
                "round(T_end/dt) and sets MAXTIME = NUMSTEP*dt — "
                "compare solutions at the SAME final time across "
                "the dt series (use dt-halving from a dt that "
                "divides T_end exactly). Signal: mismatched final "
                "times across a dt series produce error ratios that "
                "wander instead of approaching 2^p — check the "
                "actual MAXTIME/NUMSTEP echoed in the 4C stdout "
                "header before fitting the order.",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{
            "name": "temporal_mms_2d",
            "description": (
                "2D transient heat conduction MMS on a fixed QUAD4 "
                "mesh for dt-halving temporal-order studies "
                "(One-Step-Theta, exact space-time source + "
                "time-dependent Dirichlet + initial field)"),
        }]

    def get_template(self, variant: str = "temporal_mms_2d") -> str:
        from ..inline_mesh import matched_thermo_transient_mms_input
        if variant in ("temporal_mms_2d", "default"):
            return matched_thermo_transient_mms_input()
        raise ValueError(f"Unknown variant {variant!r}")

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        def _num(key, alt=None):
            v = params.get(key)
            if v is None and alt is not None:
                v = params.get(alt)
            return v

        theta = _num("theta")
        if theta is not None and not (0.0 < float(theta) <= 1.0):
            issues.append(
                f"theta={theta} is outside (0, 1] — 4C One-Step-Theta "
                "throws 'theta out of range (0.0,1.0]'. Use 0.5 for "
                "2nd order (Crank-Nicolson) or 1.0 for backward "
                "Euler.")

        dt = _num("dt")
        if dt is not None and float(dt) <= 0.0:
            issues.append(f"dt={dt} must be > 0.")

        t_end = _num("T_end", "t_end")
        if t_end is not None:
            if float(t_end) <= 0.0:
                issues.append(f"T_end={t_end} must be > 0.")
            elif dt is not None and float(dt) > 0.0:
                if float(t_end) < float(dt):
                    issues.append(
                        f"T_end={t_end} < dt={dt}: not even one time "
                        "step fits in the simulated interval.")
                else:
                    ratio = float(t_end) / float(dt)
                    if abs(ratio - round(ratio)) > 1e-8 * max(ratio, 1.0):
                        issues.append(
                            f"T_end={t_end} is not an integer multiple "
                            f"of dt={dt}: NUMSTEP is rounded and "
                            "MAXTIME becomes NUMSTEP*dt, so different "
                            "dt levels would end at different times. "
                            "Halve dt starting from a value that "
                            "divides T_end exactly.")

        for key in ("kappa", "rho", "c"):
            v = _num(key)
            if v is not None and float(v) <= 0.0:
                issues.append(
                    f"{key}={v} must be > 0 (kappa: conductivity, "
                    "rho: density, c: specific heat; CAPA = rho*c).")

        for key in ("lx", "ly"):
            v = _num(key)
            if v is not None and float(v) <= 0.0:
                issues.append(f"{key}={v} must be > 0.")

        n = _num("n")
        if n is not None:
            if int(n) < 2:
                issues.append(
                    f"n={n} is too coarse — need at least 2 elements "
                    "per side to represent the manufactured mode.")
            elif int(n) > 96:
                issues.append(
                    f"n={n} exceeds the mesh cap; the generator clamps "
                    "n to 96 (fixed-mesh temporal studies do not need "
                    "more).")

        for key in ("mode_x", "mode_y"):
            v = _num(key)
            if v is not None and int(v) < 1:
                issues.append(f"{key}={v} must be a positive integer "
                              "(spatial mode number).")

        omega = _num("omega")
        if omega is not None and float(omega) <= 0.0:
            issues.append(
                f"omega={omega} must be > 0 — with no time content the "
                "temporal error vanishes and the dt study only "
                "measures the spatial floor.")

        profile = params.get("time_profile")
        if profile is not None and profile not in ("cos", "exp"):
            issues.append(
                f"time_profile={profile!r} not supported — use 'cos' "
                "(f = cos(omega t)) or 'exp' (f = exp(-omega t)).")

        amp = _num("amp")
        grad_amp = _num("grad_amp")
        if (amp is not None and float(amp) == 0.0
                and grad_amp is not None and float(grad_amp) == 0.0):
            issues.append(
                "amp=0 and grad_amp=0: the manufactured solution "
                "degenerates to a constant — nothing to converge on.")

        return issues
