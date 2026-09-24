"""Kratos Multiphysics participant for the openPASO `couple` driver (NEUMANN side).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is mandatory), writes exports.json LAST and exits 0.
Needs KratosMultiphysics + ConvectionDiffusionApplication importable in the
interpreter named in `command`. DO NOT GUESS THAT INTERPRETER: take it from
`discover(query='list')`, which reports the one this install actually imports
Kratos with. A system python3 is a common trap — it usually exists, so nothing
looks wrong, and it raises ModuleNotFoundError because the Kratos build targets
a different Python version than the system one.

THE OTHER HALF OF participant_kratos.py. That file is the DIRICHLET side: it
imports the partner's `values` (interface temperature), fixes them as nodal
TEMPERATURE, and exports the consistent REACTION_FLUX. This file is the NEUMANN
side: it imports the partner's `normal_fluxes` and applies them as the natural
boundary condition, then exports the interface TEMPERATURE its solve produced,
which is what a Dirichlet partner consumes.

Physics: steady conduction  -div(K grad T) = f  on one rectangular subdomain of
a domain split by a straight interface at x = IFACE_X. Top and bottom edges are
natural (zero flux). The non-interface x-boundary carries a Dirichlet value
T_OUTER.

  x = OUTER_X : Dirichlet T = T_OUTER
  x = IFACE_X : INTERFACE, natural BC = the partner's imported flux
  top/bottom  : insulated (natural, nothing to do)

THE SIGN, WHICH IS THE ONLY THING A NEUMANN PARTICIPANT CAN GET SILENTLY WRONG.
Every participant exports its outward normal flux density with respect to ITS
OWN outward normal,
      q = -(K grad T) . n_own,
so on a shared interface the two sides carry OPPOSITE signs. Integrating the
weak form on THIS subdomain,
      int_Omega K grad T . grad v  =  int_Omega f v  +  int_dOmega (K grad T . n) v ds,
and on the interface n = n_own = -n_partner, so
      K grad T . n_own = -q_own = +q_partner.
The partner's number is therefore applied UNCHANGED — no minus sign anywhere.
Kratos's FluxCondition2D2N enforces exactly  K grad T . n = FACE_HEAT_FLUX
(verified by a patch test: T fixed to 0 at x=0, FACE_HEAT_FLUX=1 on x=1, K=1,
no source -> T(1)=+1.000000, i.e. dT/dx=+1 not -1), so

      FACE_HEAT_FLUX  :=  the imported `normal_fluxes`, verbatim.

If you ever flip that sign to "make the temperatures look right", you have
built a coupling that drives heat the wrong way across the interface and still
converges. The self-check printed at the end of this script is the guard: it
evaluates the discrete divergence theorem on this subdomain and must come out
at round-off.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER   = "left"        # the partner's `name` in your couple(...) call
X0, X1    = 0.6, 1.0      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.6           # the shared interface; X0/X1 for axis "x", Y0/Y1 for axis "y"
K         = 1.6           # conductivity of THIS subdomain


def F_SRC(x, y):
    """Volumetric source f in  -div(K grad T) = f, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above.
    A CONSTANT CANNOT REPRESENT A POLYNOMIAL SOURCE: if your problem states
    one, or you derived it from a manufactured solution, a single number here
    silently solves a different problem. With the whole outer boundary
    prescribed and no source, the answer degenerates to the profile between
    the outer values.

    Unlike its FEniCSx and DUNE siblings, this one is called ONCE PER NODE
    with SCALAR coordinates (see the SetSolutionStepValue loop below), so
    write it with plain math or NumPy scalars — do not assume arrays:

        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    """
    return 0.0 * x

T_OUTER   = 300.0         # Dirichlet value on the NON-interface x-boundary
FULL_OUTER_DIRICHLET = False  # True: T_OUTER also on y=Y0,Y1; corners stay outer
NX, NY    = 20, 16        # this subdomain's OWN mesh; need not match the partner
Q_INIT    = 0.0           # iteration-1 fallback interface flux density


def source(x, y):
    """Volumetric source at the nodes — the interpolation point for F_SRC.

    Sampling at the nodes is the P1 interpolant of the source, an O(h^2) load
    error, the same order as the discretization error, so it does not touch
    the second-order rate.
    """
    return F_SRC(x, y)
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides NX, NY and names the level; the per-level
#    dumps below carry that level so the coarse levels survive the fine ones.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))   # a multi-level call's level keys
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
ON_MAX_X = abs(IFACE_X - HI) < abs(IFACE_X - LO)   # interface at this side's MAX of that axis?
OUTER_X = LO if ON_MAX_X else HI
S = 1.0 if ON_MAX_X else -1.0          # outward normal at the interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)



def _partner_block(imp):
    """The partner's block from imports.json, by the name the driver actually used."""
    if not isinstance(imp, dict) or not imp:
        return None
    if PARTNER in imp:
        return imp[PARTNER] or None
    others = [k for k in imp if isinstance(imp.get(k), dict)]
    if len(others) == 1:     # named differently in couple(...) than here: read it, say so
        import sys as _sys
        print(f"NOTE: PARTNER is {PARTNER!r} but imports.json is keyed {others[0]!r} "
              f"-- reading that block; align PARTNER with the name in your "
              f"couple(...) call.", file=_sys.stderr)
        return imp[others[0]] or None
    if others:
        raise SystemExit(f"imports.json holds blocks named {others} and none is "
                         f"PARTNER={PARTNER!r}: set PARTNER to the partner's name "
                         f"in your couple(...) call.")
    return None


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text() or "{}"))
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[AL] for c in imp["coordinates"]], float)   # the coordinate ALONG the interface
    # `or []` and not `.get(key, [])`: a partner that writes the key with an
    # explicit null gets [] here instead of a TypeError out of np.asarray, and
    # falls through to the fallback like any other unusable import.
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


def build_model():
    """Structured triangulation of [X0,X1] x [Y0,Y1]; returns (mp, nid)."""
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    # FluxCondition2D2N reads its load from the SURFACE source variable named
    # here; leave it FACE_HEAT_FLUX or the condition assembles nothing and the
    # interface silently becomes insulated.
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
              KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    for j in range(NY + 1):
        for i in range(NX + 1):
            mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / NX,
                             Y0 + (Y1 - Y0) * j / NY, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props)
            eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props)
            eid += 1
    return mp, nid
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def main():
    """Solve once and write exports.json. Returns (model_part, node_index_map)
    so a verification script can `import` this file, call main(), and integrate
    the volume field against a manufactured solution — the coupling itself only
    ever needs the file handshake."""
    if min(abs(IFACE_X - X0), abs(IFACE_X - X1)) > TOL:
        sys.exit(f"IFACE_X={IFACE_X} is not an x-boundary of this subdomain "
                 f"[{X0},{X1}] — nothing is shared with the partner")
    # NX < 1 would put the interface column and the outer Dirichlet column on
    # the SAME nodes; the Dirichlet condition wins, the imported flux is
    # discarded, and the run still exits 0 with a plausible-looking export.
    if NX < 1 or NY < 1:
        sys.exit(f"NX,NY = {NX},{NY}: need at least one element in each "
                 "direction, and NX >= 1 so the interface and the outer "
                 "Dirichlet boundary do not land on the same nodes")

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    mp, nid = build_model()
    i_if = NX if ON_MAX_X else 0            # column index of the interface
    i_out = 0 if ON_MAX_X else NX           # column index of x = OUTER_X
    # The two column indices above are the ONLY place the geometry meets the
    # node numbering. If an edit to build_model ever reorders the grid they
    # would silently point at interior columns, the interface flux would be
    # applied inside the domain and the run would still exit 0. Check them
    # against the coordinates they are supposed to name.
    for idx, want, what in ((i_if, IFACE_X, "interface"),
                            (i_out, OUTER_X, "outer Dirichlet")):
        got = mp.Nodes[nid[(idx, 0)]].X
        if abs(got - want) > TOL:
            sys.exit(f"internal: the {what} column sits at x={got}, not "
                     f"x={want} — the mesh and the column indices disagree")
    y_if = np.array([Y0 + (Y1 - Y0) * j / NY for j in range(NY + 1)])
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    q_in = sample(read_imports(), "normal_fluxes", Q_INIT, y_if)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    for n in mp.Nodes:
        n.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(n.X, n.Y)))
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

    for j in range(NY + 1):                 # outer x Dirichlet boundary
        n = mp.Nodes[nid[(i_out, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
        n.Fix(KM.TEMPERATURE)
    if FULL_OUTER_DIRICHLET:
        # The interface corners lie on y=Y0,Y1, so they remain outer-boundary
        # Dirichlet nodes even though adjacent interface facets carry flux.
        for i in range(NX + 1):
            for j in (0, NY):
                n = mp.Nodes[nid[(i, j)]]
                n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
                n.Fix(KM.TEMPERATURE)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # ── the interface: the partner's flux, applied UNCHANGED (see the header) ──
    for j in range(NY + 1):
        # THE NODAL VALUE IS THE LOAD. FluxCondition2D2N integrates FACE_HEAT_FLUX
        # from its NODES; a condition created without this SetSolutionStepValue
        # assembles zero, exits 0 and is reported unresponsive (measured 2026-09-11).
        mp.Nodes[nid[(i_if, j)]].SetSolutionStepValue(
            KM.FACE_HEAT_FLUX, float(q_in[j]))
    props = mp.GetProperties()[1]
    for j in range(NY):
        mp.CreateNewCondition("FluxCondition2D2N", j + 1,
                              [nid[(i_if, j)], nid[(i_if, j + 1)]], props)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    # AddDof with a REACTION variable: without the second argument the fixed
    # dofs have nowhere to store their reaction and it is silently discarded.
    # This side does not export the reaction, but the conservation self-check
    # below is built from it.
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    # arg 4 is CalculateReactionsFlag: True, or the self-check has no reactions.
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    # TWO THINGS YOUR SOLVE ABOVE MUST DO, or the self-check below reads zeros:
    #   * AddDof(TEMPERATURE, REACTION_FLUX, mp) -- the SECOND argument gives
    #     every fixed dof a place to store its reaction; without it the
    #     reaction is silently discarded.
    #   * run the strategy with CalculateReactionsFlag=True (the 4th positional
    #     argument of ResidualBasedLinearStrategy); the conservation self-check
    #     is built from REACTION_FLUX.

    # ── served: the trace is read from the nodes AT x = IFACE_X, checked by
    #    coordinate before anything is exported. Measured 2026-09-11 on three
    #    worker scripts written from this contract: two inverted the ON_MAX_X
    #    test inside their own solve, exported the OUTER column's temperature
    #    under the interface's coordinates, and nothing else was wrong -- the
    #    flux, assembled from the conditions, was right in all three. A wrong
    #    column exits 0 and couples on; this stops it.
    _if_nodes = [mp.Nodes[nid[(i_if, j)]] for j in range(NY + 1)]
    _off = [n for n in _if_nodes if abs(n.X - IFACE_X) > TOL]
    if _off:
        sys.exit(f"EXPORT SELF-CHECK: {len(_off)} of {len(_if_nodes)} nodes read as the "
                 f"interface column sit at x={_off[0].X:.6g}, not x={IFACE_X}: i_if names "
                 f"the wrong column. The interface column is the one whose x equals "
                 f"IFACE_X: ON_MAX_X = abs(IFACE_X - X1) < abs(IFACE_X - X0), then "
                 f"i_if = NX if ON_MAX_X else 0 and i_out = 0 if ON_MAX_X else NX")
    T = np.array([n.GetSolutionStepValue(KM.TEMPERATURE) for n in _if_nodes])

    # ── this side's own outward normal flux, from what the CONDITIONS assembled
    #
    # THE ARGUMENT THIS REPLACES WAS HALF RIGHT. It said the reaction formula
    # must not be used on a Neumann side because those interface dofs are free,
    # so r = A u - b is ~0 there and -r/w would export zero. True — but only
    # because Kratos stores REACTION_FLUX on FIXED dofs alone, and only when
    # the residual is taken against a load that ALREADY CONTAINS the interface
    # term. It then fell back to an L2 projection of the elementwise P1
    # gradient, defended as "acceptable here and only here, because a Dirichlet
    # partner reads this participant's values, never its normal_fluxes".
    #
    # Any reader of the result reads them. The two-sided interface jump is a GATE on a
    # coupled cell, and this side's export is half of it.
    #
    # MEASURED on the 3-D sibling, which had the identical construction, by
    # imposing q(y,z) = 2 + 3 sin(4y) cos(3z) and asking for it back:
    #     gradient averaging  1.27e+00, 8.37e-01, 3.95e-01   order 0.60, 1.08
    #     assembled conditions 5.16e-01, 1.77e-01, 4.77e-02  order 1.55, 1.89
    # The projection is O(h) where the field is O(h^2), so it, and not the
    # physics, sets the interface order any reader will measure.
    #
    # AN ECHO OF q_in WOULD NOT DO EITHER, though it is algebraically the exact
    # answer here: FluxCondition2D2N enforces K grad T . n = FACE_HEAT_FLUX and
    # FACE_HEAT_FLUX is the partner's array verbatim. An echo never passes
    # through the discretisation, so applied on the wrong facets or with the
    # wrong sign it reads the same and the balance gate still reports roundoff.
    # Summing each condition's own right-hand side is a MEASUREMENT of what
    # entered the linear system: it is int_Gamma g phi_i ds when the interface
    # is built correctly and something else the moment it is not. On those rows
    # the discrete equation gives r_i = +(assembled interface load), so the
    # outward density is -r_i/w_i — the same expression the Dirichlet side uses.
    info = mp.ProcessInfo
    rhs_i = np.zeros(len(mp.Nodes) + 1)
    w_i = np.zeros(len(mp.Nodes) + 1)
    for cond in mp.Conditions:
        vec = KM.Vector()
        cond.CalculateRightHandSide(vec, info)
        nds = cond.GetNodes()
        # w_i = int_Gamma phi_i ds: in 2-D the interface facets are segments,
        # so a node's weight is half of each segment it belongs to.
        p0, p1 = nds[0], nds[1]
        seg = ((p1.X - p0.X) ** 2 + (p1.Y - p0.Y) ** 2) ** 0.5
        for k, nd in enumerate(nds):
            rhs_i[nd.Id] += float(vec[k])
            w_i[nd.Id] += 0.5 * seg
    ids_if = [nid[(i_if, j)] for j in range(NY + 1)]
    r_if = np.array([rhs_i[i] for i in ids_if])
    wq = np.array([w_i[i] for i in ids_if])
    Q = np.where(np.abs(wq) > 1e-14, -r_if / np.maximum(np.abs(wq), 1e-300)
                 * np.sign(np.where(wq == 0, 1.0, wq)), 0.0)
    if FULL_OUTER_DIRICHLET:
        # Corner reactions also contain the perpendicular outer-boundary flux
        # and cannot be separated into one interface contribution. Zero it: a corner reaction mixes the perpendicular outer flux and is not a clean interface datum. from grading; retain the points for exchange but not that mixed
        # reaction.
        Q[[0, -1]] = 0.0

    # ── CONSERVATION SELF-CHECK: the discrete divergence theorem ──────────────
    # Summing the unconstrained residual r = A u - b over ALL nodes gives
    # sum_i r_i = -sum_i b_i, because sum_i A_ij = int K grad(sum_i phi_i).grad
    # phi_j = 0 (the phi_i are a partition of unity). r vanishes on free rows,
    # so over the fixed rows alone
    #       sum_{outer} r_i  +  int_Omega f dOmega  +  int_Gamma q_applied ds  =  0
    # exactly, at round-off, for ANY mesh. It is not a discretisation check: it
    # fails only if the flux was applied with the wrong sign or magnitude, or
    # not applied at all — which is precisely the failure mode this side has.
    hy = (Y1 - Y0) / NY
    load_iface = float(hy * (0.5 * q_in[0] + q_in[1:-1].sum() + 0.5 * q_in[-1]))
    load_vol = 0.0
    for el in mp.Elements:
        nds = el.GetNodes()
        x = [n.X for n in nds]
        y = [n.Y for n in nds]
        det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
        load_vol += (0.5 * abs(det)) * sum(
            n.GetSolutionStepValue(KM.HEAT_FLUX) for n in nds) / 3.0
    react = sum(mp.Nodes[nid[(i_out, j)]].GetSolutionStepValue(KM.REACTION_FLUX)
                for j in range(NY + 1))
    imb_abs = abs(react + load_vol + load_iface)
    scale = max(abs(react), abs(load_vol), abs(load_iface))
    # On iteration 1 with Q_INIT = 0 and no source there is no heat flow at all,
    # every term is round-off, and a RATIO of round-off to round-off is O(1)
    # while meaning nothing. Say "trivial" instead of printing a 1.0 that reads
    # as a 100% conservation error.
    if scale <= 1e-10 * K * max(1.0, abs(T_OUTER)) * (Y1 - Y0):
        bal = f"balance trivial (no heat flow yet, |imbalance|={imb_abs:.3e})"
    else:
        bal = (f"balance |sum(reactions)+vol+iface| = {imb_abs:.3e} abs / "
               f"{imb_abs / scale:.3e} rel")

    print(f"[kratos neumann] interface n={len(T)} "
          f"q_applied=[{q_in.min():.6g},{q_in.max():.6g}] "
          f"T=[{T.min():.6g},{T.max():.6g}] {bal}")
    print(f"\nNDOF = {len(mp.Nodes)}")

    # PER-LEVEL PERSISTENCE: this level's whole field and its interface trace
    # and flux, named by LEVEL, never overwritten by the next level (exports.json
    # is). Build the task's per-level files from these.
    # Interpolate THESE onto the probe points your task names. A file the next
    # level overwrites cannot carry a mesh study.
    # A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
    # proof that this participant succeeded, and it is written after these files,
    # so an exception here would throw away a coupling iteration that worked.
    try:
        with open(f"field_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,u\n")
            for n in mp.Nodes:
                _f.write(f"{float(n.X):.11e},{float(n.Y):.11e},{float(n.GetSolutionStepValue(KM.TEMPERATURE)):.11e}\n")
        with open(f"interface_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,u,qn\n")
            for n, t, q in zip(_if_nodes, T, Q):
                _f.write(f"{float(n.X):.11e},{float(n.Y):.11e},{float(t):.11e},{float(q):.11e}\n")
    except Exception as _dump_exc:
        # AND LEAVE NO HALF-WRITTEN FILE BEHIND. `open(..., "w")` truncates
        # before it fails, so a dump that died mid-way leaves a header-only
        # CSV -- a file that looks like a submission and carries no rows.
        for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
            try:
                if Path(_partial).is_file() and len(
                        Path(_partial).read_text().splitlines()) <= 1:
                    Path(_partial).unlink()
            except OSError:
                pass
        print(f"[kratos_neumann per-level dump] level {LEVEL} dump failed: "
              f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
              f"continues, but this level has no field file to hand in. Fix the\n"
              f"names the dump reads and run this level again.")

    # exports.json LAST: the driver takes its existence as proof of success.
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(T)),
        # the coordinates of the SAME node objects the values were read from
        "coordinates": [[float(n.X), float(n.Y)] for n in _if_nodes],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))
    return mp, nid


if __name__ == "__main__":
    main()
