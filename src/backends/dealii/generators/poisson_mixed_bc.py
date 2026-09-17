"""3D Poisson with MIXED Dirichlet-Neumann boundaries + MMS convergence.

Extension of the Poisson family (see poisson.py): unit cube [0,1]^3,
manufactured solution (MMS)

    u*(x,y,z) = amp * sin(a*pi*x) * sin(b*pi*y) * sin(c*pi*z) + d*x*y*z

with exact Dirichlet data u = u* on some faces and exact Neumann data
g = grad(u*) . n on the remaining faces, f = -laplacian(u*), FE_Q(degree),
a global-refinement convergence loop and a per-cycle L2 error printed as

    cycle <c> dofs <n> L2 <err>

Theoretical L2 order for FE_Q(k) is k+1. The order a run actually attains
is an OUTPUT of that run — read it off the per-cycle L2 errors rather
than assuming it.

DESIGN PRINCIPLE (project rule "no anchoring"): every problem dimension
is a parameter with placeholder defaults — MMS coefficients/frequencies,
polynomial degree, number of cycles, and which faces carry Neumann data.
Nothing is hard-coded to a specific evaluation instance.

Derivation of the exact data (checked with sympy 1.14.0):

    grad(u*) = ( amp*a*pi*cos(a*pi*x)*sin(b*pi*y)*sin(c*pi*z) + d*y*z,
                 amp*b*pi*sin(a*pi*x)*cos(b*pi*y)*sin(c*pi*z) + d*x*z,
                 amp*c*pi*sin(a*pi*x)*sin(b*pi*y)*cos(c*pi*z) + d*x*y )

    -laplacian(u*) = amp*pi^2*(a^2+b^2+c^2)
                     * sin(a*pi*x)*sin(b*pi*y)*sin(c*pi*z)

(the trilinear part d*x*y*z is harmonic). The Neumann datum is evaluated
in the template as gradient(u*) . n with the outward normal supplied by
FEFaceValues::normal_vector(q) — no per-face sign bookkeeping, so it is
exact for any Neumann-face subset.

Boundary ids follow GridGenerator::hyper_cube(tria, 0, 1, colorize=true):
    0: x=0   1: x=1   2: y=0   3: y=1   4: z=0   5: z=1
"""

from __future__ import annotations

import math

_ALL_FACES = (0, 1, 2, 3, 4, 5)

# Placeholder defaults — NOT tuned to any evaluation instance.
_DEFAULTS: dict = {
    "degree": 1,                 # FE_Q polynomial degree k
    "cycles": 4,                 # global-refinement cycles
    "initial_refinements": 1,    # refine_global before cycle 0
    "a": 1.0,                    # frequency along x
    "b": 1.0,                    # frequency along y
    "c": 1.0,                    # frequency along z
    "d": 1.0,                    # trilinear (harmonic) coefficient
    "amp": 1.0,                  # amplitude of the sin product
    "neumann_faces": [1, 3, 5],  # boundary ids with Neumann data
}


def validate_parameters(params: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = OK).

    Checks the merged (defaults + user) parameter set, so a partial
    params dict is fine.
    """
    problems: list[str] = []
    merged = {**_DEFAULTS, **(params or {})}

    def _is_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _is_num(v) -> bool:
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v))

    degree = merged["degree"]
    if not _is_int(degree):
        problems.append(f"degree must be an integer, got {degree!r}")
    elif not 1 <= degree <= 6:
        problems.append(f"degree must be in [1, 6], got {degree}")

    cycles = merged["cycles"]
    if not _is_int(cycles):
        problems.append(f"cycles must be an integer, got {cycles!r}")
    elif not 1 <= cycles <= 10:
        problems.append(f"cycles must be in [1, 10], got {cycles}")

    init_ref = merged["initial_refinements"]
    if not _is_int(init_ref):
        problems.append("initial_refinements must be an integer, "
                        f"got {init_ref!r}")
    elif not 0 <= init_ref <= 6:
        problems.append(f"initial_refinements must be in [0, 6], "
                        f"got {init_ref}")

    # Mesh-size guard: finest mesh has 2^(3*(initial_refinements +
    # cycles - 1)) cells — cap the exponent to keep the serial
    # CG/SSOR solve tractable.
    if _is_int(cycles) and _is_int(init_ref) and init_ref + cycles > 8:
        problems.append(
            "initial_refinements + cycles must be <= 8 "
            f"(finest 3D mesh would have 2^(3*"
            f"{init_ref + cycles - 1}) cells), got "
            f"{init_ref} + {cycles}")

    for name in ("a", "b", "c", "d", "amp"):
        v = merged[name]
        if not _is_num(v):
            problems.append(
                f"coefficient {name!r} must be a finite number, got {v!r}")

    if (_is_num(merged["amp"]) and _is_num(merged["d"])
            and merged["amp"] == 0 and merged["d"] == 0):
        problems.append(
            "amp and d are both 0 — the manufactured solution is "
            "identically zero, so the convergence study is meaningless")

    faces = merged["neumann_faces"]
    if isinstance(faces, (list, tuple, set)):
        face_list = list(faces)
        bad = [f for f in face_list if not _is_int(f) or f not in _ALL_FACES]
        if bad:
            problems.append(
                f"neumann_faces entries must be integer boundary ids in "
                f"0..5 (hyper_cube colorize convention), got {bad!r}")
        elif len(set(face_list)) != len(face_list):
            problems.append(
                f"neumann_faces contains duplicates: {face_list!r}")
        elif len(set(face_list)) == len(_ALL_FACES):
            problems.append(
                "all 6 faces are Neumann — the pure-Neumann Poisson "
                "problem is singular (solution only defined up to a "
                "constant); keep at least one Dirichlet face")
    else:
        problems.append(
            f"neumann_faces must be a list/tuple/set of boundary ids, "
            f"got {faces!r}")

    return problems


def _poisson_3d_mixed_bc(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set
    values appropriate to the specific problem being solved.
    Mixed Dirichlet-Neumann MMS convergence study (step-7 pattern).
    """
    problems = validate_parameters(params)
    if problems:
        raise ValueError(
            "invalid parameters for poisson_3d_mixed_bc: "
            + "; ".join(problems))

    merged = {**_DEFAULTS, **(params or {})}
    degree = int(merged["degree"])
    cycles = int(merged["cycles"])
    init_ref = int(merged["initial_refinements"])
    a_lit = repr(float(merged["a"]))
    b_lit = repr(float(merged["b"]))
    c_lit = repr(float(merged["c"]))
    d_lit = repr(float(merged["d"]))
    amp_lit = repr(float(merged["amp"]))
    neumann = sorted(set(int(f) for f in merged["neumann_faces"]))
    dirichlet = [f for f in _ALL_FACES if f not in neumann]
    neumann_set_lit = "{" + ", ".join(str(f) for f in neumann) + "}"
    neumann_comment = ", ".join(str(f) for f in neumann) if neumann \
        else "(none)"
    dirichlet_comment = ", ".join(str(f) for f in dirichlet)

    return f'''\
/* Poisson equation on the unit cube [0,1]^3 with MIXED
 * Dirichlet-Neumann boundary conditions and a manufactured-solution
 * (MMS) convergence study — deal.II, step-7 pattern.
 *
 *   u*(x,y,z) = amp*sin(a*pi*x)*sin(b*pi*y)*sin(c*pi*z) + d*x*y*z
 *   f = -laplacian(u*) = amp*pi^2*(a^2+b^2+c^2)*sin(a*pi*x)*sin(b*pi*y)*sin(c*pi*z)
 *
 * Boundary ids (hyper_cube colorize): 0:x=0 1:x=1 2:y=0 3:y=1 4:z=0 5:z=1
 * Neumann faces:   {neumann_comment}  (g = grad(u*) . n, exact)
 * Dirichlet faces: {dirichlet_comment}  (u = u*, exact)
 *
 * Expected L2 convergence order for FE_Q({degree}): {degree + 1}
 * Machine-readable per-cycle output:  cycle <c> dofs <n> L2 <err>
 */
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>

using namespace dealii;

/* Manufactured solution parameters */
const double mms_amp = {amp_lit};
const double mms_a   = {a_lit};
const double mms_b   = {b_lit};
const double mms_c   = {c_lit};
const double mms_d   = {d_lit};

class ExactSolution : public Function<3>
{{
public:
  virtual double value(const Point<3> &p,
                       const unsigned int /*component*/ = 0) const override
  {{
    return mms_amp * std::sin(mms_a * numbers::PI * p[0])
                   * std::sin(mms_b * numbers::PI * p[1])
                   * std::sin(mms_c * numbers::PI * p[2])
           + mms_d * p[0] * p[1] * p[2];
  }}

  virtual Tensor<1, 3> gradient(const Point<3> &p,
                                const unsigned int /*component*/ = 0) const override
  {{
    const double sx = std::sin(mms_a * numbers::PI * p[0]);
    const double cx = std::cos(mms_a * numbers::PI * p[0]);
    const double sy = std::sin(mms_b * numbers::PI * p[1]);
    const double cy = std::cos(mms_b * numbers::PI * p[1]);
    const double sz = std::sin(mms_c * numbers::PI * p[2]);
    const double cz = std::cos(mms_c * numbers::PI * p[2]);
    Tensor<1, 3> grad;
    grad[0] = mms_amp * mms_a * numbers::PI * cx * sy * sz
              + mms_d * p[1] * p[2];
    grad[1] = mms_amp * mms_b * numbers::PI * sx * cy * sz
              + mms_d * p[0] * p[2];
    grad[2] = mms_amp * mms_c * numbers::PI * sx * sy * cz
              + mms_d * p[0] * p[1];
    return grad;
  }}
}};

class RightHandSide : public Function<3>
{{
public:
  virtual double value(const Point<3> &p,
                       const unsigned int /*component*/ = 0) const override
  {{
    /* f = -laplacian(u*); the d*x*y*z part is harmonic. */
    return mms_amp * numbers::PI * numbers::PI
           * (mms_a * mms_a + mms_b * mms_b + mms_c * mms_c)
           * std::sin(mms_a * numbers::PI * p[0])
           * std::sin(mms_b * numbers::PI * p[1])
           * std::sin(mms_c * numbers::PI * p[2]);
  }}
}};

int main()
{{
  const unsigned int degree               = {degree};
  const unsigned int n_cycles             = {cycles};
  const unsigned int initial_refinements  = {init_ref};
  /* Neumann boundary ids; all other faces get Dirichlet u = u*. */
  const std::set<types::boundary_id> neumann_ids = {neumann_set_lit};

  ExactSolution exact_solution;
  RightHandSide rhs_function;

  Triangulation<3> triangulation;
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0, /*colorize=*/true);
  triangulation.refine_global(initial_refinements);

  FE_Q<3> fe(degree);
  DoFHandler<3> dof_handler(triangulation);

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {{
      if (cycle > 0)
        triangulation.refine_global(1);

      dof_handler.distribute_dofs(fe);

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      SparsityPattern sparsity_pattern;
      sparsity_pattern.copy_from(dsp);

      SparseMatrix<double> system_matrix;
      system_matrix.reinit(sparsity_pattern);
      Vector<double> solution(dof_handler.n_dofs());
      Vector<double> system_rhs(dof_handler.n_dofs());

      QGauss<3> quadrature(fe.degree + 2);
      QGauss<2> face_quadrature(fe.degree + 2);
      FEValues<3> fe_values(fe, quadrature,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);
      FEFaceValues<3> fe_face_values(fe, face_quadrature,
        update_values | update_quadrature_points |
        update_normal_vectors | update_JxW_values);

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {{
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;

          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {{
              const double f_q =
                rhs_function.value(fe_values.quadrature_point(q));
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {{
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                         fe_values.shape_grad(j, q) *
                                         fe_values.JxW(q);
                  cell_rhs(i) += fe_values.shape_value(i, q) * f_q *
                                 fe_values.JxW(q);
                }}
            }}

          /* Neumann surface term: + int_Gamma_N g v ds with
           * g = grad(u*) . n — evaluated with the outward normal from
           * FEFaceValues, so it is exact for any Neumann-face subset. */
          for (unsigned int face_no = 0;
               face_no < GeometryInfo<3>::faces_per_cell; ++face_no)
            if (cell->face(face_no)->at_boundary() &&
                neumann_ids.count(cell->face(face_no)->boundary_id()) > 0)
              {{
                fe_face_values.reinit(cell, face_no);
                for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                  {{
                    const double g_q =
                      exact_solution.gradient(
                        fe_face_values.quadrature_point(q)) *
                      fe_face_values.normal_vector(q);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) += g_q *
                                     fe_face_values.shape_value(i, q) *
                                     fe_face_values.JxW(q);
                  }}
              }}

          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {{
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));
              system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }}
        }}

      /* Dirichlet u = u* on every non-Neumann face.
       * Dirichlet faces here: {dirichlet_comment} */
      std::map<types::global_dof_index, double> boundary_values;
      for (types::boundary_id id = 0; id < 6; ++id)
        if (neumann_ids.count(id) == 0)
          VectorTools::interpolate_boundary_values(
            dof_handler, id, exact_solution, boundary_values);
      MatrixTools::apply_boundary_values(
        boundary_values, system_matrix, solution, system_rhs);

      SolverControl solver_control(10000,
                                   1e-12 * system_rhs.l2_norm());
      SolverCG<Vector<double>> solver(solver_control);
      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(system_matrix, 1.2);
      solver.solve(system_matrix, solution, system_rhs, preconditioner);

      Vector<float> difference_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(dof_handler, solution,
                                        exact_solution,
                                        difference_per_cell,
                                        QGauss<3>(fe.degree + 2),
                                        VectorTools::L2_norm);
      const double L2_error =
        VectorTools::compute_global_error(triangulation,
                                          difference_per_cell,
                                          VectorTools::L2_norm);

      std::cout << "cycle " << cycle
                << " dofs " << dof_handler.n_dofs()
                << " L2 " << std::setprecision(12) << std::scientific
                << L2_error << std::endl;

      if (cycle + 1 == n_cycles)
        {{
          DataOut<3> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(solution, "solution");
          data_out.build_patches();
          std::ofstream output("result.vtu");
          data_out.write_vtu(output);
          std::cout << "Output written to result.vtu" << std::endl;
        }}
    }}

  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": (
        "3D Poisson on the unit cube with MIXED Dirichlet-Neumann "
        "boundaries and a manufactured-solution (MMS) convergence "
        "study (deal.II step-7 pattern). Global refinement, per-cycle "
        "L2 error via VectorTools::integrate_difference + "
        "compute_global_error, machine-readable 'cycle <c> dofs <n> "
        "L2 <err>' output."),
    "tutorial_steps": ["step-7 (MMS + Neumann BC + convergence tables)",
                       "step-3/4 (basic Poisson assembly)"],
    "function_space": "FE_Q<3>(k) — Lagrange, any order k",
    # The measured order table that used to close this string was our own
    # development result — the answer to a convergence study, readable from
    # inside the tool that study evaluates. Removed 2026-08-06 by the
    # contamination merge gate. The rule below is theory, not measurement, and
    # the pre-asymptotic caveat is the genuinely useful part: it tells an agent
    # not to read the coarsest pair as a failure.
    "expected_order": (
        "THEORY: the L2 error of FE_Q(k) on this problem is "
        "O(h^(k+1)) under global refinement, so halving h should "
        "divide the error by 2^(k+1). That is the number to check "
        "your own run against — compute it from YOUR errors. This "
        "template has been run at degree 1 and degree 2 over five "
        "global-refinement cycles up to a few hundred thousand DoFs "
        "and the observed order approached the theoretical value from "
        "below in both cases, which is the expected behaviour. No "
        "error table is reproduced here on purpose: a table from "
        "somebody else's run is not evidence about yours, and "
        "comparing against a pasted number is how a wrong mesh, a "
        "wrong exact solution or a wrong quadrature rule goes "
        "unnoticed."),
    "mixed_bc_assembly": (
        "Neumann data enters the WEAK FORM ONLY, as the surface "
        "integral + int_{Gamma_N} g v ds added to the right-hand "
        "side. Assemble it with FEFaceValues on boundary faces whose "
        "boundary_id is in the Neumann set, using update_values | "
        "update_quadrature_points | update_normal_vectors | "
        "update_JxW_values. Evaluating g as grad(u*) . n with "
        "FEFaceValues::normal_vector(q) (outward unit normal) avoids "
        "per-face sign bookkeeping entirely. Dirichlet faces are "
        "handled separately via interpolate_boundary_values + "
        "apply_boundary_values (or an AffineConstraints object)."),
    "boundary_ids": (
        "GridGenerator::hyper_cube(tria, 0, 1, colorize=true) assigns "
        "face ids 0:x=0, 1:x=1, 2:y=0, 3:y=1, 4:z=0, 5:z=1. With "
        "colorize=false (the default) ALL faces get id 0 and a "
        "Dirichlet/Neumann split by id is impossible."),
    "error_measurement": (
        "VectorTools::integrate_difference(dof_handler, solution, "
        "exact, per_cell, QGauss<3>(fe.degree + 2), "
        "VectorTools::L2_norm) fills a per-cell Vector<float>; "
        "VectorTools::compute_global_error(tria, per_cell, L2_norm) "
        "reduces it to the global norm. Use a quadrature at least "
        "one order above the FE degree so the quadrature error does "
        "not pollute the measured convergence order."),
    "pitfalls": [
        "[Numerical] Forgetting the Neumann surface integral does NOT "
        "produce a crash or a solver failure — the program silently "
        "solves the problem with homogeneous Neumann data (g = 0) on "
        "those faces and converges to the WRONG solution. Signal: the "
        "per-cycle L2 error stalls at an O(1) mesh-independent level "
        "instead of decreasing at order k+1. Verified by deleting "
        "the surface term from this template and re-running the "
        "convergence study: the error stopped responding to "
        "refinement altogether — it stayed at the same O(1) level "
        "cycle after cycle, so the observed order sat at roughly zero "
        "instead of approaching k+1, while the solver reported "
        "healthy convergence at every cycle. Compute the observed "
        "order from YOUR OWN two finest cycles; an order near zero "
        "with a healthy solver is this bug.",
        "[Numerical] A wrong SIGN on the Neumann datum (g = -grad(u*) "
        ". n, or an inward normal) is the same silent failure mode: "
        "the run completes and the L2 error plateaus at a "
        "mesh-independent value. Signal: the per-cycle L2 error "
        "stops responding to refinement. Verified by flipping the "
        "sign of g in this template and re-running: the error "
        "plateaued at an O(1) mesh-independent level and the observed "
        "order sat at roughly zero, exactly as in the "
        "missing-term case — so the two bugs are NOT "
        "distinguishable by the convergence curve alone. What "
        "distinguishes them: with the wrong sign the error plateau is "
        "at a DIFFERENT level from the missing-term plateau, and the "
        "solution is visibly wrong on the Neumann faces in DataOut. "
        "Evaluate g with the outward normal from "
        "FEFaceValues::normal_vector(q) instead of hand-coding "
        "per-face normal components.",
        "[Syntax] GridGenerator::hyper_cube must be called with "
        "colorize=true to get distinct per-face boundary ids (0..5). "
        "With the default colorize=false every face has id 0. Signal: "
        "the Dirichlet loop over non-Neumann ids matches no faces (or "
        "all faces), and interpolate_boundary_values applied to ids "
        "1..5 is a no-op.",
        "[Numerical] Marking ALL faces Neumann makes the Poisson "
        "problem singular (solution defined only up to a constant): "
        "the CG solve may still 'converge' on the consistent part, "
        "but apply_boundary_values constrains nothing and the "
        "reported L2 error contains an arbitrary constant offset. "
        "Signal: validate_parameters() in this module returns a "
        "non-empty problem list for an all-Neumann spec, and if the "
        "guard is bypassed the per-cycle L2 error plateaus at an "
        "O(1) mesh-independent level exactly as in the "
        "missing-Neumann-term case. Keep at least one Dirichlet "
        "face.",
        "[Numerical] Use QGauss(fe.degree + 2) for "
        "integrate_difference, not degree + 1: the lower rule "
        "under-integrates the error of higher-order elements "
        "against a trigonometric exact solution and BIASES THE "
        "ERROR MAGNITUDE LOW. It does NOT visibly distort the "
        "observed ORDER, which is what this entry used to claim. "
        "Verified at degree 2 over four cycles, changing only the "
        "error-quadrature rule: the reported error MAGNITUDES "
        "differed by up to about a fifth, while the observed orders "
        "moved by only a few hundredths — far too little to notice. "
        "Signal: compare "
        "the same cycle's L2 value under QGauss(fe.degree + 1) and "
        "QGauss(fe.degree + 2) — a gap of more than a few percent "
        "means the lower rule is under-integrating. Do NOT use the "
        "observed order as the tell; it barely moves. Still use "
        "degree + 2 — it is cheap and it is the honest number.",
        "[Numerical] The linear-solver tolerance must sit well below "
        "the finest-cycle discretization error, otherwise the "
        "convergence curve flattens at the solver tolerance rather "
        "than the FE error. A relative SolverControl target of "
        "1e-12 * ||b|| was verified to keep the degree-2 orders "
        "clean all the way to the finest cycle of this study (a few "
        "hundred thousand DoFs), where the discretisation error is "
        "several orders of magnitude above that tolerance. Signal: "
        "the observed "
        "order collapses toward 0 on the FINEST cycles only, while "
        "the coarse cycles still show k+1 — and SolverControl::"
        "last_value() on those cycles sits at the requested "
        "tolerance rather than well below it.",
    ],
}
