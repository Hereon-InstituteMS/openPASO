"""Helmholtz equation templates for deal.II.

Based on deal.II tutorial step-29 (complex-valued Helmholtz).
"""


def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    refinements = params.get("refinements", 5)
    omega = params.get("omega", 10.0)
    c = params.get("wave_speed", 1.0)
    return f'''\
/* Helmholtz equation on unit square — deal.II (step-29 inspired)
 * -laplacian(u) - k^2 * u = f  where k = omega / c
 * Complex-valued: split into real and imaginary parts.
 * Uses FESystem with 2 components: (u_re, u_im).
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/component_mask.h>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;

// Source term: point-like source (real part only)
template <int dim>
class HelmholtzRHS : public Function<dim>
{{
public:
  HelmholtzRHS() : Function<dim>(2) {{}}
  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
  {{
    values = 0;
    // Localized source in real part
    const double r2 = p.square();
    values[0] = std::exp(-50.0 * r2);  // real part source
    values[1] = 0.0;                    // imaginary part source
  }}
}};

int main()
{{
  const int dim = 2;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global({refinements});

  // System with 2 components: (u_real, u_imag)
  FESystem<dim> fe(FE_Q<dim>(1), 2);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Helmholtz DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  const double omega = {omega};
  const double c     = {c};
  const double k2    = (omega / c) * (omega / c);

  // Assemble: for each component pair
  // Real-real block: (grad u_re, grad v_re) - k^2 (u_re, v_re)
  // Imag-imag block: (grad u_im, grad v_im) - k^2 (u_im, v_im)
  // Cross terms are zero for real-valued k
  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dpc = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dpc, dpc);
  Vector<double> cell_rhs(dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);

  HelmholtzRHS<dim> rhs_function;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {{
          Vector<double> f_val(2);
          rhs_function.vector_value(fe_values.quadrature_point(q), f_val);

          for (unsigned int i = 0; i < dpc; ++i)
            {{
              const unsigned int ci = fe.system_to_component_index(i).first;
              for (unsigned int j = 0; j < dpc; ++j)
                {{
                  const unsigned int cj = fe.system_to_component_index(j).first;
                  if (ci == cj)
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q) *
                                          fe_values.shape_grad(j, q)
                                          - k2 * fe_values.shape_value(i, q) *
                                                 fe_values.shape_value(j, q))
                                         * fe_values.JxW(q);
                }}
              cell_rhs(i) += fe_values.shape_value(i, q) * f_val[ci] * fe_values.JxW(q);
            }}
        }}

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dpc; ++i)
        {{
          for (unsigned int j = 0; j < dpc; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // Boundary conditions: u = 0 on all boundaries (both components)
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(2), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  // Solve — indefinite system, use GMRES
  SolverControl solver_control(5000, 1e-10);
  SolverGMRES<Vector<double>> solver(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  std::cout << "Helmholtz: " << solver_control.last_step() << " GMRES iterations" << std::endl;

  // Output
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> names = {{"u_real", "u_imag"}};
  data_out.add_data_vector(solution, names);
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Helmholtz equation: -laplacian(u) - k^2*u = f (step-29 inspired)",
    "tutorial_steps": ["step-29 (complex-valued Helmholtz with absorbing BCs)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(1), 2) — real and imaginary parts as components",
    "solver": "GMRES + SSOR (indefinite system due to -k^2 mass term)",
    "elements": {
        "FESystem":
            "Real+imag pair on a real-valued solver. "
            "FESystem<dim>(FE_Q<dim>(degree), 2) is the canonical "
            "complex-as-real-2-component formulation for scalar "
            "acoustic Helmholtz.",
        "FE_Q":
            "Single-component formulation on a complex-valued "
            "matrix (PETSc / Trilinos complex build). degree ≥ 2 "
            "strongly recommended at high wavenumber k to "
            "mitigate the pollution effect.",
        "FE_Q_Hierarchical":
            "Required for hp-adaptive refinement to counteract "
            "the pollution effect at high k.",
        "FE_Nedelec":
            "H(curl) element for VECTOR Helmholtz (time-harmonic "
            "Maxwell E-field). Do NOT pick for scalar acoustic "
            "Helmholtz — that uses FE_Q.",
    },
    "mesh_generators": {
        "hyper_cube": "Closed-cavity acoustics with sound-hard (Neumann) BCs.",
        "hyper_ball": "Radial acoustic radiation; pair with absorbing BC on outer boundary.",
        "hyper_shell": "Annular waveguide / acoustic resonator.",
        "hyper_cube_with_cylindrical_hole": "Scattering off a cylinder; classic radiation test.",
        "cheese": "Scattering by array of holes (sonic-crystal demos).",
        "extrude_triangulation": "2D acoustic mesh → 3D waveguide; ≥ 8 slices per wavelength.",
    },
    "solvers": [
        "SolverGMRES<>                — default for indefinite Helmholtz",
        "SolverMinRes<>                — symmetric indefinite variant; only when the bilinear form stays symmetric",
        "TrilinosWrappers::SolverDirect (UMFPACK / MUMPS) — ground truth at high k where iterative methods struggle",
        "Complex shifted-Laplacian preconditioner — preconditions (K - k^2 M) with (K - (k^2 + i*shift) M); shift damps oscillations enough for multigrid",
    ],
    "pitfalls": [
        "[Numerical] System is INDEFINITE once k^2 exceeds the "
        "smallest Laplace eigenvalue — SolverCG then has no "
        "convergence guarantee, so prefer SolverGMRES / "
        "SolverMinRes / a direct solver. But CG does NOT announce "
        "this. Signal: there is none from the solver — check "
        "definiteness directly (a few hundred power iterations on "
        "A and on sigma*I - A give lambda_max and lambda_min) "
        "rather than waiting for SolverCG to complain. Measured on "
        "deal.II 9.8 on the "
        "matrix this template assembles (omega=10, c=1, 5 global "
        "refinements, 2178 DoFs): a power-iteration estimate gives "
        "lambda_max = 3.85 and lambda_min = -0.37, i.e. genuinely "
        "INDEFINITE — and SolverCG with PreconditionSSOR still "
        "CONVERGED in 126 iterations to residual 7.0e-11, while "
        "SolverGMRES on the same system took 1718. There was no "
        "breakdown of any kind. (This entry used to quote "
        "\"SolverCG reports 'breakdown' immediately\"; it does "
        "not.) Test definiteness explicitly if it matters; do not "
        "use CG's silence as evidence the system is SPD.",
        "[Syntax] Complex splitting: 2-component FESystem<dim>(FE_Q, 2) "
        "with (u_re, u_im) doubles the DOF count and requires the "
        "bilinear form to assemble the 2x2 block carrying the "
        "imaginary coupling. Forgetting the off-diagonal coupling "
        "decouples real and imaginary parts and the absorbing BC "
        "silently becomes a reflecting wall. Signal: "
        "`solution.block(1).linfty_norm()` (the imaginary "
        "component) is exactly 0 despite the source f being "
        "complex; DataOut shows a real-valued standing-wave "
        "pattern instead of a complex outgoing wave.",
        "[Physics] For absorbing BCs (Sommerfeld radiation condition): "
        "add boundary integral -i*k*u*v*dS on the outer boundary. "
        "Missing the i (so adding -k*u*v*dS instead) gives a real "
        "lossy boundary, not a radiating one. Signal: DataOut shows "
        "wave amplitude DECAYING from the source toward the outer "
        "boundary (drops to <0.1 of peak amplitude at the boundary) "
        "instead of maintaining constant amplitude along a radial "
        "ray; comparison against Hankel-function reference gives "
        "L2-error of O(1).",
        "[Numerical] High frequency (large k) — need ~10 DOFs per "
        "wavelength minimum (h < lambda/10). For accurate amplitude "
        "at k > 50 use 20 DOFs/wavelength OR higher polynomial order. "
        "Signal: VectorTools::integrate_difference for the FE_Q "
        "solution vs the analytic Hankel-function reference shows "
        "amplitude error of factor 2-5x at the scattering centre; "
        "refining h further (well below lambda/10) does NOT close "
        "the gap (the resolution is fine; the polynomial order is "
        "wrong for the wavenumber).",
        "[Numerical] Pollution effect — phase error grows as "
        "O(k^{p+1} h^{2p+1}) where p is the FE polynomial order. At "
        "k > 100 even 10 DOFs/wavelength gives visibly wrong phase. "
        "Mitigations: higher-order p, hp-FEM, or stabilised methods "
        "(CIP, GLS). Signal: VectorTools::integrate_difference vs a "
        "Hankel-function reference shows amplitude error stays at "
        "10-20% but phase error grows like k^{p+1} h^{2p+1}; "
        "DataOut node positions are correctly spaced (h < lambda/10) "
        "but appear shifted by a constant fraction of a wavelength "
        "relative to the analytic solution.",
        "[Integration] PML (perfectly matched layer) modifies "
        "coefficients in a thin shell around the domain to absorb "
        "outgoing waves; the modification is COMPLEX-valued. A "
        "real-coefficient 'PML' is not a PML. Signal: DataOut shows "
        "the outgoing wave reflecting back into the interior — "
        "standing-wave pattern visible with `solution.linfty_norm()` "
        "rising periodically rather than monotonically decaying; "
        "reflection coefficient measured at the PML interface "
        "exceeds 1% (a real PML achieves <1e-4).",
    ],
    "materials": {
        "omega": {"range": [0.1, 1000.0], "unit": "rad/s (angular frequency)"},
        "wave_speed": {"range": [0.1, 10000.0], "unit": "m/s"},
    },
}
