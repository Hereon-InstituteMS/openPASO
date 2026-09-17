/* TRANSIENT heat conduction  rho_c dT/dt - div(k grad T) = f(x,y,t)  on ONE
 * rectangular subdomain of the canonical openPASO coupling problem (two subdomains
 * sharing a straight interface at x = iface_x).  Compiled partner of
 * participant_dealii_transient.py; the steady sibling is heat_iface_dealii.cc.
 *
 * Time integration is the theta-scheme:
 *     (M/dt + theta*K) u^{n+1} = (M/dt - (1-theta)*K) u^n
 *                                + theta*F^{n+1} + (1-theta)*F^n [+ N^{n+theta}]
 * theta = 0.5 is Crank-Nicolson and is SECOND ORDER; theta = 1 is backward
 * Euler, L-stable but first order.
 *
 * ONE RUN MARCHES THE WHOLE COUPLING WINDOW and exchanges the ENTIRE interface
 * trace — the WAVEFORM strategy, for the reasons set out at length in the
 * module docstring of participant_fenics_transient.py.  In particular the
 * `couple` contract has nowhere to put a time axis (InterfaceData carries five
 * keys and none of them is time), so a per-time-step exchange inside ONE
 * couple call is not expressible; and this executable must stay a PURE
 * FUNCTION of its input file, because the driver re-runs each participant on
 * identical imports to measure its noise floor and to probe its sensitivity.
 *
 * On the interface x = iface_x this participant is either
 *   side = 0 (DIRICHLET): T = T_if(y, t^{n+1})  per step, and it EXPORTS the
 *                         consistent (reaction) flux, THETA-AVERAGED over the
 *                         step — see the derivation at the recovery below;
 *   side = 1 (NEUMANN)  : + \int g(y) v ds added to the theta-combined RHS,
 *                         g = the partner's exported theta-averaged flux for
 *                         that step, applied UNCHANGED.
 *
 * Input file (argv[1]):
 *   line 1:  side k rho_c x0 x1 y0 y1 iface_x nx ny degree theta t_start t_end
 *            n_steps outer_faces
 *              outer_faces = 0: only the non-interface x-face is Dirichlet, the
 *                               two y-faces are natural (zero flux)
 *              outer_faces = 1: the WHOLE non-interface boundary is Dirichlet
 *   line 2:  T_initial(x,y)   muparser expression
 *   line 3:  T_outer(x,y,t)   muparser expression
 *   line 4:  f_src(x,y,t)     muparser expression
 *   line 5:  n_samples n_steps_trace
 *   then n_samples lines:  y  v_1 v_2 ... v_{n_steps_trace}
 *            (one row per interface sample point, one column per TIME STEP:
 *             column n is the partner's datum for the step t^n -> t^{n+1})
 *
 * A COMPILED BACKEND CANNOT TAKE A PYTHON LAMBDA, so the three problem
 * functions arrive as muparser expression STRINGS in the variables x, y, t.
 * That is the only real difference from the FEniCSx participant's EDIT block.
 *
 * Output file (argv[2]):
 *   line 1:  n_nodes n_steps
 *   then n_nodes lines, sorted by y:
 *            y  T^1 ... T^N  q^1 ... q^N
 *   with T^n the interface temperature at t^n and q^n the THETA-AVERAGED
 *   outward normal flux density of THIS subdomain over step n.
 *
 * All problem numbers come from the input file — nothing is hardcoded.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace dealii;

/* Piecewise-linear interpolant of samples v(y), clamped outside the sample
 * range — identical semantics to numpy.interp. */
class Sampled1D
{
public:
  Sampled1D() = default;
  Sampled1D(std::vector<double> ys, std::vector<double> vs)
    : ys_(std::move(ys)), vs_(std::move(vs))
  {}

  double operator()(const double y) const
  {
    if (ys_.empty())
      return 0.0;
    if (ys_.size() == 1 || y <= ys_.front())
      return vs_.front();
    if (y >= ys_.back())
      return vs_.back();
    const auto        it = std::upper_bound(ys_.begin(), ys_.end(), y);
    const std::size_t i  = std::distance(ys_.begin(), it);
    const double      w  = (y - ys_[i - 1]) / (ys_[i] - ys_[i - 1]);
    return (1.0 - w) * vs_[i - 1] + w * vs_[i];
  }

private:
  std::vector<double> ys_, vs_;
};

/* Sampled1D as a deal.II Function of y (for interpolate_boundary_values). */
class InterfaceFunction : public Function<2>
{
public:
  explicit InterfaceFunction(const Sampled1D &s)
    : s_(s)
  {}
  virtual double value(const Point<2> &p, const unsigned int = 0) const override
  {
    return s_(p[1]);
  }

private:
  const Sampled1D &s_;
};

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cerr << "usage: heat_iface_dealii_transient <input.txt> <output.txt>\n";
      return 1;
    }

  std::ifstream in(argv[1]);
  if (!in)
    {
      std::cerr << "cannot open input file " << argv[1] << "\n";
      return 1;
    }

  unsigned int side, nx, ny, degree, n_steps, outer_faces;
  double       k, rho_c, x0, x1, y0, y1, iface_x, theta, t_start, t_end;
  in >> side >> k >> rho_c >> x0 >> x1 >> y0 >> y1 >> iface_x >> nx >> ny >>
    degree >> theta >> t_start >> t_end >> n_steps >> outer_faces;
  std::string dummy, expr_init, expr_outer, expr_src;
  std::getline(in, dummy); // finish the header line
  std::getline(in, expr_init);
  std::getline(in, expr_outer);
  std::getline(in, expr_src);

  unsigned int n_samples = 0, n_steps_trace = 0;
  in >> n_samples >> n_steps_trace;
  std::vector<double>              sy(n_samples);
  std::vector<std::vector<double>> sv(n_samples,
                                      std::vector<double>(n_steps_trace, 0.0));
  for (unsigned int i = 0; i < n_samples; ++i)
    {
      in >> sy[i];
      for (unsigned int n = 0; n < n_steps_trace; ++n)
        in >> sv[i][n];
    }
  if (!in || n_samples == 0 || n_steps == 0)
    {
      std::cerr << "malformed input file\n";
      return 1;
    }
  /* THE ONE TIME-WINDOW ERROR THE PAYLOAD CAN REVEAL. The exchange carries no
   * time axis, so a partner configured with a different window is invisible
   * unless its trace LENGTH differs.  Refuse loudly rather than march on a
   * trace whose columns mean other times than this participant's steps. */
  if (n_steps_trace != n_steps)
    {
      std::cerr << "partner trace has " << n_steps_trace
                << " time levels, this participant's window has " << n_steps
                << " steps. Give both participants the same "
                   "t_start/t_end/n_steps/theta.\n";
      return 1;
    }

  const double dt = (t_end - t_start) / static_cast<double>(n_steps);

  /* Which colorized boundary id is the interface?  colorize=true gives
   * 0: x = x0, 1: x = x1, 2: y = y0, 3: y = y1. */
  const bool         iface_at_x1 = std::abs(iface_x - x1) < std::abs(iface_x - x0);
  const unsigned int iface_id    = iface_at_x1 ? 1 : 0;
  const unsigned int outer_id    = iface_at_x1 ? 0 : 1;
  const double       s_out       = iface_at_x1 ? 1.0 : -1.0; // outward normal = s * e_x

  FunctionParser<2> f_init(1), f_outer(1), f_src(1);
  try
    {
      const std::map<std::string, double> consts;
      f_init.initialize("x,y,t", expr_init, consts, /*time_dependent=*/true);
      f_outer.initialize("x,y,t", expr_outer, consts, true);
      f_src.initialize("x,y,t", expr_src, consts, true);
    }
  catch (const std::exception &e)
    {
      std::cerr << "cannot parse a problem expression: " << e.what() << "\n";
      return 1;
    }

  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {nx, ny}, Point<2>(x0, y0), Point<2>(x1, y1), /*colorize=*/true);

  const FE_Q<2> fe(degree);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  const unsigned int n_dofs = dof_handler.n_dofs();

  /* ONE sparsity pattern, UNCONSTRAINED.  The consistent interface flux below
   * is the residual of the discrete equations on the CONSTRAINED rows, so the
   * matrix it is read from must keep the constrained couplings; and because
   * the Dirichlet VALUES change every step while the constrained dof SET does
   * not, the cheap way to march is to keep the unconstrained system and
   * eliminate into a scratch copy per step (MatrixTools::apply_boundary_values)
   * rather than re-assembling with constraints n_steps times. */
  DynamicSparsityPattern dsp(n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);

  SparseMatrix<double> mass(sparsity), stiff(sparsity), A_free(sparsity),
    A_step(sparsity);
  Vector<double> solution(n_dofs), u_old(n_dofs), rhs_free(n_dofs),
    rhs_step(n_dofs), F_old(n_dofs), F_new(n_dofs), tmp(n_dofs);

  const QGauss<2> quadrature(degree + 1);
  FEValues<2>     fe_values(fe, quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const QGauss<1> face_quadrature(degree + 2);
  FEFaceValues<2> fe_face_rhs(fe, face_quadrature,
                              update_values | update_quadrature_points |
                                update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiff(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_mass  = 0.;
      cell_stiff = 0.;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              cell_mass(i, j) += rho_c * fe_values.shape_value(i, q) *
                                 fe_values.shape_value(j, q) * fe_values.JxW(q);
              cell_stiff(i, j) += k * fe_values.shape_grad(i, q) *
                                  fe_values.shape_grad(j, q) * fe_values.JxW(q);
            }
      cell->get_dof_indices(local_dofs);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            mass.add(local_dofs[i], local_dofs[j], cell_mass(i, j));
            stiff.add(local_dofs[i], local_dofs[j], cell_stiff(i, j));
          }
    }

  // A_free = M/dt + theta*K   (constant in time: dt and the material are)
  A_free.copy_from(mass);
  A_free *= 1.0 / dt;
  A_free.add(theta, stiff);

  /* w_i = \int_Gamma phi_i ds — the nodal interface weight.  Non-zero exactly
   * on the interface dofs, which is also how they are identified. */
  Vector<double>  weight(n_dofs);
  FEFaceValues<2> fe_face_w(fe, face_quadrature,
                            update_values | update_JxW_values);
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (cell->face(f)->at_boundary() &&
          cell->face(f)->boundary_id() == iface_id)
        {
          fe_face_w.reinit(cell, f);
          cell->get_dof_indices(local_dofs);
          for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              weight(local_dofs[i]) +=
                fe_face_w.shape_value(i, q) * fe_face_w.JxW(q);
        }

  std::vector<Point<2>> support_points(n_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler,
                                       support_points);

  /* The interface node list, built ONCE and in a fixed order (sorted by y):
   * the driver relaxes export vectors entry by entry, so the exported layout
   * must be identical on every iteration. */
  std::vector<types::global_dof_index> iface_dofs;
  for (types::global_dof_index i = 0; i < n_dofs; ++i)
    if (std::abs(weight(i)) > 1e-14)
      iface_dofs.push_back(i);
  std::sort(iface_dofs.begin(), iface_dofs.end(),
            [&](types::global_dof_index a, types::global_dof_index b) {
              return support_points[a][1] < support_points[b][1];
            });
  if (iface_dofs.empty())
    {
      std::cerr << "no interface dofs at x=" << iface_x << "\n";
      return 1;
    }

  /* An interface node that ALSO lies on the outer Dirichlet boundary carries
   * the OUTER reaction as well, so its residual is not this interface's flux.
   * Only relevant when outer_faces = 1 (the y-faces are Dirichlet too). */
  std::vector<types::boundary_id> outer_ids{
    static_cast<types::boundary_id>(outer_id)};
  if (outer_faces == 1)
    {
      outer_ids.push_back(2);
      outer_ids.push_back(3);
    }
  const IndexSet outer_dofs = DoFTools::extract_boundary_dofs(
    dof_handler, ComponentMask(),
    std::set<types::boundary_id>(outer_ids.begin(), outer_ids.end()));
  std::vector<char> suspect(iface_dofs.size(), 0);
  for (std::size_t i = 0; i < iface_dofs.size(); ++i)
    suspect[i] = outer_dofs.is_element(iface_dofs[i]) ? 1 : 0;
  std::vector<std::size_t> fixup(iface_dofs.size(), iface_dofs.size());
  for (std::size_t i = 0; i < iface_dofs.size(); ++i)
    if (suspect[i])
      {
        std::size_t best = iface_dofs.size();
        for (std::size_t j = 0; j < iface_dofs.size(); ++j)
          if (!suspect[j] &&
              (best == iface_dofs.size() ||
               std::abs(static_cast<long>(j) - static_cast<long>(i)) <
                 std::abs(static_cast<long>(best) - static_cast<long>(i))))
            best = j;
        fixup[i] = best;
      }

  /* Source functional F^m_i = \int f(.,t^m) phi_i dx. */
  auto assemble_source = [&](const double t, Vector<double> &out) {
    out = 0.;
    f_src.set_time(t);
    Vector<double> cell_rhs(dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_rhs = 0.;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const double fq = f_src.value(fe_values.quadrature_point(q));
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += fq * fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          out(local_dofs[i]) += cell_rhs(i);
      }
  };

  /* Nodal recovery of the outward normal flux by AVERAGING grad T over the
   * cells adjacent to each interface node.  NEUMANN SIDE ONLY (see below). */
  auto recover_gradient_flux = [&](const Vector<double> &u,
                                   std::vector<double>  &q) {
    Vector<double> acc(n_dofs), cnt(n_dofs);
    const Quadrature<1> face_support(fe.get_unit_face_support_points());
    FEFaceValues<2>     fe_face(fe, face_support,
                                update_values | update_gradients);
    std::vector<Tensor<1, 2>> face_grad(face_support.size());
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const unsigned int f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == iface_id)
          {
            fe_face.reinit(cell, f);
            fe_face.get_function_gradients(u, face_grad);
            cell->get_dof_indices(local_dofs);
            for (unsigned int qq = 0; qq < face_support.size(); ++qq)
              {
                const unsigned int i = fe.face_to_cell_index(qq, f);
                acc(local_dofs[i]) += -k * s_out * face_grad[qq][0];
                cnt(local_dofs[i]) += 1.0;
              }
          }
    q.resize(iface_dofs.size());
    for (std::size_t i = 0; i < iface_dofs.size(); ++i)
      q[i] = cnt(iface_dofs[i]) > 0 ? acc(iface_dofs[i]) / cnt(iface_dofs[i]) : 0.0;
  };

  // ── initial condition ────────────────────────────────────────────────────
  f_init.set_time(t_start);
  VectorTools::interpolate(dof_handler, f_init, u_old);
  assemble_source(t_start, F_old);
  std::vector<double> q_prev, q_new;
  if (side == 1)
    recover_gradient_flux(u_old, q_prev);

  std::vector<std::vector<double>> T_out(iface_dofs.size(),
                                         std::vector<double>(n_steps, 0.0));
  std::vector<std::vector<double>> Q_out(iface_dofs.size(),
                                         std::vector<double>(n_steps, 0.0));

  // ── the march. ONE run = the WHOLE window (waveform) ─────────────────────
  for (unsigned int n = 0; n < n_steps; ++n)
    {
      const double t_new = t_start + (n + 1) * dt;
      assemble_source(t_new, F_new);

      // rhs_free = (M/dt - (1-theta)K) u^n + theta F^{n+1} + (1-theta) F^n
      mass.vmult(rhs_free, u_old);
      rhs_free *= 1.0 / dt;
      stiff.vmult(tmp, u_old);
      rhs_free.add(-(1.0 - theta), tmp);
      rhs_free.add(theta, F_new, 1.0 - theta, F_old);

      // this step's column of the partner's trace
      std::vector<double> col(n_samples);
      for (unsigned int i = 0; i < n_samples; ++i)
        col[i] = sv[i][n];
      const Sampled1D samples(sy, col);

      if (side == 1)
        {
          /* NEUMANN: + \int g v ds on the interface, added to the THETA-COMBINED
           * right-hand side.  g is the partner's THETA-AVERAGED flux for this
           * step and is applied UNCHANGED — that average is exactly the
           * theta*N^{n+1} + (1-theta)*N^n the scheme asks for, so no time
           * interpolation happens anywhere. */
          Vector<double> cell_rhs(dofs_per_cell);
          for (const auto &cell : dof_handler.active_cell_iterators())
            for (const unsigned int f : cell->face_indices())
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == iface_id)
                {
                  fe_face_rhs.reinit(cell, f);
                  cell_rhs = 0.;
                  for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                    {
                      const double g =
                        samples(fe_face_rhs.quadrature_point(q)[1]);
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        cell_rhs(i) += g * fe_face_rhs.shape_value(i, q) *
                                       fe_face_rhs.JxW(q);
                    }
                  cell->get_dof_indices(local_dofs);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    rhs_free(local_dofs[i]) += cell_rhs(i);
                }
        }

      /* Dirichlet data at t^{n+1}.  INTERFACE FIRST, OUTER SECOND: a node on
       * both keeps the OUTER value (interpolate_boundary_values overwrites),
       * because with outer_faces = 1 the two interface end nodes sit on a
       * y-face that carries prescribed data in the un-split problem. */
      std::map<types::global_dof_index, double> bv;
      const InterfaceFunction iface_fun(samples);
      if (side == 0)
        VectorTools::interpolate_boundary_values(dof_handler, iface_id,
                                                 iface_fun, bv);
      f_outer.set_time(t_new);
      for (const auto id : outer_ids)
        VectorTools::interpolate_boundary_values(dof_handler, id, f_outer, bv);

      A_step.copy_from(A_free);
      rhs_step = rhs_free;
      MatrixTools::apply_boundary_values(bv, A_step, solution, rhs_step);

      SolverControl control(20000, 1e-13 * rhs_step.l2_norm() + 1e-16);
      SolverCG<Vector<double>>               solver(control);
      PreconditionSSOR<SparseMatrix<double>> precond;
      precond.initialize(A_step, 1.2);
      solver.solve(A_step, solution, rhs_step, precond);

      /* Outward normal flux density q = -(k grad T).n on the interface,
       * THETA-AVERAGED over this step.
       *
       * WHY NOT THE GRADIENT OF THE SOLUTION. The gradient of a P1/Q1 solution
       * is only O(h) accurate ON the boundary — the superconvergence points are
       * interior — and the boundary trace is exactly what the coupling reads.
       * Measured on a manufactured transient two-material problem with h and dt
       * refined together, this program fed the EXACT interface trace so the
       * coupling is not involved: the gradient recovery (side 1) converges at
       * 0.94 then 0.97, the consistent flux below (side 0) at 2.00 then 2.00.
       * The final-time field is 2.00 either way — the recovery is what the
       * PARTNER consumes, so a first-order recovery costs the partner's order,
       * not this side's, which is exactly why it is invisible from here.
       *   Worth knowing when comparing against the FEniCSx participant: on
       * these Q1 quadrilaterals the consistent flux is 2.00 at the interface
       * END nodes too, while participant_fenics_transient.py measures 1.03
       * there on P1 triangles for the same problem. Same formula, different
       * corner discretisation. Measure your own mesh rather than assuming
       * either result carries over.
       *
       * THE CONSISTENT (REACTION) FLUX, FOR A TIME-STEPPED SYSTEM.  From
       *     a(u,v) - (f,v) = \int_{dOmega}(k grad u . n) v ds = -\int_Gamma qn v ds
       * applied to the theta-scheme's OWN discrete equation, for every basis
       * function phi_i on the interface
       *     \int_Gamma [theta q^{n+1} + (1-theta) q^n] phi_i ds = -r_i,
       *     r = A_free u^{n+1} - rhs_free
       * with r the UNCONSTRAINED residual (A_free / rhs_free above: nothing
       * eliminated, constrained rows NOT zeroed, because on the Dirichlet side
       * those rows ARE the reaction).  So what comes out is the THETA-AVERAGE
       * of the flux over the step, NOT the flux at t^{n+1} — and that average
       * is precisely what the Neumann partner's own theta-combined RHS needs.
       * Dividing by w_i = \int_Gamma phi_i ds turns the functional into a
       * density the partner can interpolate pointwise.  The outward normal is
       * already in the identity, so no s_out factor appears.
       *
       * ONLY ON THE DIRICHLET SIDE.  On the Neumann side the interface dofs are
       * free, the discrete equations hold there, so r is ~0 and this expression
       * would silently export ZERO flux with no error raised.  That side keeps
       * the gradient recovery; its flux export is not what the partner consumes
       * anyway, but it IS what the conservation check balances, so it is
       * theta-averaged too, on the same step grid. */
      if (side == 0)
        {
          Vector<double> residual(n_dofs);
          A_free.vmult(residual, solution);
          residual -= rhs_free;
          std::vector<double> q(iface_dofs.size());
          for (std::size_t i = 0; i < iface_dofs.size(); ++i)
            q[i] = -residual(iface_dofs[i]) / weight(iface_dofs[i]);
          for (std::size_t i = 0; i < iface_dofs.size(); ++i)
            if (suspect[i] && fixup[i] < iface_dofs.size())
              q[i] = q[fixup[i]];
          for (std::size_t i = 0; i < iface_dofs.size(); ++i)
            Q_out[i][n] = q[i];
        }
      else
        {
          recover_gradient_flux(solution, q_new);
          for (std::size_t i = 0; i < iface_dofs.size(); ++i)
            Q_out[i][n] = theta * q_new[i] + (1.0 - theta) * q_prev[i];
          q_prev = q_new;
        }

      for (std::size_t i = 0; i < iface_dofs.size(); ++i)
        T_out[i][n] = solution(iface_dofs[i]);

      u_old = solution;
      F_old = F_new;
    }

  /* FIELD DUMP. The `couple` contract carries interface data only, so a
   * transient run that returns nothing but the interface is useless. One line
   * per dof: x y T(t_end) w, with w = \int phi_i dx the nodal volume weight, so
   * a mass-lumped L2 norm of any nodal field is sqrt(sum(w*f^2)) with no mesh
   * work outside this program. (participant_fenics_transient.py writes the
   * same four arrays as field_final.npz.) */
  {
    Vector<double> wvol(n_dofs);
    Vector<double> cell_w(dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_w = 0.;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_w(i) += fe_values.shape_value(i, q) * fe_values.JxW(q);
        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          wvol(local_dofs[i]) += cell_w(i);
      }
    std::ofstream fld("field_final.txt");
    fld.precision(16);
    for (types::global_dof_index i = 0; i < n_dofs; ++i)
      fld << support_points[i][0] << " " << support_points[i][1] << " "
          << solution(i) << " " << wvol(i) << "\n";
  }

  std::ofstream out(argv[2]);
  out.precision(16);
  out << iface_dofs.size() << " " << n_steps << "\n";
  for (std::size_t i = 0; i < iface_dofs.size(); ++i)
    {
      out << support_points[iface_dofs[i]][1];
      for (unsigned int n = 0; n < n_steps; ++n)
        out << " " << T_out[i][n];
      for (unsigned int n = 0; n < n_steps; ++n)
        out << " " << Q_out[i][n];
      out << "\n";
    }
  return 0;
}
