/* Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
 * subdomain of the openPASO VECTOR coupling problem (two subdomains sharing a
 * straight interface at x = iface_x).  The vector counterpart of
 * heat_iface_dealii.cc.
 *
 * Domain [x0,x1] x [y0,y1].  The WHOLE non-interface boundary (the outer
 * x-face and both y-faces) carries a prescribed displacement given as a
 * polynomial in (x, y):
 *     u_x = cx[0] + cx[1]*x + cx[2]*y + cx[3]*y*y
 *     u_y = cy[0] + cy[1]*x + cy[2]*y + cy[3]*y*y
 * On the interface x = iface_x this participant is either
 *   side = 0 (DIRICHLET): u = u_if(y)  (piecewise linear from the samples), or
 *   side = 1 (NEUMANN)  : + \int g(y) . v ds  added to the weak-form RHS
 *                         (g = the partner's exported traction, UNCHANGED).
 *
 * THE TWO INTERFACE CORNERS (iface_x, y0) and (iface_x, y1) belong to the
 * OUTER boundary on BOTH sides: they sit on a y-face, which is Dirichlet in
 * the un-split problem. The outer constraint therefore WINS there — see the
 * std::map merge below, which relies on std::map::insert not overwriting.
 * Handing them to the interface leaves them unconstrained on the Neumann side;
 * that subproblem is still well posed, still converges, and lands a few
 * percent off with a residual of 1e-10 and a balanced interface.
 *
 * SIGN CONVENTION for the exported traction, identical to the shipped Python
 * vector participants and to the scalar (heat) ones:
 *     q_out = -(sigma . n_own),   n_own = s_out * e_x
 * so the two sides' exports cancel and the Neumann side applies the partner's
 * numbers unchanged.
 *
 * Input file (argv[1], whitespace separated):
 *   side E nu x0 x1 y0 y1 iface_x nx ny degree
 *   cx0 cx1 cx2 cx3
 *   cy0 cy1 cy2 cy3
 *   n_samples
 *   y_0 vx_0 vy_0
 *   ...
 * Output file (argv[2]): one line "y ux uy qx qy" per interface node,
 *   sorted by y.
 *
 * All problem numbers come from the input file — nothing is hardcoded.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/table.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

using namespace dealii;

/* Piecewise-linear interpolant of vector samples v(y), clamped outside the
 * sample range — identical semantics to numpy.interp applied per component. */
class Sampled2D
{
public:
  Sampled2D(std::vector<double> ys, std::vector<double> vx,
            std::vector<double> vy)
    : ys_(std::move(ys)), vx_(std::move(vx)), vy_(std::move(vy))
  {}

  double operator()(const double y, const unsigned int c) const
  {
    const std::vector<double> &v = (c == 0 ? vx_ : vy_);
    if (ys_.size() == 1 || y <= ys_.front())
      return v.front();
    if (y >= ys_.back())
      return v.back();
    const auto        it = std::upper_bound(ys_.begin(), ys_.end(), y);
    const std::size_t i  = std::distance(ys_.begin(), it);
    const double      w  = (y - ys_[i - 1]) / (ys_[i] - ys_[i - 1]);
    return (1.0 - w) * v[i - 1] + w * v[i];
  }

private:
  std::vector<double> ys_, vx_, vy_;
};

/* The prescribed displacement on the non-interface boundary. */
class PolyBC : public Function<2>
{
public:
  PolyBC(const std::array<double, 4> &cx, const std::array<double, 4> &cy)
    : Function<2>(2)
    , cx_(cx)
    , cy_(cy)
  {}
  virtual double value(const Point<2>  &p,
                       const unsigned int component = 0) const override
  {
    const std::array<double, 4> &c = (component == 0 ? cx_ : cy_);
    return c[0] + c[1] * p[0] + c[2] * p[1] + c[3] * p[1] * p[1];
  }

private:
  const std::array<double, 4> cx_, cy_;
};

/* Sampled2D as a 2-component deal.II Function of y. */
class InterfaceFunction : public Function<2>
{
public:
  explicit InterfaceFunction(const Sampled2D &s)
    : Function<2>(2)
    , s_(s)
  {}
  virtual double value(const Point<2>  &p,
                       const unsigned int component = 0) const override
  {
    return s_(p[1], component);
  }

private:
  const Sampled2D &s_;
};

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cerr << "usage: elast_iface_dealii <input.txt> <output.txt>\n";
      return 1;
    }

  std::ifstream in(argv[1]);
  if (!in)
    {
      std::cerr << "cannot open input file " << argv[1] << "\n";
      return 1;
    }

  unsigned int side;
  double       E, nu, x0, x1, y0, y1, iface_x;
  unsigned int nx, ny, degree, n_samples;
  in >> side >> E >> nu >> x0 >> x1 >> y0 >> y1 >> iface_x >> nx >> ny >>
    degree;
  std::array<double, 4> cx{}, cy{};
  for (unsigned int i = 0; i < 4; ++i)
    in >> cx[i];
  for (unsigned int i = 0; i < 4; ++i)
    in >> cy[i];
  in >> n_samples;
  std::vector<double> sy(n_samples), svx(n_samples), svy(n_samples);
  for (unsigned int i = 0; i < n_samples; ++i)
    in >> sy[i] >> svx[i] >> svy[i];
  if (!in || n_samples == 0)
    {
      std::cerr << "malformed input file\n";
      return 1;
    }
  const Sampled2D samples(sy, svx, svy);

  /* OPTIONAL BODY FORCE, appended at the END of the input file so an input
   * written by a wrapper that knows nothing about it still parses:
   *     nbx nby
   *     bx by      (nbx*nby pairs, x index fastest)
   * sampled on a UNIFORM tensor grid spanning [x0,x1] x [y0,y1] -- the wrapper
   * puts the samples on the FE node grid, so bilinear interpolation of them IS
   * the Q1 interpolant of the source for degree 1.  Absent or malformed means
   * ZERO, i.e. the -div(sigma) = 0 this file used to hardcode. */
  unsigned int     nbx = 0, nby = 0;
  Table<2, double> btab_x, btab_y;
  bool             have_body = false;
  if ((in >> nbx >> nby) && nbx >= 2 && nby >= 2)
    {
      btab_x.reinit(nbx, nby);
      btab_y.reinit(nbx, nby);
      for (unsigned int j = 0; j < nby; ++j)
        for (unsigned int i = 0; i < nbx; ++i)
          in >> btab_x(i, j) >> btab_y(i, j);
      have_body = static_cast<bool>(in);
      if (!have_body)
        {
          std::cerr << "malformed body-force block\n";
          return 1;
        }
    }
  /* Announce it unconditionally. A binary built BEFORE this block existed
   * stops reading at the samples and ignores a body force in silence, which
   * returns u = 0 -- the exact bug this block was added to fix, now hidden
   * behind a participant that looks correct. The wrapper checks for this line
   * and refuses to believe a zero result without it. */
  std::cout << "BODY_FORCE " << (have_body ? "on " : "off ") << nbx << " "
            << nby << std::endl;

  /* Plane strain. */
  const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const double mu     = E / (2.0 * (1.0 + nu));

  /* Bilinear interpolation of the sampled body force, one scalar field per
   * component.  InterpolatedUniformGridData is deal.II's own device for exactly
   * this and needs no extra dependency. */
  std::unique_ptr<Functions::InterpolatedUniformGridData<2>> body_x, body_y;
  if (have_body)
    {
      const std::array<std::pair<double, double>, 2> ends{
        {{x0, x1}, {y0, y1}}};
      const std::array<unsigned int, 2> nsub{{nbx - 1, nby - 1}};
      body_x = std::make_unique<Functions::InterpolatedUniformGridData<2>>(
        ends, nsub, btab_x);
      body_y = std::make_unique<Functions::InterpolatedUniformGridData<2>>(
        ends, nsub, btab_y);
    }

  /* colorize=true gives 0: x = x0, 1: x = x1, 2: y = y0, 3: y = y1. */
  const bool         iface_at_x1 = std::abs(iface_x - x1) < std::abs(iface_x - x0);
  const unsigned int iface_id    = iface_at_x1 ? 1 : 0;
  const unsigned int outer_id    = iface_at_x1 ? 0 : 1;
  const double       s_out       = iface_at_x1 ? 1.0 : -1.0; // n_own = s * e_x

  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {nx, ny}, Point<2>(x0, y0), Point<2>(x1, y1), /*colorize=*/true);

  const FE_Q<2>     base(degree);
  const FESystem<2> fe(base, 2);
  DoFHandler<2>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  /* Constraints: the OUTER boundary first, then the interface, merged with
   * std::map::insert so that a dof already fixed by the outer boundary — the
   * two interface corners — keeps that value. */
  const PolyBC                             outer_fun(cx, cy);
  std::map<types::global_dof_index, double> fixed;
  for (const unsigned int id : {outer_id, 2u, 3u})
    VectorTools::interpolate_boundary_values(dof_handler, id, outer_fun, fixed);
  if (side == 0)
    {
      const InterfaceFunction                   iface_fun(samples);
      std::map<types::global_dof_index, double> iface_vals;
      VectorTools::interpolate_boundary_values(dof_handler, iface_id, iface_fun,
                                               iface_vals);
      for (const auto &kv : iface_vals)
        fixed.insert(kv); // does NOT overwrite: the outer boundary wins
    }

  AffineConstraints<double> constraints;
  for (const auto &kv : fixed)
    {
      constraints.add_line(kv.first);
      constraints.set_inhomogeneity(kv.first, kv.second);
    }
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);

  SparseMatrix<double> system_matrix(sparsity);
  Vector<double>       solution(dof_handler.n_dofs());
  Vector<double>       rhs(dof_handler.n_dofs());

  /* A SECOND, UNCONSTRAINED copy of the same system.  The consistent interface
   * traction below is the residual of the discrete equations on the CONSTRAINED
   * rows, and constraints.distribute_local_to_global() destroys exactly those
   * rows.  So the reaction has to be read off a matrix assembled with NO
   * constraints at all; its sparsity pattern must keep those couplings too. */
  DynamicSparsityPattern dsp_free(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp_free);
  SparsityPattern sparsity_free;
  sparsity_free.copy_from(dsp_free);
  SparseMatrix<double> free_matrix(sparsity_free);
  Vector<double>       free_rhs(dof_handler.n_dofs());

  const QGauss<2> quadrature(degree + 1);
  FEValues<2>     fe_values(fe, quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const QGauss<1> face_quadrature(degree + 2);
  FEFaceValues<2> fe_face_rhs(fe, face_quadrature,
                              update_values | update_quadrature_points |
                                update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.;
      cell_rhs    = 0.;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const SymmetricTensor<2, 2> eps_i =
              fe_values[FEValuesExtractors::Vector(0)].symmetric_gradient(i, q);
            const double div_i =
              fe_values[FEValuesExtractors::Vector(0)].divergence(i, q);
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const SymmetricTensor<2, 2> eps_j =
                  fe_values[FEValuesExtractors::Vector(0)].symmetric_gradient(j,
                                                                              q);
                const double div_j =
                  fe_values[FEValuesExtractors::Vector(0)].divergence(j, q);
                cell_matrix(i, j) +=
                  (2.0 * mu * (eps_i * eps_j) + lambda * div_i * div_j) *
                  fe_values.JxW(q);
              }
          }

      /* BODY FORCE: + \int b . v dx.  It goes into cell_rhs, which feeds BOTH
       * the constrained system AND free_rhs, so the consistent reaction
       * traction below is computed against the FULL right-hand side -- drop it
       * there and the exported traction is wrong by the load the cell carries.
       */
      if (have_body)
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const Point<2> &p = fe_values.quadrature_point(q);
            const double    b[2] = {body_x->value(p), body_y->value(p)};
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int c = fe.system_to_component_index(i).first;
                cell_rhs(i) +=
                  b[c] * fe_values.shape_value(i, q) * fe_values.JxW(q);
              }
          }

      /* Neumann side: + \int g(y) . v ds on the interface faces. */
      if (side == 1)
        for (const unsigned int f : cell->face_indices())
          if (cell->face(f)->at_boundary() &&
              cell->face(f)->boundary_id() == iface_id)
            {
              fe_face_rhs.reinit(cell, f);
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  const double y = fe_face_rhs.quadrature_point(q)[1];
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const unsigned int c =
                        fe.system_to_component_index(i).first;
                      cell_rhs(i) += samples(y, c) *
                                     fe_face_rhs.shape_value(i, q) *
                                     fe_face_rhs.JxW(q);
                    }
                }
            }

      cell->get_dof_indices(local_dofs);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            free_matrix.add(local_dofs[i], local_dofs[j], cell_matrix(i, j));
          free_rhs(local_dofs[i]) += cell_rhs(i);
        }
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dofs,
                                             system_matrix, rhs);
    }

  SolverControl            control(50000, 1e-14 * rhs.l2_norm() + 1e-18);
  SolverCG<Vector<double>> solver(control);
  PreconditionSSOR<SparseMatrix<double>> precond;
  precond.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, rhs, precond);
  constraints.distribute(solution);

  /* Interface displacement and outward traction export, one value per interface
   * node, sorted by y.
   *
   * WHY NOT THE STRESS OF THE SOLUTION. That is what this file used to do:
   * evaluate sigma(u_h) at the FE support points of the interface faces and
   * average over the adjacent cells. The gradient of a Q1 solution — and
   * therefore the stress — is only O(h) accurate ON the boundary; the
   * superconvergence points are interior, and the boundary trace is exactly
   * what the coupling reads. Measured against a manufactured solution with a
   * known exact interface traction, that recovery converges at order ~1 while
   * the consistent traction below converges at ~2, so the recovery, not the
   * physics and not the partner, was setting the answer.
   *
   * THE CONSISTENT (REACTION) TRACTION. From
   *     a(u,v) - (f,v) = \int_{dOmega} (sigma(u).n).v ds = -\int_Gamma q_out.v ds
   * (the second equality is this file's sign convention, q_out = -(sigma.n_own))
   * it follows that for every vector basis function phi_i on the interface
   *     \int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b
   * with r the UNCONSTRAINED residual (free_matrix / free_rhs above).  Dividing
   * by w_i = \int_Gamma phi_i ds turns the functional into a density the
   * partner can interpolate pointwise.  The outward normal is already in the
   * identity, so no s_out factor appears.
   *
   * ONLY ON THE DIRICHLET SIDE.  On the Neumann side the interface dofs are
   * free, the discrete equations hold there, so r is ~0 and this expression
   * would silently export ZERO traction with no error raised.  That side keeps
   * the stress recovery; its export is not what the partner consumes anyway.
   */
  struct IfaceNode
  {
    double y, u[2], q[2];
    bool   suspect;
  };
  std::vector<IfaceNode> nodes;

  if (side == 0)
    {
      Vector<double> residual(dof_handler.n_dofs());
      free_matrix.vmult(residual, solution);
      residual -= free_rhs; // r = A u_h - b

      /* w_i = \int_Gamma phi_i ds.  FESystem(FE_Q, 2) is primitive, so
       * shape_value(i, q) is the value of dof i's single non-zero component. */
      Vector<double>  weight(dof_handler.n_dofs());
      FEFaceValues<2> fe_face_w(fe, face_quadrature,
                                update_values | update_JxW_values);
      std::vector<unsigned int> dof_comp(dof_handler.n_dofs(), 0);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(local_dofs);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            dof_comp[local_dofs[i]] = fe.system_to_component_index(i).first;
          for (const unsigned int f : cell->face_indices())
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == iface_id)
              {
                fe_face_w.reinit(cell, f);
                for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    weight(local_dofs[i]) +=
                      fe_face_w.shape_value(i, q) * fe_face_w.JxW(q);
              }
        }

      std::vector<Point<2>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler,
                                           support_points);
      /* THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (a
       * y-face), so their rows carry the OUTER reaction too and their residual
       * is not this interface's traction. */
      const IndexSet outer_dofs = DoFTools::extract_boundary_dofs(
        dof_handler, ComponentMask(),
        {static_cast<types::boundary_id>(outer_id),
         static_cast<types::boundary_id>(2), static_cast<types::boundary_id>(3)});

      std::map<double, IfaceNode> by_y;
      for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
        if (std::abs(weight(i)) > 1e-14)
          {
            const double key = std::round(support_points[i][1] * 1e10) / 1e10;
            auto        &e   = by_y[key];
            e.y              = support_points[i][1];
            e.u[dof_comp[i]] = solution(i);
            e.q[dof_comp[i]] = -residual(i) / weight(i);
            e.suspect        = e.suspect || outer_dofs.is_element(i);
          }
      for (const auto &kv : by_y)
        nodes.push_back(kv.second); // std::map is already sorted by y

      /* Replace a suspect node's traction by the nearest interior interface
       * value.  Non-suspect entries are never written, so in-place is safe. */
      for (std::size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].suspect)
          {
            std::size_t best = nodes.size();
            for (std::size_t j = 0; j < nodes.size(); ++j)
              if (!nodes[j].suspect &&
                  (best == nodes.size() ||
                   std::abs(static_cast<long>(j) - static_cast<long>(i)) <
                     std::abs(static_cast<long>(best) - static_cast<long>(i))))
                best = j;
            if (best < nodes.size())
              {
                nodes[i].q[0] = nodes[best].q[0];
                nodes[i].q[1] = nodes[best].q[1];
              }
          }
    }
  else
    {
      const Quadrature<1> face_support(base.get_unit_face_support_points());
      FEFaceValues<2>     fe_face(fe, face_support,
                                  update_values | update_gradients |
                                    update_quadrature_points);
      std::vector<Vector<double>> face_u(face_support.size(), Vector<double>(2));
      std::vector<std::vector<Tensor<1, 2>>> face_grad(
        face_support.size(), std::vector<Tensor<1, 2>>(2));
      std::map<double, std::array<double, 5>> iface; // y -> (ux,uy,qx,qy,count)

      for (const auto &cell : dof_handler.active_cell_iterators())
        for (const unsigned int f : cell->face_indices())
          if (cell->face(f)->at_boundary() &&
              cell->face(f)->boundary_id() == iface_id)
            {
              fe_face.reinit(cell, f);
              fe_face.get_function_values(solution, face_u);
              fe_face.get_function_gradients(solution, face_grad);
              for (unsigned int q = 0; q < face_support.size(); ++q)
                {
                  const double y   = fe_face.quadrature_point(q)[1];
                  const double key = std::round(y * 1e10) / 1e10;
                  const double exx = face_grad[q][0][0];
                  const double eyy = face_grad[q][1][1];
                  const double exy =
                    0.5 * (face_grad[q][0][1] + face_grad[q][1][0]);
                  const double sxx = 2.0 * mu * exx + lambda * (exx + eyy);
                  const double sxy = 2.0 * mu * exy;
                  auto        &e   = iface[key];
                  e[0] += face_u[q][0];
                  e[1] += face_u[q][1];
                  e[2] += -s_out * sxx;
                  e[3] += -s_out * sxy;
                  e[4] += 1.0;
                }
            }
      for (const auto &[y, e] : iface)
        nodes.push_back({y,
                         {e[0] / e[4], e[1] / e[4]},
                         {e[2] / e[4], e[3] / e[4]},
                         false});
    }

  std::ofstream out(argv[2]);
  out.precision(16);
  for (const auto &n : nodes)
    out << n.y << " " << n.u[0] << " " << n.u[1] << " " << n.q[0] << " "
        << n.q[1] << "\n";
  return 0;
}
