/* Steady heat conduction  -div(k grad T) = f  on ONE rectangular subdomain of
 * the canonical openPASO coupling problem (two subdomains sharing a straight
 * interface at x = iface_x).
 *
 * Domain [x0,x1] x [y0,y1].  Dirichlet T = T_outer on the NON-interface
 * x-boundary, natural (zero flux) on y = y0 and y = y1.
 * On the interface x = iface_x this participant is either
 *   side = 0 (DIRICHLET): T = T_if(y)   (piecewise linear from the samples), or
 *   side = 1 (NEUMANN)  : + \int g(y) v ds  added to the weak-form RHS
 *                         (g = the partner's exported normal fluxes, UNCHANGED).
 *
 * Input file (argv[1], whitespace separated):
 *   side k x0 x1 y0 y1 iface_x T_outer f_src nx ny degree
 *   n_samples
 *   y_0 v_0
 *   ...
 *   nfx nfy                        <- OPTIONAL, see the volume-source block
 *   f_00 f_10 ... f_(nfx-1)0       <- nfx*nfy values, x index fastest
 *   ...
 * Output file (argv[2]): one line "y T q" per interface node, sorted by y,
 *   q = -k * s * dT/dx at (iface_x, y)  with  s = +1 if iface_x > outer_x
 *   else -1, i.e. the outward normal flux density of THIS subdomain.
 *
 * All problem numbers come from the input file — nothing is hardcoded.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
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

/* Piecewise-linear interpolant of samples v(y), clamped outside the sample
 * range — identical semantics to numpy.interp. */
class Sampled1D
{
public:
  Sampled1D(std::vector<double> ys, std::vector<double> vs)
    : ys_(std::move(ys)), vs_(std::move(vs))
  {}

  double operator()(const double y) const
  {
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
      std::cerr << "usage: heat_iface_dealii <input.txt> <output.txt>\n";
      return 1;
    }

  std::ifstream in(argv[1]);
  if (!in)
    {
      std::cerr << "cannot open input file " << argv[1] << "\n";
      return 1;
    }

  unsigned int side;
  double       k, x0, x1, y0, y1, iface_x, T_outer, f_src;
  unsigned int nx, ny, degree, n_samples;
  in >> side >> k >> x0 >> x1 >> y0 >> y1 >> iface_x >> T_outer >> f_src >> nx >>
    ny >> degree >> n_samples;
  std::vector<double> sy(n_samples), sv(n_samples);
  for (unsigned int i = 0; i < n_samples; ++i)
    in >> sy[i] >> sv[i];
  if (!in || n_samples == 0)
    {
      std::cerr << "malformed input file\n";
      return 1;
    }
  const Sampled1D samples(sy, sv);

  /* OPTIONAL SPATIALLY VARYING VOLUME SOURCE, appended at the END of the input
   * file so an input written by a wrapper that knows nothing about it still
   * parses:
   *     nfx nfy
   *     f          (nfx*nfy values, x index fastest)
   * sampled on a UNIFORM tensor grid spanning [x0,x1] x [y0,y1] -- the wrapper
   * puts the samples on the FE node grid, so bilinear interpolation of them IS
   * the Q1 interpolant of the source for degree 1.  Absent or malformed means
   * ZERO.
   *
   * WHY THIS EXISTS AT ALL, when the header already carries f_src.  f_src is a
   * single number, and a source that varies with position -- which is what
   * every manufactured solution produces -- is not a number.  The header field
   * is kept so an input file written by an older wrapper still parses, and it
   * is simply ADDED to the sampled field; current wrappers write 0.0 there and
   * put the whole source in this block.  This is the scalar counterpart of the
   * body-force block in elast_iface_dealii.cc. */
  unsigned int     nfx = 0, nfy = 0;
  Table<2, double> ftab;
  bool             have_src = false;
  if ((in >> nfx >> nfy) && nfx >= 2 && nfy >= 2)
    {
      ftab.reinit(nfx, nfy);
      for (unsigned int j = 0; j < nfy; ++j)
        for (unsigned int i = 0; i < nfx; ++i)
          in >> ftab(i, j);
      have_src = static_cast<bool>(in);
      if (!have_src)
        {
          std::cerr << "malformed volume-source block\n";
          return 1;
        }
    }
  /* Announce it unconditionally. A binary built BEFORE this block existed stops
   * reading at the samples and ignores the source in silence, which returns the
   * boundary-data-only solution -- the exact bug this block was added to fix,
   * now hidden behind a participant that looks correct. The wrapper checks for
   * this line and refuses to believe a source-free result without it. */
  std::cout << "VOLUME_SOURCE " << (have_src ? "on " : "off ") << nfx << " "
            << nfy << std::endl;

  /* Bilinear interpolation of the sampled source.
   * InterpolatedUniformGridData is deal.II's own device for exactly this and
   * needs no extra dependency. */
  std::unique_ptr<Functions::InterpolatedUniformGridData<2>> src_fun;
  if (have_src)
    {
      const std::array<std::pair<double, double>, 2> ends{{{x0, x1}, {y0, y1}}};
      const std::array<unsigned int, 2>              nsub{{nfx - 1, nfy - 1}};
      src_fun = std::make_unique<Functions::InterpolatedUniformGridData<2>>(
        ends, nsub, ftab);
    }

  /* Which colorized boundary id is the interface?  colorize=true gives
   * 0: x = x0, 1: x = x1, 2: y = y0, 3: y = y1. */
  const bool         iface_at_x1 = std::abs(iface_x - x1) < std::abs(iface_x - x0);
  const unsigned int iface_id    = iface_at_x1 ? 1 : 0;
  const unsigned int outer_id    = iface_at_x1 ? 0 : 1;
  const double       s_out       = iface_at_x1 ? 1.0 : -1.0; // outward normal = s * e_x

  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {nx, ny}, Point<2>(x0, y0), Point<2>(x1, y1), /*colorize=*/true);

  const FE_Q<2> fe(degree);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    dof_handler, outer_id, Functions::ConstantFunction<2>(T_outer), constraints);
  const InterfaceFunction iface_fun(samples);
  if (side == 0)
    VectorTools::interpolate_boundary_values(dof_handler, iface_id, iface_fun,
                                             constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);

  SparseMatrix<double> system_matrix(sparsity);
  Vector<double>       solution(dof_handler.n_dofs());
  Vector<double>       rhs(dof_handler.n_dofs());

  /* A SECOND, UNCONSTRAINED copy of the same system.  The consistent interface
   * flux below is the residual of the discrete equations on the CONSTRAINED
   * rows, and constraints.distribute_local_to_global() destroys exactly those
   * rows (it zeroes them and puts the constraint on the diagonal).  So the
   * reaction has to be read off a matrix assembled with NO constraints at all;
   * its sparsity pattern must keep the constrained couplings too. */
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
  const QGauss<1>  face_quadrature(degree + 2);
  FEFaceValues<2>  fe_face_rhs(fe, face_quadrature,
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
        {
          /* VOLUME SOURCE: + \int f v dx, the header constant plus the sampled
           * field.  It goes into cell_rhs, which feeds BOTH the constrained
           * system AND free_rhs, so the consistent reaction flux below is
           * computed against the FULL right-hand side -- drop it there and the
           * exported flux is wrong by the load the cell carries. */
          const double f_q =
            f_src + (have_src ? src_fun->value(fe_values.quadrature_point(q))
                              : 0.0);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += k * fe_values.shape_grad(i, q) *
                                     fe_values.shape_grad(j, q) *
                                     fe_values.JxW(q);
              cell_rhs(i) +=
                f_q * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      /* Neumann side: + \int g(y) v ds on the interface faces. */
      if (side == 1)
        for (const unsigned int f : cell->face_indices())
          if (cell->face(f)->at_boundary() &&
              cell->face(f)->boundary_id() == iface_id)
            {
              fe_face_rhs.reinit(cell, f);
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  const double g =
                    samples(fe_face_rhs.quadrature_point(q)[1]);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += g * fe_face_rhs.shape_value(i, q) *
                                   fe_face_rhs.JxW(q);
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

  SolverControl            control(20000, 1e-14 * rhs.l2_norm() + 1e-16);
  SolverCG<Vector<double>> solver(control);
  PreconditionSSOR<SparseMatrix<double>> precond;
  precond.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, rhs, precond);
  constraints.distribute(solution);

  /* Interface temperature and outward normal heat flux density
   * q = -(k grad T).n, one value per interface node, sorted by y.
   *
   * WHY NOT THE GRADIENT OF THE SOLUTION. That is what this file used to do:
   * evaluate grad T at the FE support points of the interface faces and average
   * over the adjacent cells. The gradient of a P1/Q1 solution is only O(h)
   * accurate ON the boundary — the superconvergence points are interior — and
   * the boundary trace is exactly what the coupling reads. Measured against a
   * manufactured solution with a known exact interface flux, that recovery
   * converges at order ~1 while the consistent flux below converges at ~2, so
   * the recovery, not the physics and not the partner, was setting the answer.
   *
   * THE CONSISTENT (REACTION) FLUX. From
   *     a(u,v) - (f,v) = \int_{dOmega} (k grad u . n) v ds = -\int_Gamma qn v ds
   * it follows that for every basis function phi_i on the interface
   *     \int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
   * with r the UNCONSTRAINED residual (free_matrix / free_rhs above: no
   * constraint applied, constrained rows NOT zeroed, because on the Dirichlet
   * side those rows ARE the reaction).  Dividing by w_i = \int_Gamma phi_i ds
   * turns the functional into a density the partner can interpolate pointwise.
   * The outward normal is already in the identity, so no s_out factor appears.
   *
   * ONLY ON THE DIRICHLET SIDE.  On the Neumann side the interface dofs are
   * free, the discrete equations hold there, so r is ~0 and this expression
   * would silently export ZERO flux with no error raised.  That side keeps the
   * gradient recovery; its flux export is not what the partner consumes anyway.
   */
  struct IfaceNode
  {
    double y, T, q;
    bool   suspect;
  };
  std::vector<IfaceNode> nodes;

  if (side == 0)
    {
      Vector<double> residual(dof_handler.n_dofs());
      free_matrix.vmult(residual, solution);
      residual -= free_rhs; // r = A u_h - b

      Vector<double>  weight(dof_handler.n_dofs()); // w_i = \int_Gamma phi_i ds
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

      const std::vector<Point<2>> support_points =
        [&] {
          std::vector<Point<2>> p(dof_handler.n_dofs());
          DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler, p);
          return p;
        }();
      /* An interface node that ALSO lies on the outer Dirichlet boundary
       * carries the OUTER reaction as well, so its residual is not this
       * interface's flux. */
      const IndexSet outer_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(),
                                        {static_cast<types::boundary_id>(
                                          outer_id)});

      for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
        if (std::abs(weight(i)) > 1e-14)
          nodes.push_back({support_points[i][1], solution(i),
                           -residual(i) / weight(i),
                           outer_dofs.is_element(i)});

      std::sort(nodes.begin(), nodes.end(),
                [](const IfaceNode &a, const IfaceNode &b) { return a.y < b.y; });

      /* Replace a suspect node's flux by the nearest interior interface value
       * rather than exporting a corner value that is physically a different
       * quantity.  Non-suspect entries are never written, so in-place is safe. */
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
              nodes[i].q = nodes[best].q;
          }
    }
  else
    {
      const Quadrature<1> face_support(fe.get_unit_face_support_points());
      FEFaceValues<2>     fe_face(fe, face_support,
                                  update_values | update_gradients |
                                    update_quadrature_points);
      std::vector<double>       face_T(face_support.size());
      std::vector<Tensor<1, 2>> face_grad(face_support.size());
      std::map<double, std::array<double, 3>> iface; // y -> (sum T, sum q, count)

      for (const auto &cell : dof_handler.active_cell_iterators())
        for (const unsigned int f : cell->face_indices())
          if (cell->face(f)->at_boundary() &&
              cell->face(f)->boundary_id() == iface_id)
            {
              fe_face.reinit(cell, f);
              fe_face.get_function_values(solution, face_T);
              fe_face.get_function_gradients(solution, face_grad);
              for (unsigned int q = 0; q < face_support.size(); ++q)
                {
                  const double y   = fe_face.quadrature_point(q)[1];
                  const double key = std::round(y * 1e10) / 1e10;
                  auto        &e   = iface[key];
                  e[0] += face_T[q];
                  e[1] += -k * s_out * face_grad[q][0];
                  e[2] += 1.0;
                }
            }
      for (const auto &[y, e] : iface)
        nodes.push_back({y, e[0] / e[2], e[1] / e[2], false});
    }

  std::ofstream out(argv[2]);
  out.precision(16);
  for (const auto &n : nodes)
    out << n.y << " " << n.T << " " << n.q << "\n";
  return 0;
}
