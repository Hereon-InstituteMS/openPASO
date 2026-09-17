/* deal.II as a preCICE PARTICIPANT — a compiled C++ executable that links
 * libprecice directly and drives its own coupling loop.
 *
 * deal.II has no Python API, so this is the only way it can be a preCICE
 * participant at all: everything the Python participants get from `import
 * precice` this file gets from <precice/precice.hpp> and a link line against
 * libprecice.  The C++ API mirrors the Python one one-for-one —
 * precice::Participant, setMeshVertices, readData, writeData, advance,
 * requiresWritingCheckpoint / requiresReadingCheckpoint.
 *
 * PHYSICS.  Steady heat conduction  -div(k grad T) = f  on ONE rectangular
 * subdomain of the canonical two-subdomain problem (interface at x = iface_x).
 * Dirichlet T = T_outer on the NON-interface x-boundary, natural (zero flux)
 * on y = y0 and y = y1.  On the interface this participant is either
 *   side = 0 (DIRICHLET): reads a temperature, imposes it, writes back the
 *                         outward normal flux density it costs;
 *   side = 1 (NEUMANN)  : reads a flux density, adds \int g v ds to the RHS,
 *                         writes back the interface temperature.
 *
 * Input file (argv[1]), whitespace separated:
 *   side k x0 x1 y0 y1 iface_x T_outer f_src nx ny degree
 *   participant_name mesh_name write_data_name read_data_name config_path
 * Output file (argv[2]): JSON with the same shape the Python participants
 *   write — participant, sub_iterations, coordinates, values, normal_fluxes.
 *
 * All problem numbers come from the input file — nothing is hardcoded.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
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

#include <precice/precice.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dealii;

/* Piecewise-linear interpolant of samples v(y), clamped outside the sample
 * range — identical semantics to numpy.interp.  preCICE hands each participant
 * values AT ITS OWN VERTICES, so the samples are this subdomain's interface
 * support points and the interpolation only fills in between them for the face
 * quadrature. */
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


class HeatSubdomain
{
public:
  HeatSubdomain(unsigned int side,
                double       k,
                double       x0,
                double       x1,
                double       y0,
                double       y1,
                double       iface_x,
                double       T_outer,
                double       f_src,
                unsigned int nx,
                unsigned int ny,
                unsigned int degree)
    : side_(side)
    , k_(k)
    , iface_x_(iface_x)
    , T_outer_(T_outer)
    , f_src_(f_src)
    , fe_(degree)
    , dof_handler_(tria_)
    , quadrature_(degree + 1)
    , face_quadrature_(degree + 2)
  {
    /* colorize=true gives 0: x = x0, 1: x = x1, 2: y = y0, 3: y = y1. */
    const bool iface_at_x1 = std::abs(iface_x - x1) < std::abs(iface_x - x0);
    iface_id_              = iface_at_x1 ? 1 : 0;
    outer_id_              = iface_at_x1 ? 0 : 1;
    s_out_                 = iface_at_x1 ? 1.0 : -1.0; // outward normal = s * e_x

    GridGenerator::subdivided_hyper_rectangle(
      tria_, {nx, ny}, Point<2>(x0, y0), Point<2>(x1, y1), /*colorize=*/true);
    dof_handler_.distribute_dofs(fe_);

    /* The interface vertices, in a FIXED order (sorted by y).  preCICE indexes
     * by the ids setMeshVertices returned, so this order is chosen once and
     * never revisited. */
    const Quadrature<1> face_support(fe_.get_unit_face_support_points());
    FEFaceValues<2>     fe_face(fe_, face_support, update_quadrature_points);
    std::map<double, int> seen;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      for (const unsigned int f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == iface_id_)
          {
            fe_face.reinit(cell, f);
            for (unsigned int q = 0; q < face_support.size(); ++q)
              seen[key(fe_face.quadrature_point(q)[1])] = 1;
          }
    for (const auto &[y, _] : seen)
      iface_y_.push_back(y);
  }

  const std::vector<double> &interface_y() const
  {
    return iface_y_;
  }
  double interface_x() const
  {
    return iface_x_;
  }

  /* One solve with `incoming` imposed at the interface (a temperature when
   * side = 0, a flux density when side = 1).  Fills T and q at the interface
   * vertices, q being THIS subdomain's OUTWARD normal flux density. */
  void solve(const std::vector<double> &incoming,
             std::vector<double>       &T,
             std::vector<double>       &q)
  {
    const Sampled1D samples(iface_y_, incoming);

    AffineConstraints<double> constraints;
    VectorTools::interpolate_boundary_values(
      dof_handler_, outer_id_, Functions::ConstantFunction<2>(T_outer_),
      constraints);
    const InterfaceFunction iface_fun(samples);
    if (side_ == 0)
      VectorTools::interpolate_boundary_values(dof_handler_, iface_id_,
                                               iface_fun, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_, dsp, constraints, false);
    SparsityPattern sparsity;
    sparsity.copy_from(dsp);

    SparseMatrix<double> system_matrix(sparsity);
    Vector<double>       solution(dof_handler_.n_dofs());
    Vector<double>       rhs(dof_handler_.n_dofs());

    /* A SECOND, UNCONSTRAINED copy of the same system.  The consistent
     * interface flux below is the residual of the discrete equations on the
     * CONSTRAINED rows, and constraints.distribute_local_to_global() destroys
     * exactly those rows.  So the reaction has to be read off a matrix
     * assembled with NO constraints at all. */
    DynamicSparsityPattern dsp_free(dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_, dsp_free);
    SparsityPattern sparsity_free;
    sparsity_free.copy_from(dsp_free);
    SparseMatrix<double> free_matrix(sparsity_free);
    Vector<double>       free_rhs(dof_handler_.n_dofs());

    FEValues<2>     fe_values(fe_, quadrature_,
                              update_values | update_gradients |
                                update_JxW_values);
    FEFaceValues<2> fe_face_rhs(fe_, face_quadrature_,
                                update_values | update_quadrature_points |
                                  update_JxW_values);

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrix = 0.;
        cell_rhs    = 0.;
        for (unsigned int qp = 0; qp < quadrature_.size(); ++qp)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += k_ * fe_values.shape_grad(i, qp) *
                                     fe_values.shape_grad(j, qp) *
                                     fe_values.JxW(qp);
              cell_rhs(i) +=
                f_src_ * fe_values.shape_value(i, qp) * fe_values.JxW(qp);
            }

        if (side_ == 1)
          for (const unsigned int f : cell->face_indices())
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == iface_id_)
              {
                fe_face_rhs.reinit(cell, f);
                for (unsigned int qp = 0; qp < face_quadrature_.size(); ++qp)
                  {
                    const double g =
                      samples(fe_face_rhs.quadrature_point(qp)[1]);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) += g * fe_face_rhs.shape_value(i, qp) *
                                     fe_face_rhs.JxW(qp);
                  }
              }

        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              free_matrix.add(local_dofs[i], local_dofs[j], cell_matrix(i, j));
            free_rhs(local_dofs[i]) += cell_rhs(i);
          }
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                               local_dofs, system_matrix, rhs);
      }

    SolverControl            control(20000, 1e-14 * rhs.l2_norm() + 1e-16);
    SolverCG<Vector<double>> solver(control);
    PreconditionSSOR<SparseMatrix<double>> precond;
    precond.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, rhs, precond);
    constraints.distribute(solution);

    /* Interface temperature and outward normal flux density q = -(k grad T).n.
     *
     * WHY NOT THE GRADIENT OF THE SOLUTION. That is what this file used to do:
     * evaluate grad T at the FE support points of the interface faces and
     * average over the adjacent cells. The gradient of a P1/Q1 solution is only
     * O(h) accurate ON the boundary — the superconvergence points are interior
     * — and the boundary trace is exactly what preCICE ships to the partner.
     * Measured against a manufactured solution with a known exact interface
     * flux, that recovery converges at order ~1 while the consistent flux below
     * converges at ~2, so the recovery, not the physics and not the partner,
     * was setting the answer.
     *
     * THE CONSISTENT (REACTION) FLUX. From
     *   a(u,v) - (f,v) = \int_{dOmega} (k grad u . n) v ds = -\int_Gamma qn v ds
     * it follows that for every basis function phi_i on the interface
     *   \int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
     * with r the UNCONSTRAINED residual (free_matrix / free_rhs above: no
     * constraint applied, constrained rows NOT zeroed, because on the Dirichlet
     * side those rows ARE the reaction).  Dividing by w_i = \int_Gamma phi_i ds
     * turns the functional into a density the partner can interpolate.  The
     * outward normal is already in the identity, so no s_out_ factor appears.
     *
     * ONLY WHEN side_ == 0.  On the Neumann side the interface dofs are free,
     * the discrete equations hold there, so r is ~0 and this expression would
     * silently ship ZERO flux with no error raised.  That side keeps the
     * gradient recovery; its flux is not what the partner consumes anyway.
     */
    const Quadrature<1> face_support(fe_.get_unit_face_support_points());
    FEFaceValues<2>     fe_face(fe_, face_support,
                                update_values | update_gradients |
                                  update_quadrature_points);
    std::vector<double>       face_T(face_support.size());
    std::vector<Tensor<1, 2>> face_grad(face_support.size());
    std::map<double, std::array<double, 3>> acc;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      for (const unsigned int f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == iface_id_)
          {
            fe_face.reinit(cell, f);
            fe_face.get_function_values(solution, face_T);
            fe_face.get_function_gradients(solution, face_grad);
            for (unsigned int qp = 0; qp < face_support.size(); ++qp)
              {
                auto &e = acc[key(fe_face.quadrature_point(qp)[1])];
                e[0] += face_T[qp];
                e[1] += -k_ * s_out_ * face_grad[qp][0];
                e[2] += 1.0;
              }
          }

    std::map<double, double> q_cons;
    if (side_ == 0)
      {
        Vector<double> residual(dof_handler_.n_dofs());
        free_matrix.vmult(residual, solution);
        residual -= free_rhs; // r = A u_h - b

        Vector<double>  weight(dof_handler_.n_dofs()); // \int_Gamma phi_i ds
        FEFaceValues<2> fe_face_w(fe_, face_quadrature_,
                                  update_values | update_JxW_values);
        for (const auto &cell : dof_handler_.active_cell_iterators())
          for (const unsigned int f : cell->face_indices())
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == iface_id_)
              {
                fe_face_w.reinit(cell, f);
                cell->get_dof_indices(local_dofs);
                for (unsigned int qp = 0; qp < face_quadrature_.size(); ++qp)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    weight(local_dofs[i]) +=
                      fe_face_w.shape_value(i, qp) * fe_face_w.JxW(qp);
              }

        std::vector<Point<2>> support_points(dof_handler_.n_dofs());
        DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler_,
                                             support_points);
        /* An interface node that ALSO lies on the outer Dirichlet boundary
         * carries the OUTER reaction as well, so its residual is not this
         * interface's flux. */
        const IndexSet outer_dofs = DoFTools::extract_boundary_dofs(
          dof_handler_, ComponentMask(),
          {static_cast<types::boundary_id>(outer_id_)});

        std::map<double, double> suspect;
        for (types::global_dof_index i = 0; i < dof_handler_.n_dofs(); ++i)
          if (std::abs(weight(i)) > 1e-14)
            {
              const double kk = key(support_points[i][1]);
              q_cons[kk]      = -residual(i) / weight(i);
              suspect[kk]     = outer_dofs.is_element(i) ? 1.0 : 0.0;
            }
        /* Replace a suspect node's flux by the nearest interior interface
         * value: iface_y_ is sorted, so nearest in index is nearest in y. */
        for (std::size_t i = 0; i < iface_y_.size(); ++i)
          if (suspect.at(iface_y_[i]) != 0.0)
            {
              std::size_t best = iface_y_.size();
              for (std::size_t j = 0; j < iface_y_.size(); ++j)
                if (suspect.at(iface_y_[j]) == 0.0 &&
                    (best == iface_y_.size() ||
                     std::abs(static_cast<long>(j) - static_cast<long>(i)) <
                       std::abs(static_cast<long>(best) - static_cast<long>(i))))
                  best = j;
              if (best < iface_y_.size())
                q_cons[iface_y_[i]] = q_cons[iface_y_[best]];
            }
      }

    T.assign(iface_y_.size(), 0.0);
    q.assign(iface_y_.size(), 0.0);
    for (std::size_t i = 0; i < iface_y_.size(); ++i)
      {
        const auto &e = acc.at(iface_y_[i]);
        T[i]          = e[0] / e[2];
        q[i]          = side_ == 0 ? q_cons.at(iface_y_[i]) : e[1] / e[2];
      }
  }

private:
  static double key(const double y)
  {
    return std::round(y * 1e10) / 1e10;
  }

  unsigned int        side_;
  double              k_, iface_x_, T_outer_, f_src_, s_out_ = 1.0;
  unsigned int        iface_id_ = 0, outer_id_ = 1;
  Triangulation<2>    tria_;
  FE_Q<2>             fe_;
  DoFHandler<2>       dof_handler_;
  QGauss<2>           quadrature_;
  QGauss<1>           face_quadrature_;
  std::vector<double> iface_y_;
};


int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cerr << "usage: precice_heat_dealii <input.txt> <output.json>\n";
      return 1;
    }

  std::ifstream in(argv[1]);
  if (!in)
    {
      std::cerr << "cannot open input file " << argv[1] << "\n";
      return 1;
    }

  unsigned int side, nx, ny, degree;
  double       k, x0, x1, y0, y1, iface_x, T_outer, f_src;
  std::string  pname, mesh_name, write_data, read_data, config_path;
  in >> side >> k >> x0 >> x1 >> y0 >> y1 >> iface_x >> T_outer >> f_src >>
    nx >> ny >> degree;
  in >> pname >> mesh_name >> write_data >> read_data >> config_path;
  if (!in)
    {
      std::cerr << "malformed input file\n";
      return 1;
    }

  /* The mesh, the DoFs and the interface vertex ORDER are built ONCE, before
   * the Participant is constructed: the partner blocks on the connection
   * handshake, so any setup done after this point is setup the partner waits
   * through. */
  HeatSubdomain problem(side, k, x0, x1, y0, y1, iface_x, T_outer, f_src, nx,
                        ny, degree);
  const std::vector<double> &ys = problem.interface_y();
  if (ys.empty())
    {
      std::cerr << "no interface support points at x=" << iface_x << "\n";
      return 1;
    }

  precice::Participant participant(pname, config_path, 0, 1);

  std::vector<double> coords;
  coords.reserve(2 * ys.size());
  for (const double y : ys)
    {
      coords.push_back(problem.interface_x());
      coords.push_back(y);
    }
  std::vector<precice::VertexID> ids(ys.size());
  participant.setMeshVertices(mesh_name, coords, ids);

  std::vector<double> incoming(ys.size(), 0.0);
  std::vector<double> T(ys.size(), 0.0), q(ys.size(), 0.0);
  std::vector<double> outgoing(ys.size(), 0.0);

  if (participant.requiresInitialData())
    participant.writeData(mesh_name, write_data, ids, outgoing);
  participant.initialize();

  unsigned int n_it = 0;
  while (participant.isCouplingOngoing())
    {
      if (participant.requiresWritingCheckpoint())
        {
          /* steady solve: no state to save.  The CALL is what preCICE
           * requires — an implicit scheme aborts without it. */
        }
      const double dt = participant.getMaxTimeStepSize();
      participant.readData(mesh_name, read_data, ids, dt, incoming);
      problem.solve(incoming, T, q);
      outgoing = (side == 0) ? q : T;
      participant.writeData(mesh_name, write_data, ids, outgoing);
      participant.advance(dt);
      ++n_it;
      if (participant.requiresReadingCheckpoint())
        {
          /* ... and none to restore. */
        }
    }
  participant.finalize();

  std::ofstream out(argv[2]);
  out.precision(16);
  out << "{\n  \"participant\": \"" << pname << "\",\n";
  out << "  \"sub_iterations\": " << n_it << ",\n";
  out << "  \"coordinates\": [";
  for (std::size_t i = 0; i < ys.size(); ++i)
    out << (i ? ", " : "") << "[" << problem.interface_x() << ", " << ys[i]
        << "]";
  out << "],\n  \"values\": [";
  for (std::size_t i = 0; i < T.size(); ++i)
    out << (i ? ", " : "") << T[i];
  out << "],\n  \"normal_fluxes\": [";
  for (std::size_t i = 0; i < q.size(); ++i)
    out << (i ? ", " : "") << q[i];
  out << "]\n}\n";
  return 0;
}
