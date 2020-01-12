#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>



const unsigned int dim       = 2;
const unsigned int fe_degree = 1;
using Number                 = double;

using namespace dealii;

int
main()
{
  Triangulation<dim> tria;

  if (false)
    {
      GridGenerator::hyper_cube(tria);
      tria.refine_global(2);
    }
  else
    {
      GridGenerator::hyper_cube(tria, -1.0, 1.0);
      tria.refine_global();
      for (auto &cell : tria.active_cell_iterators())
        if (cell->active() && cell->center()[0] < 0.0)
          cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }

  const FE_Q<dim> fe(fe_degree);

  // setup dof-handlers
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraint;
  DoFTools::make_hanging_node_constraints(dof_handler, constraint);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraint);


  QGauss<dim>        quadrature_formula(fe.degree + 1);
  FEValues<dim>      fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<Number> diagonal_global;
  diagonal_global.reinit(dof_handler.n_dofs());

  // SparseMatrix<double> system_matrix;
  // DynamicSparsityPattern dsp(dof_handler.n_dofs());
  // DoFTools::make_sparsity_pattern(dof_handler, dsp);
  // sparsity_pattern.copy_from(dsp);
  // system_matrix.reinit(sparsity_pattern);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
              (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
               fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
               fe_values.JxW(q_index));           // dx

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      auto get_constraint_vector = [&](const unsigned int i) {
        std::vector<std::pair<unsigned int, Number>> v;

        if (constraint.is_constrained(i))
          return v;

        std::vector<Number> temp(dofs_per_cell);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          if (i == local_dof_indices[j])
            v.emplace_back(j, 1.0);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            if (!constraint.is_constrained(local_dof_indices[j]))
              continue;
            for (auto c :
                 *constraint.get_constraint_entries(local_dof_indices[j]))
              if (c.first == i)
                v.emplace_back(j, c.second);
          }

        std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
          return a.first < b.first;
        });

        return v;
      };

      auto compute_ith_column_of_matrix = [&](unsigned int col) {
        Vector<Number> v(dofs_per_cell);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          v[j] = cell_matrix[j][col];

        return v;
      };

      // setup CSR-storage
      std::vector<unsigned int> c_pool_row{0};
      std::vector<unsigned int> c_pool_col;
      std::vector<Number>       c_pool_val;

      std::vector<unsigned int> lid_to_gid;

      for (unsigned int j = 0; j < dof_handler.n_dofs(); ++j)
        {
          auto constraints = get_constraint_vector(j);
          if (constraints.size() == 0)
            continue;

          lid_to_gid.emplace_back(j);

          for (const auto &constraint : constraints)
            {
              c_pool_col.emplace_back(constraint.first);
              c_pool_val.emplace_back(constraint.second);
            }
          c_pool_row.emplace_back(c_pool_col.size());
        }

      std::vector<Number> diagonal_local_constrained(lid_to_gid.size());

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // compute i-th column of element stiffness matrix
          const auto ith_column = compute_ith_column_of_matrix(i);

          // apply local constraint matrix from left and from right
          //
          for (unsigned int j = 0; j < c_pool_row.size() - 1; j++)
            {
              // check if the result will zero, so that we can skipp the
              // following computations
              const auto scale_iterator =
                std::find(c_pool_col.begin() + c_pool_row[j],
                          c_pool_col.begin() + c_pool_row[j + 1],
                          i);

              if (scale_iterator == c_pool_col.begin() + c_pool_row[j + 1])
                continue;

              // apply constraint matrix from the left
              Number temp = 0.0;
              for (unsigned int k = c_pool_row[j]; k < c_pool_row[j + 1]; k++)
                temp += c_pool_val[k] * ith_column[c_pool_col[k]];

              // apply constraint matrix from the right
              diagonal_local_constrained[j] +=
                temp *
                c_pool_val[std::distance(c_pool_col.begin(), scale_iterator)];
            }
        }

      for (unsigned int j = 0; j < c_pool_row.size() - 1; j++)
        diagonal_global[lid_to_gid[j]] += diagonal_local_constrained[j];
    }

  diagonal_global.print(std::cout);
}