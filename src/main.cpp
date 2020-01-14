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

      auto compute_ith_column_of_matrix = [&](unsigned int col) {
        Vector<Number> v(dofs_per_cell);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          v[j] = cell_matrix[j][col];

        return v;
      };

      std::vector<std::tuple<unsigned int, unsigned int, Number, unsigned int>>
        locally_relevant_dof_indices;

      for (unsigned int i = 0; i < local_dof_indices.size(); i++)
        {
          const auto &local_dof_index = local_dof_indices[i];

          if (!constraint.is_constrained(local_dof_index))
            {
              locally_relevant_dof_indices.emplace_back(local_dof_index,
                                                        local_dof_index,
                                                        1.0,
                                                        i);
              continue;
            }

          for (const auto &c :
               *constraint.get_constraint_entries(local_dof_index))
            if (!constraint.is_constrained(c.first))
              locally_relevant_dof_indices.emplace_back(local_dof_index,
                                                        c.first,
                                                        c.second,
                                                        i);
        }

      std::sort(locally_relevant_dof_indices.begin(),
                locally_relevant_dof_indices.end(),
                [](const auto &a, const auto &b) {
                  if (std::get<1>(a) < std::get<1>(b))
                    return true;
                  return (std::get<1>(a) == std::get<1>(b)) &&
                         (std::get<3>(a) < std::get<3>(b));
                });

      locally_relevant_dof_indices.erase(
        unique(locally_relevant_dof_indices.begin(),
               locally_relevant_dof_indices.end(),
               [](const auto &a, const auto &b) {
                 return (std::get<1>(a) == std::get<1>(b)) &&
                        (std::get<3>(a) == std::get<3>(b));
               }),
        locally_relevant_dof_indices.end());

      // setup CSR-storage
      std::vector<unsigned int> c_pool_row_lid_to_gid;
      std::vector<unsigned int> c_pool_row{0};
      std::vector<unsigned int> c_pool_col;
      std::vector<Number>       c_pool_val;

      {
        if (locally_relevant_dof_indices.size() > 0)
          c_pool_row_lid_to_gid.emplace_back(
            std::get<1>(locally_relevant_dof_indices.front()));
        for (const auto &j : locally_relevant_dof_indices)
          {
            if (c_pool_row_lid_to_gid.back() != std::get<1>(j))
              {
                c_pool_row_lid_to_gid.push_back(std::get<1>(j));
                c_pool_row.push_back(c_pool_val.size());
              }

            c_pool_col.emplace_back(std::get<3>(j));
            c_pool_val.emplace_back(std::get<2>(j));
          }

        if (c_pool_val.size() > 0)
          c_pool_row.push_back(c_pool_val.size());
      }

      // local storage: buffer so that we access the global vector once
      // note: may be larger then dofs_per_cell in the presence of constraints!
      std::vector<Number> diagonal_local_constrained(
        c_pool_row_lid_to_gid.size(), Number(0.0));

      // loop over all columns of element stiffness matrix
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // compute i-th column of element stiffness matrix:
          // this could be simply performed as done at the moment with
          // matrix-free operator evaluation applied to a ith-basis vector
          const auto ith_column = compute_ith_column_of_matrix(i);

          // apply local constraint matrix from left and from right:
          // loop over all rows of transposed constrained matrix
          for (unsigned int j = 0; j < c_pool_row.size() - 1; j++)
            {
              // check if the result will be zero, so that we can skip the
              // following computations -> binary search
              const auto scale_iterator =
                std::lower_bound(c_pool_col.begin() + c_pool_row[j],
                                 c_pool_col.begin() + c_pool_row[j + 1],
                                 i);

              if (scale_iterator == c_pool_col.begin() + c_pool_row[j + 1])
                continue;

              if (*scale_iterator != i)
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

      // assembly results: add into global vector
      for (unsigned int j = 0; j < c_pool_row.size() - 1; j++)
        diagonal_global[c_pool_row_lid_to_gid[j]] +=
          diagonal_local_constrained[j];
    }

  diagonal_global.print(std::cout);
}