//
// Created by tsv on 07.05.16.
//

#include <fstream>
#include "BurnModelSolver.hpp"


const Params& BurnModelSolver::get_params()
{
    return params;
}

void BurnModelSolver::save(const std::string& file_path, const std::string& file_name)
{
    std::ofstream file = get_file(file_path, file_name);

    file << params << std::endl
    << "u = " << params.u(get_y()[1] / get_h()) << std::endl;

    Solver::save(file_path, file_name);
}
