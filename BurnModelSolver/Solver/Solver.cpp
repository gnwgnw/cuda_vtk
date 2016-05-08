//
// Created by tsv on 05.05.16.
//

#include <algorithm>
#include <fstream>
#include <iomanip>
#include "Solver.hpp"

#define THROW_INVALID_ARG(name) throw std::invalid_argument(#name " must be positive\n")
#define NAME_TO_OUTPUT(name) #name << " = " << name


Solver::Solver(size_t N)
        : N(N)
        , size(N * sizeof(float))
        , blocks(threads)
        , grids(N / threads + (N % threads ? 1 : 0))
{
    if (N < 1) {
        throw std::invalid_argument("N must be non-zero\n");
    }

    recalc_h();
    alloc_mem();
}

Solver::~Solver()
{
    free_mem();
}

void Solver::recalc_h()
{
    h = (x1 - x0) / N;
    is_x_consistent = false;
}

void Solver::save(const std::string& file_path, const std::string& file_name)
{
    if (!is_x_consistent) {
        fill_x();
    }

    std::ofstream file = get_file(file_path, file_name);

    char fill_char = '-';
    file << std::setw(31) << std::setfill(fill_char) << fill_char << std::setfill(' ') << std::endl;

    file << NAME_TO_OUTPUT(N) << std::endl
    << NAME_TO_OUTPUT(t) << std::endl
    << NAME_TO_OUTPUT(tau) << std::endl
    << NAME_TO_OUTPUT(h) << std::endl;

    file << std::setw(31) << std::setfill(fill_char) << fill_char << std::setfill(' ') << std::endl;

    file << std::setw(15) << "X" << " " << std::setw(15) << "Y" << std::endl;

    file.precision(6);
    file << std::scientific;

    for (size_t i = 0; i < N; ++i) {
        file << std::setw(15) << x[i] << " " << std::setw(15) << y[i] << std::endl;
    }
}

std::ofstream Solver::get_file(const std::string& file_path, const std::string& file_name)
{
    std::string full_path(file_path);
    full_path += "/";
    full_path += file_name;

    std::ofstream file;
    file.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    file.open(full_path, std::ios_base::app);

    return file;
}

bool Solver::is_done()
{
    return done;
}

const std::vector<float>& Solver::get_x()
{
    if (!is_x_consistent) {
        fill_x();
    }
    return x;
}

const std::vector<float>& Solver::get_y()
{
    if (is_y_current) {
        return y;
    }

    copy_from_device();
    return y;
}

const float& Solver::get_t()
{
    return t;
}

const float& Solver::get_h()
{
    return h;
}

const float& Solver::get_tau()
{
    return tau;
}

void Solver::set_x0(const float& new_x0)
{
    x0 = new_x0;
    recalc_h();
}

void Solver::set_x1(const float& new_x1)
{
    x1 = new_x1;
    recalc_h();
}

void Solver::set_y(const std::vector<float>& new_y)
{
    y = new_y;
    set_N(y.size());
}

void Solver::set_tau(const float& new_tau)
{
    if (new_tau <= 0) {
        THROW_INVALID_ARG(tau);
    }

    tau = new_tau;
}

void Solver::set_h(const float& new_h)
{
    h = new_h;
    x1 = x0 + h * N;
    is_x_consistent = false;
}

void Solver::set_N(const size_t& new_N)
{
    if (new_N < 1) {
        THROW_INVALID_ARG(N);
    }

    N = new_N;
    size = N * sizeof(float);

    grids = (N / threads + (N % threads ? 1 : 0));

    free_mem();
    alloc_mem();

    recalc_h();
}

void Solver::set_t_end(const float& new_t_end)
{
    t_end = new_t_end;
}

void Solver::fill_x()
{
    float x_it = x0;
    float h = this->h;

    x[0] = x_it;
    std::generate(x.begin() + 1, x.end(), [&x_it, h]() {
        return x_it += h;
    });

    is_x_consistent = true;
}

const size_t& Solver::get_N()
{
    return N;
}

float* Solver::get_d_y_in()
{
    return d_y_in;
}

float* Solver::get_d_y_out()
{
    return d_y_out;
}

const size_t& Solver::get_threads()
{
    return threads;
}

const size_t& Solver::get_blocks()
{
    return blocks;
}

const size_t& Solver::get_grids()
{
    return grids;
}






