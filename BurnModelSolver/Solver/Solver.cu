#include "Solver.hpp"
#include "cuda_utils.cuh"


void Solver::step() {
  if (!done) {
    next_step();
    cuda_check_error(cudaGetLastError());

    std::swap(d_y_in, d_y_out);

    t += tau;
    if (t > t_end) {
      done = true;
    }

    if (is_y_current) {
      is_y_current = false;
    }
  }
}

void Solver::alloc_mem() {
  y.resize(N);
  x.resize(N);
  cuda_check_error(cudaMalloc((void **) &d_y_in, size));
  cuda_check_error(cudaMalloc((void **) &d_y_out, size));
  cuda_check_error(cudaMemcpy(d_y_in, y.data(), size, cudaMemcpyHostToDevice));
  cuda_check_error(cudaMemcpy(d_y_out, y.data(), size, cudaMemcpyHostToDevice));
}

void Solver::free_mem() {
  cuda_check_error(cudaFree(d_y_in));
  cuda_check_error(cudaFree(d_y_out));
}

void Solver::copy_from_device() {
  cuda_check_error(cudaMemcpy(y.data(), d_y_in, size, cudaMemcpyDeviceToHost));
  is_y_current = true;
}
