#include "BurnModelSolver.hpp"
#include "Solver/cuda_utils.cuh"


BurnModelSolver::~BurnModelSolver() {
  if (d_params) {
    cudaFree(d_params);
  }
}

void BurnModelSolver::set_params(const Params &new_params) {
  params = new_params;

  size_t param_size = sizeof(Params);

  if (d_params) {
    cudaFree(d_params);
  }
  cuda_check_error(cudaMalloc((void **) &d_params, param_size));
  cuda_check_error(cudaMemcpy(d_params, &params, param_size, cudaMemcpyHostToDevice));
}
