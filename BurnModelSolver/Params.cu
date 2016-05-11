//
// Created by tsv on 06.05.16.
//

#include "Params.hpp"


__host__ __device__ float Params::RT(float g) {
  return A_rt + B_rt * g + C_rt * g * g;
}

__host__ __device__ float Params::u(float dg0) {
  return -(B * dg0 + Q) / (C * rho_t);
}
