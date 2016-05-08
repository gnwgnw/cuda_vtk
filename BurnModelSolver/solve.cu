#include "Params.hpp"
#include "BurnModelSolver.hpp"

__global__ void _solver_step(float* g_in, float* g_out, size_t N, float h, float tau, Params* params)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > 0 && i < (N - 1)) {
        float dg0 = (g_in[1] - g_in[0]) / h;
        float gl = g_in[i - 1];
        float g = g_in[i];
        float gr = g_in[i + 1];

        float RT = params->RT(g);

        g_out[i] = (
                               RT / params->P *
                               (
                                       params->D * (gr - 2 * g + gl) / (h * h)
                                       - (params->B * dg0 + params->Q) / params->C * (gr - gl) / (2.f * h)
                               )
                               + (1.f - g) * params->A * expf(params->E / RT)
                       ) * tau
                       + g;
    }

    if (i == (N - 1)) {
        g_out[i] = g_out[i - 1];
    }
}

void BurnModelSolver::next_step()
{
    if (!d_params) {
        throw std::invalid_argument("Params not setted\n");
    }

    _solver_step<<<get_grids(), get_blocks()>>>(get_d_y_in(), get_d_y_out(), get_N(), get_h(), get_tau(), d_params);
    cudaDeviceSynchronize();
}
