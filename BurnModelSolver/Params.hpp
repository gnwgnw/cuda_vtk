//
// Created by tsv on 06.05.16.
//

#ifndef CUDA_VTK_PARAMS_HPP
#define CUDA_VTK_PARAMS_HPP

#include <iostream>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

class Params {
    private:
        void fill_A();
        void fill_B();
        void fill_C();
        void fill_D();
        void fill_E();
        void fill_P();
        void fill_Q();

        void fill_A_rt();
        void fill_B_rt();
        void fill_C_rt();

    public:
        float A_k;
        float E_a;
        float G_t;
        float C_pt;
        float T_s0;
        float T_s;
        float T_max;
        float R_s;
        float R_max;
        float lambda;
        float C_p;
        float p_k;
        float q_r;
        float rho_t;

        float A;
        float B;
        float C;
        float D;
        float E;
        float P;
        float Q;

        float A_rt;
        float B_rt;
        float C_rt;

        void create_params();

        __host__ __device__ float RT(float g);
        __host__ __device__ float u(float dg0);

        friend std::ostream& operator<<(std::ostream& os, const Params& params);
};

std::ostream& operator<<(std::ostream& os, const Params& params);


#endif //CUDA_VTK_PARAMS_HPP
