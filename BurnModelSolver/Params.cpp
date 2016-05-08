//
// Created by tsv on 07.05.16.
//

#include "Params.hpp"

#define NAME_TO_OUTPUT(name) #name << " = " << name


void Params::fill_A()
{
    A = A_k;
}

void Params::fill_B()
{
    B = lambda * (T_max - T_s);
}

void Params::fill_C()
{
    C = G_t + C_pt * (T_s - T_s0);
}

void Params::fill_D()
{
    D = lambda / C_p;
}

void Params::fill_E()
{
    E = -E_a;
}

void Params::fill_P()
{
    P = p_k;
}

void Params::fill_Q()
{
    Q = q_r;
}

void Params::create_params()
{
    fill_A();
    fill_B();
    fill_C();
    fill_D();
    fill_E();
    fill_P();
    fill_Q();

    fill_A_rt();
    fill_B_rt();
    fill_C_rt();
}

void Params::fill_A_rt()
{
    A_rt = R_s * T_s;
}

void Params::fill_B_rt()
{
    B_rt = R_s * T_max - 2.f * R_s * T_s + R_max * T_s;
}

void Params::fill_C_rt()
{
    C_rt = (T_max - T_s) * (R_max - R_s);
}

std::ostream& operator<<(std::ostream& os, const Params& params)
{
    os << NAME_TO_OUTPUT(params.A_k) << std::endl
    << NAME_TO_OUTPUT(params.E_a) << std::endl
    << NAME_TO_OUTPUT(params.G_t) << std::endl
    << NAME_TO_OUTPUT(params.C_pt) << std::endl
    << NAME_TO_OUTPUT(params.T_s0) << std::endl
    << NAME_TO_OUTPUT(params.T_s) << std::endl
    << NAME_TO_OUTPUT(params.T_max) << std::endl
    << NAME_TO_OUTPUT(params.R_s) << std::endl
    << NAME_TO_OUTPUT(params.R_max) << std::endl
    << NAME_TO_OUTPUT(params.lambda) << std::endl
    << NAME_TO_OUTPUT(params.C_p) << std::endl
    << NAME_TO_OUTPUT(params.p_k) << std::endl
    << NAME_TO_OUTPUT(params.q_r) << std::endl
    << NAME_TO_OUTPUT(params.rho_t) << std::endl;

    return os;
}

