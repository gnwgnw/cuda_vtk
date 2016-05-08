cmake_minimum_required(VERSION 3.5)

project(burn_model_solver)

add_subdirectory(Solver)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_FORCE_INLINES")

find_package(CUDA REQUIRED)

set(
        CUDA_SOURCE_FILES
        solve.cu
        Params.cu
        BurnModelSolver.cu
        Params.hpp
)

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(
        cuda_burn_model_solver
        ${CUDA_SOURCE_FILES}
)

set(
        SOURCE_FILES
        BurnModelSolver.hpp
        BurnModelSolver.cpp
        Params.hpp
        Params.cpp
)

add_library(
        burn_model_solver
        ${SOURCE_FILES}
)

target_link_libraries(
        burn_model_solver
        cuda_burn_model_solver
        solver
        ${CUDA_LIBRARIES}
)