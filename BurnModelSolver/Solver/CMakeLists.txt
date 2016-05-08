cmake_minimum_required(VERSION 3.5)

project(solver)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_FORCE_INLINES")

find_package(CUDA REQUIRED)

set(
        CUDA_SOURCE_FILES
        Solver.cu
        cuda_utils.cuh
)

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(
        cuda_solver
        ${CUDA_SOURCE_FILES}
)


set(
        SOURCE_FILES
        Solver.hpp
        Solver.cpp
)

add_library(
        solver
        ${SOURCE_FILES}
)

target_link_libraries(
        solver
        cuda_solver
        ${CUDA_LIBRARIES}
)
