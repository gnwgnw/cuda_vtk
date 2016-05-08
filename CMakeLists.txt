cmake_minimum_required(VERSION 3.5)

project(cuda_vtk)

add_subdirectory(BurnModelSolver)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_FORCE_INLINES")

set(
        SOURCE_FILES
        main.cpp
)

add_executable(
        out
        ${SOURCE_FILES}
)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

target_link_libraries(
        out
        burn_model_solver
        ${VTK_LIBRARIES}
)
