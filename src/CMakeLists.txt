cmake_minimum_required(VERSION 3.8)

project(IterativeSolvers CXX CUDA)


# Pass options to NVCC and GCC
# "-Xcompiler -rdynamic -lineinfo" -> needed for cuda-memcheck
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -rdynamic -lineinfo")

set(src common.h ./include/conjugate-gradient_cpu.h)

add_executable(IterativeSolvers main.c ${src})

# set_property(TARGET IterativeSolvers PROPERTY LINK_FLAGS "-lm")
# add_executable(gram-schmitt gram-schmitt.cu)


