cmake_minimum_required(VERSION 3.8)

# project(IterativeSolvers C CXX CUDA)
project(IterativeSolvers C)


# Pass options to NVCC and GCC
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -rdynamic -lineinfo")

add_executable(conjugate-gradient ./conjugate-gradient_cpu.c)
# set_property(TARGET IterativeSolvers PROPERTY LINK_FLAGS "-lm")
# set_target_properties(IterativeSolvers PROPERTIES LINKER_LANGUAGE CXX)


