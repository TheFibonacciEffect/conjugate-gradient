cmake_minimum_required(VERSION 3.8)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


project(IterativeSolvers C CXX CUDA)


# Pass options to NVCC and GCC
# "-Xcompiler -rdynamic -lineinfo" -> needed for cuda-memcheck
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -rdynamic -lineinfo")

set(src common.h ./conjugate-gradient_cpu.h conjugate-gradient_cpu.cu)

add_executable(IterativeSolvers main.cu ${src})
set_property(TARGET IterativeSolvers PROPERTY COMPILE_FLAGS "${COMPILE_FLAGS}")
# set_target_properties(IterativeSolvers PROPERTIES LINKER_LANGUAGE CXX)
# set_property(TARGET IterativeSolvers PROPERTY LINK_FLAGS "-lm")
# add_executable(gram-schmitt gram-schmitt.cu)


