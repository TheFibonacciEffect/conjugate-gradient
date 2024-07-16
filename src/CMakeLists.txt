cmake_minimum_required(VERSION 3.8)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


project(IterativeSolvers C CXX CUDA)


# Pass options to NVCC and GCC
add_library(pdep_custom STATIC custom_pdep_stuff.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -rdynamic -lineinfo -Xptxas -v")

set(src common.h ./conjugate-gradient_cpu.h conjugate-gradient_cpu.cu preconditioner_gpu.h preconditioner_gpu.cu)

add_executable(IterativeSolvers main.cu ${src})
target_link_libraries(IterativeSolvers pdep_custom)
set_property(TARGET IterativeSolvers PROPERTY COMPILE_FLAGS "${COMPILE_FLAGS}")

# create shared libraries for julia
add_library(conjugate_gradient SHARED conjugate-gradient_cpu.cu)
add_library(conjugate_gradient_gpu SHARED main.cu)
# set_target_properties(IterativeSolvers PROPERTIES LINKER_LANGUAGE CXX)


