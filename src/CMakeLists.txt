cmake_minimum_required(VERSION 3.8)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


project(IterativeSolvers C CXX CUDA)


# Pass options to NVCC and GCC
add_library(pdep_custom STATIC custom_pdep.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -rdynamic -lineinfo -Xptxas -v -dopt=on  -G --resource-usage -restrict -arch=sm_75")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -rdynamic -lineinfo -Xptxas -v")
set(src conjugate-gradient_cpu.cu conjugate-gradient_gpu.cu)

add_executable(main main.cu ${src})

add_executable(tests tests.cu ${src})
target_link_libraries(tests pdep_custom)
set_property(TARGET tests PROPERTY COMPILE_FLAGS "${COMPILE_FLAGS}")


# create shared libraries for julia
add_library(conjugate_gradient_gpu SHARED julia.cu ${src})
# set_target_properties(IterativeSolvers PROPERTIES LINKER_LANGUAGE CXX)


