using Libdl
using CUDA

function get_ptr(A)
    Base.unsafe_convert(CuPtr{Cfloat},A)
end
lib = Libdl.dlopen("./build/libconjugate_gradient_gpu.so") # Open the library explicitly.
sym = Libdl.dlsym(lib, :add_jl)  # Get a symbol for the function to call.
N = 10
A = CUDA.fill(1.0f0, N)
B = CUDA.fill(3.0f0, N)
# 
@ccall $sym(get_ptr(A)::CuPtr{Cfloat},get_ptr(B)::CuPtr{Cfloat},9::Cint,1::Cint)::Cvoid # Use the pointer `sym` instead of the library.symbol tuple.
A

add = Libdl.dlsym(lib, :add)

Libdl.dlclose(lib) # Close the library explicitly.

# CuModule("../build/libconjugate_gradient_gpu.so")
