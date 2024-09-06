using Libdl
using CUDA
using Plots
using Test

# TODO Make library!!
# Function to get the pointer of a CUDA array
function get_ptr(A)
    return Base.unsafe_convert(CuPtr{Cfloat}, A)
end
# Load the shared library
lib = Libdl.dlopen("./build/libconjugate_gradient_gpu.so")

# Define a wrapper for the `add_jl` function
function add_jl(A::CuArray{Float32}, B::CuArray{Float32}, blocks, threads)
    sym = Libdl.dlsym(lib, :add_jl)
    @ccall $sym(get_ptr(A)::CuPtr{Cfloat}, get_ptr(B)::CuPtr{Cfloat}, blocks::Cint, threads::Cint)::Cvoid
end

# Define a wrapper for the `laplace_gpu_jl` function
function laplace_gpu_jl(ddf::CuArray{Float32}, u::CuArray{Float32}, dx::Cfloat, d, L, N, index_mode, blocks, threads)
    sym = Libdl.dlsym(lib, :laplace_gpu_jl)
    @ccall $sym(get_ptr(ddf)::CuPtr{Cfloat}, get_ptr(u)::CuPtr{Cfloat}, dx::Cfloat, d::Cint, L::Cint, N::Cint, index_mode::Cuint, blocks::Cuint, threads::Cuint)::Cvoid
end

# Define a wrapper for the `inner_product_gpu` function
function inner_product_gpu(v::CuArray{Float64}, w::CuArray{Float64}, N::Cuint)
    sym = Libdl.dlsym(lib, :inner_product_gpu)
    result = Ref{Cdouble}()
    @ccall $sym(get_ptr(v)::CuPtr{Cdouble}, get_ptr(w)::CuPtr{Cdouble}, N::Cuint)::Cdouble
end

function neighbour_index_gpu(ind, direction, amount, L, d, N, index_mode)::Cint
    sym = Libdl.dlsym(lib, :neighbour_index_gpu)
    @ccall $sym(ind::Cint, direction::Cint, amount::Cint, L::Cint, d::Cint, N::Cint, index_mode::Cint)::Cint
end


# Define the wrapper function for `conjugate_gradient_gpu`
function conjugate_gradient_gpu(b::CuArray{Float32}, x::CuArray{Float32}, L, d)::Cfloat
    sym = Libdl.dlsym(lib, :conjugate_gradient_gpu)
    @ccall $sym(get_ptr(b)::CuPtr{Cfloat}, get_ptr(x)::CuPtr{Cfloat}, L::Cint, d::Cint)::Cfloat
end

@testset "indexing on GPU" begin
    @test neighbour_index_gpu(2,1,1,3,2,9,0) == 5
    # edges
    @test neighbour_index_gpu(2,0,1,3,2,9,0) == 9
    @test neighbour_index_gpu(3,0,-1,3,2,9,0) == 9
    
end;

Libdl.dlclose(lib) # Close the library explicitly.
