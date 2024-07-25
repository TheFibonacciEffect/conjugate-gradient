# @ccall "./mylib.so".say_y(5::Cint)::Cvoid
cptr = @ccall "./mylib.so".alloc(3::Cint)::Ptr{Cint}
A = unsafe_wrap(Array,cptr , 3, own=true) # own gives julia owndership of the array
@ccall "./mylib.so".print_array(A::Ptr{Cint},3::Cint)::Cvoid
