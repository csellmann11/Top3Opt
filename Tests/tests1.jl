




using LinearAlgebra 
using SparseArrays 



A = sprand(100,100,0.01)


a = diag(A) |> collect



B = rand(10,10)

v = rand(10)

V = diagm(v)

v .* B == V * B

B .* v' == B * V


