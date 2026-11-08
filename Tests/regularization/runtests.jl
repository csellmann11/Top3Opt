using Test, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "..", "src", "regularization", "load.jl"))
include("projection.jl")
include("kernels.jl")
