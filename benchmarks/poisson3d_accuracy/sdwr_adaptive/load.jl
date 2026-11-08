# Standalone accuracy experiment; no production/topology-optimization includes.
using LinearAlgebra, SparseArrays, Statistics, Printf, TOML
include("../quadrature.jl")
include("../problems.jl")
include("../geometry.jl")
include("../primal.jl")
include("../metrics.jl")
include("mesh.jl")
include("primal.jl")
include("goals.jl")
include("estimators.jl")
