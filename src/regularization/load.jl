# Public entry point. Plain includes keep definitions editable in the caller's scope.
# See docs/regularization.md for the design -> regularization -> projection -> activity flow.
using LinearAlgebra, SparseArrays
using Krylov: CgWorkspace, cg!

include("projection.jl")
include("coefficient.jl")
include("types.jl")
include("tpfa_operator.jl")
include("constraints.jl")
include("diagnostics.jl")
include("gradient_penalty.jl")
include("helmholtz_filter.jl")
