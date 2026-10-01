using SparseArrays, LinearAlgebra
include("../src/density_implicit.jl")
include("../src/density_timestepping.jl")
BLAS.set_num_threads(1)

function audit(n=10_000)
    R = spdiagm(-1=>fill(10.,n-1),0=>fill(-20.,n),1=>fill(10.,n-1))
    R[1,1] = R[end,end] = -10.
    h = ones(n)
    rhs = [0.5+0.2sin(0.01i) for i in 1:n]
    result = similar(rhs)
    ctor() = DensityImplicitCache(R,15.,1.,h;beta_in_operator=true)
    cache = ctor()
    implicit_density_solve!(result,rhs,cache)
    density_substeps(R,15.,1.;beta_in_operator=true,update_mode=:implicit)
    wsctor() = Krylov.GmresWorkspace(cache.scaled_matrix,rhs;memory=30)
    wsctor()
    GC.gc()
    constructor_bytes = @allocated ctor()
    workspace_bytes = @allocated wsctor()
    solve_bytes = @allocated implicit_density_solve!(result,rhs,cache)
    diagnostic_bytes = @allocated density_substeps(R,15.,1.;beta_in_operator=true,update_mode=:implicit)
    println((n=n,nnz=nnz(R),constructor_bytes=constructor_bytes,
             gmres_workspace_bytes=workspace_bytes,solve_bytes=solve_bytes,
             row_bound_diagnostic_bytes=diagnostic_bytes,
             stored_scaled_matrix_bytes=Base.summarysize(cache.scaled_matrix),
             stored_gmres_workspace_bytes=Base.summarysize(cache.workspace),
             krylov_source=pathof(Krylov)))
end
audit()
