# Includes the existing actual-HYPRE solve and per-step logging regression.
include("hypre_diagnostics.jl")

function hypre_conversion_fixtures()
    first_pattern = sparse([4.0 -1 0 0; -1 4 0 0; 0 0 4 -1; 0 0 -1 4])
    second_pattern = sparse([5.0 0 -2 0; 0 5 0 -2; -2 0 5 0; 0 -2 0 5])
    block = sparse(Matrix{Float64}(I,3,3))
    return [kron(first_pattern,block),kron(second_pattern,block),
        kron(spdiagm(-1=>-ones(7),0=>4ones(8),1=>-ones(7)),block),
        kron(spdiagm(0=>[3.0,4.0,5.0]),block)]
end

@testset "HYPRE buffers track changing structure and size" begin
    workspace = HypreConversionWorkspace()
    matrices = hypre_conversion_fixtures()
    @test size(matrices[1]) == size(matrices[2])
    @test nnz(matrices[1]) == nnz(matrices[2])
    @test matrices[1].rowval != matrices[2].rowval
    nonsymmetric = sparse([4.0 2 -1; 0 5 0; 3 1 6])
    stored_zero = sparse([1,1,2,3],[1,3,2,3],[3.0,0.0,4.0,5.0],3,3)
    @test nnz(stored_zero) == 4
    for A in [matrices; matrices[1:1]; [nonsymmetric,stored_zero]]
        original = copy(A)
        reference = HYPRE.Internals.to_hypre_data(A,1,size(A,1))
        @test hypre_conversion_data!(workspace,A) == reference
        @test A.colptr == original.colptr
        @test A.rowval == original.rowval
        @test A.nzval == original.nzval
    end
    @test_throws DimensionMismatch hypre_conversion_data!(workspace,spzeros(2,3))
    @test_throws ArgumentError hypre_conversion_data!(workspace,spzeros(0,0))
    @test_throws DimensionMismatch solve_lse_hypre(matrices[1],ones(2); workspace)
end

@testset "HYPRE uses current matrix after buffer reuse" begin
    workspace = HypreConversionWorkspace()
    matrices = hypre_conversion_fixtures()
    for A in [matrices; matrices[1:1]]
        exact = sin.(1:size(A,1))
        rhs = A*exact
        solution = solve_lse_hypre(A,rhs; workspace)
        @test norm(A*solution-rhs)/norm(rhs) <= 1e-3
        @test solution ≈ exact rtol=1e-3
    end

    # The native matrix owns its values after assembly. Reusing Julia staging
    # storage must not change an already-created matrix.
    A = first(matrices)
    native = _hypre_matrix(A,workspace)
    solver = b = x = nothing
    try
        fill!(workspace.values,0.0)
        exact = cos.(1:size(A,1))
        rhs = A*exact
        b = HYPRE.HYPREVector(rhs)
        x = zero(b)
        solver = HYPRE.PCG(;Tol=1e-10,MaxIter=100,PrintLevel=0)
        HYPRE.solve!(solver,x,native,b)
        solution = similar(rhs)
        copy!(solution,x)
        @test norm(A*solution-rhs)/norm(rhs) <= 1e-9
    finally
        for object in (solver,x,b,native)
            object === nothing || Base.finalize(object)
        end
    end
end

function reused_conversion_bytes(workspace,A)
    return @allocated hypre_conversion_data!(workspace,A)
end

@testset "Default HYPRE workspaces are task-local and reusable" begin
    workspace = _hypre_conversion_workspace()
    @test _hypre_conversion_workspace() === workspace
    other = fetch(@async _hypre_conversion_workspace())
    @test other !== workspace
    @test other.values !== workspace.values
    @test _hypre_conversion_workspace() === workspace

    # Existing capacity needs no matrix-sized allocation on subsequent calls.
    A = hypre_conversion_fixtures()[3]
    hypre_conversion_data!(workspace,A)
    reused_conversion_bytes(workspace,A)
    @test reused_conversion_bytes(workspace,A) == 0
end
