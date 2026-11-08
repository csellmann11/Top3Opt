using LinearAlgebra, Random, Statistics, Test
include("../src/refinement_utils/pu_tetra.jl")
BLAS.set_num_threads(1)

function original_pu_solve!(matrix,rhs,n)
    @inbounds for j in 1:n
        diagonal = matrix[j,j]
        for k in 1:j-1
            diagonal -= matrix[j,k]^2
        end
        isfinite(diagonal) && diagonal > 0 || throw(PosDefException(j))
        matrix[j,j] = sqrt(diagonal)
        for i in j+1:n
            value = matrix[i,j]
            for k in 1:j-1
                value -= matrix[i,k]*matrix[j,k]
            end
            matrix[i,j] = value/matrix[j,j]
        end
    end
    @inbounds for i in 1:n
        value = rhs[i]
        for j in 1:i-1
            value -= matrix[i,j]*rhs[j]
        end
        rhs[i] = value/matrix[i,i]
    end
    @inbounds for i in n:-1:1
        value = rhs[i]
        for j in i+1:n
            value -= matrix[j,i]*rhs[j]
        end
        rhs[i] = value/matrix[i,i]
    end
    return nothing
end

function candidate_lapack_solve!(matrix,rhs,n)
    n == 0 && return nothing
    A = @view matrix[1:n,1:n]
    b = @view rhs[1:n]
    _,info = LAPACK.potrf!('L',A)
    info == 0 || throw(PosDefException(info))
    @inbounds for i in 1:n
        isfinite(matrix[i,i]) && matrix[i,i] > 0 || throw(PosDefException(i))
    end
    LAPACK.potrs!('L',A,b)
    return nothing
end

function candidate_column_solve!(matrix,rhs,n)
    @inbounds for j in 1:n
        diagonal = matrix[j,j]
        isfinite(diagonal) && diagonal > 0 || throw(PosDefException(j))
        diagonal = sqrt(diagonal)
        matrix[j,j] = diagonal
        inverse = inv(diagonal)
        @simd for i in j+1:n
            matrix[i,j] *= inverse
        end
        for k in j+1:n
            value = matrix[k,j]
            @simd for i in k:n
                matrix[i,k] -= matrix[i,j]*value
            end
        end
        value = rhs[j]*inverse
        rhs[j] = value
        @simd for i in j+1:n
            rhs[i] -= matrix[i,j]*value
        end
    end
    @inbounds for i in n:-1:1
        value = rhs[i]
        @simd for j in i+1:n
            value -= matrix[j,i]*rhs[j]
        end
        rhs[i] = value/matrix[i,i]
    end
    return nothing
end

function candidate_dot_solve!(matrix,rhs,n)
    @inbounds for j in 1:n
        diagonal = matrix[j,j]
        @simd for k in 1:j-1
            diagonal -= matrix[j,k]^2
        end
        isfinite(diagonal) && diagonal > 0 || throw(PosDefException(j))
        matrix[j,j] = sqrt(diagonal)
        for i in j+1:n
            value = matrix[i,j]
            @simd for k in 1:j-1
                value -= matrix[i,k]*matrix[j,k]
            end
            matrix[i,j] = value/matrix[j,j]
        end
    end
    @inbounds for i in 1:n
        value = rhs[i]
        @simd for j in 1:i-1
            value -= matrix[i,j]*rhs[j]
        end
        rhs[i] = value/matrix[i,i]
    end
    @inbounds for i in n:-1:1
        value = rhs[i]
        @simd for j in i+1:n
            value -= matrix[j,i]*rhs[j]
        end
        rhs[i] = value/matrix[i,i]
    end
    return nothing
end

function solver_trial(f,A,b,matrix,rhs,n,repetitions)
    elapsed = @elapsed for _ in 1:repetitions
        @inbounds for j in 1:n,i in j:n
            matrix[i,j] = A[i,j]
        end
        copyto!(rhs,1,b,1,n)
        f(matrix,rhs,n)
    end
    return elapsed/repetitions
end

function benchmark_patch_solvers()
    Random.seed!(1417)
    @testset "Exact active-view patch solvers" begin
        for n in (3,6,12,21,32,45,63,81,102,180)
            R = randn(n,n)
            A = R*R'+I
            b = randn(n)
            reference = A\b
            capacity = n+13
            matrix = fill(NaN,capacity,capacity)
            rhs = fill(NaN,capacity)
            times = Float64[]
            allocations = Int[]
            for solver in (original_pu_solve!,candidate_lapack_solve!,candidate_column_solve!,candidate_dot_solve!,pu3d_solve!)
                solver_trial(solver,A,b,matrix,rhs,n,2)
                @test rhs[1:n] ≈ reference rtol=3e-12
                @test all(isnan,rhs[n+1:end])
                @test all(isnan,matrix[n+1:end,n+1:end])
                @test all(isnan,matrix[1:n,n+1:end])
                repetitions = max(200,div(100_000,n))
                push!(times,median(solver_trial(solver,A,b,matrix,rhs,n,repetitions) for _ in 1:5))
                push!(allocations,@allocated solver_trial(solver,A,b,matrix,rhs,n,1))
                @test_throws PosDefException solver(zeros(n,n),ones(n),n)
                @test_throws PosDefException solver(fill(NaN,n,n),ones(n),n)
                nonfinite = Matrix{Float64}(I,n,n)
                nonfinite[1,1] = Inf
                @test_throws PosDefException solver(nonfinite,ones(n),n)
                @test solver(zeros(0,0),Float64[],0) === nothing
            end
            println((;n,original_us=1e6times[1],lapack_us=1e6times[2],column_us=1e6times[3],
                dot_us=1e6times[4],production_us=1e6times[5],lapack_speedup=times[1]/times[2],
                column_speedup=times[1]/times[3],dot_speedup=times[1]/times[4],allocations))
            flush(stdout)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && benchmark_patch_solvers()
