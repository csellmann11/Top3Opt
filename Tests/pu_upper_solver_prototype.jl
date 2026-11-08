# Contiguous upper-factor experiment; production solver is unchanged.
isdefined(@__MODULE__,:prototype_patch_matrix) || include("pu_block_solver_prototype.jl")

function prototype_upper_solve!(A,b,n)
    @inbounds for j in 1:n,i in 1:j-1
        A[i,j] = A[j,i]
    end
    @inbounds for j in 1:n
        total = 0.0
        @simd for k in 1:j-1
            total += A[k,j]*A[k,j]
        end
        diagonal = A[j,j]-total
        isfinite(diagonal) && diagonal > 0 || throw(PosDefException(j))
        diagonal = sqrt(diagonal)
        A[j,j] = diagonal
        for i in j+1:n
            product = 0.0
            @simd for k in 1:j-1
                product += A[k,j]*A[k,i]
            end
            A[j,i] = (A[j,i]-product)/diagonal
        end
    end
    @inbounds for i in 1:n
        product = 0.0
        @simd for j in 1:i-1
            product += A[j,i]*b[j]
        end
        b[i] = (b[i]-product)/A[i,i]
    end
    @inbounds for i in n:-1:1
        value = b[i]/A[i,i]
        b[i] = value
        @simd for j in 1:i-1
            b[j] -= A[j,i]*value
        end
    end
    return nothing
end

function benchmark_upper_prototype()
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(3)
    pu3d_prepare!(c,u,states,ed,pars,states.χ_vec,1.)
    cases = Dict{Int,Tuple{Matrix{Float64},Vector{Float64}}}()
    for node in eachindex(c.nodal)
        A,b = prototype_patch_matrix(c,node)
        !haskey(cases,length(b)) && (cases[length(b)] = (A,b))
    end
    @testset "Contiguous upper-factor prototype versus direct solve" begin
        for n in sort!(collect(keys(cases)))
            A,b = cases[n]
            work,rhs = copy(A),copy(b)
            prototype_upper_solve!(work,rhs,n)
            @test rhs ≈ A\b rtol=2e-10 atol=1e-12
            prototype_solver_trial(pu3d_solve!,A,b,work,rhs,3)
            prototype_solver_trial(prototype_upper_solve!,A,b,work,rhs,3)
            dense = minimum(@elapsed(prototype_solver_trial(pu3d_solve!,A,b,work,rhs,2000)) for _ in 1:3)
            upper = minimum(@elapsed(prototype_solver_trial(prototype_upper_solve!,A,b,work,rhs,2000)) for _ in 1:3)
            allocation = @allocated prototype_solver_trial(prototype_upper_solve!,A,b,work,rhs,1)
            println((;n,dense_ns=dense*5e5,upper_ns=upper*5e5,speedup=dense/upper,allocation))
        end
        # Padding matches actual reusable workspace leading dimensions.
        for n in (3,21,45,81,102,180)
            Random.seed!(n)
            X = randn(n,n)
            A = X*X'+n*I
            b = randn(n)
            work,rhs = zeros(n+17,n+17),zeros(n+17)
            work[1:n,1:n] .= A
            rhs[1:n] .= b
            prototype_upper_solve!(work,rhs,n)
            @test rhs[1:n] ≈ A\b rtol=2e-12
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && benchmark_upper_prototype()
