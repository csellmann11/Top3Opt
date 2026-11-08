# Exact sparse 3x3-block factorization prototype; production code is unchanged.
isdefined(@__MODULE__,:pu_clamped_problem) || include("pu_dwr.jl")
using Statistics

@inline function prototype_block(A,i,j)
    r,s = 3i-3,3j-3
    return SMatrix{3,3,Float64,9}(ntuple(Val(9)) do k
        row,column = r+mod1(k,3),s+div(k-1,3)+1
        i == j && row < column && return A[column,row]
        return A[row,column]
    end)
end

@inline function prototype_store!(A,B,i,j)
    r,s = 3i-3,3j-3
    @inbounds for b in 1:3,a in 1:3
        i == j && a < b && continue
        A[r+a,s+b] = B[a,b]
    end
end

@inline function prototype_chol3(A)
    d1 = A[1,1]
    isfinite(d1) && d1 > 0 || throw(PosDefException(1))
    l11 = sqrt(d1)
    l21,l31 = A[2,1]/l11,A[3,1]/l11
    d2 = A[2,2]-l21*l21
    isfinite(d2) && d2 > 0 || throw(PosDefException(2))
    l22 = sqrt(d2)
    l32 = (A[3,2]-l31*l21)/l22
    d3 = A[3,3]-l31*l31-l32*l32
    isfinite(d3) && d3 > 0 || throw(PosDefException(3))
    return SA[l11 0.0 0.0;l21 l22 0.0;l31 l32 sqrt(d3)]
end

@inline function prototype_right_solve(A,L)
    a = A[:,1]/L[1,1]
    b = (A[:,2]-L[2,1]*a)/L[2,2]
    c = (A[:,3]-L[3,1]*a-L[3,2]*b)/L[3,3]
    return SMatrix{3,3,Float64,9}(a[1],a[2],a[3],b[1],b[2],b[3],c[1],c[2],c[3])
end

function prototype_block_solve!(A,b,n)
    (n % 3 != 0 || n > 192) && return pu3d_solve!(A,b,n)
    nb = n÷3
    @inbounds for j in 1:nb
        L = prototype_chol3(prototype_block(A,j,j))
        prototype_store!(A,L,j,j)
        active = UInt64(0)
        for i in j+1:nb
            block = prototype_block(A,i,j)
            iszero(block) && continue
            prototype_store!(A,prototype_right_solve(block,L),i,j)
            active |= UInt64(1) << (i-1)
        end
        columns = active
        while columns != 0
            k = trailing_zeros(columns)+1
            Lk = prototype_block(A,k,j)
            rows = columns
            while rows != 0
                i = trailing_zeros(rows)+1
                update = prototype_block(A,i,j)*transpose(Lk)
                prototype_store!(A,prototype_block(A,i,k)-update,i,k)
                rows &= rows-1
            end
            columns &= columns-1
        end
    end
    @inbounds for i in 1:n
        value = b[i]
        for j in 1:i-1
            value -= A[i,j]*b[j]
        end
        b[i] = value/A[i,i]
    end
    @inbounds for i in n:-1:1
        value = b[i]
        for j in i+1:n
            value -= A[j,i]*b[j]
        end
        b[i] = value/A[i,i]
    end
    return nothing
end

function prototype_patch_matrix(c,node)
    w = first(c.workspaces)
    indices = pu3d_patch_number!(w,c,node)
    for (localtet,index) in enumerate(indices)
        ti,hat = c.incidence[index]
        pu3d_add_tetra!(w,c,ti,hat,w.maps[localtet])
    end
    ti,hat = c.incidence[first(indices)]
    center_dofs = w.dofs[first(w.maps)[hat]]
    permutation = [i for i in 1:w.ndofs if i ∉ center_dofs]
    append!(permutation,filter(!iszero,collect(center_dofs)))
    A = Matrix(Symmetric(w.matrix[1:w.ndofs,1:w.ndofs],:L))[permutation,permutation]
    return A,copy(w.rhs[permutation])
end

function prototype_solver_trial(solve!,A,b,work,rhs,repeats)
    for _ in 1:repeats
        copyto!(work,A);copyto!(rhs,b)
        solve!(work,rhs,length(b))
    end
    return sum(rhs)
end

function benchmark_block_prototype()
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(3)
    pu3d_prepare!(c,u,states,ed,pars,states.χ_vec,1.)
    cases = Dict{Int,Tuple{Matrix{Float64},Vector{Float64}}}()
    for node in eachindex(c.nodal)
        A,b = prototype_patch_matrix(c,node)
        length(b) % 3 == 0 && !haskey(cases,length(b)) && (cases[length(b)] = (A,b))
    end
    @testset "Sparse block prototype versus exact dense solve" begin
        for n in sort!(collect(keys(cases)))
            A,b = cases[n]
            work,rhs = copy(A),copy(b)
            prototype_block_solve!(work,rhs,n)
            @test rhs ≈ A\b rtol=2e-10 atol=1e-12
            prototype_solver_trial(pu3d_solve!,A,b,work,rhs,3)
            prototype_solver_trial(prototype_block_solve!,A,b,work,rhs,3)
            dense = minimum(@elapsed(prototype_solver_trial(pu3d_solve!,A,b,work,rhs,1000)) for _ in 1:3)
            blocks = minimum(@elapsed(prototype_solver_trial(prototype_block_solve!,A,b,work,rhs,1000)) for _ in 1:3)
            allocation = @allocated prototype_solver_trial(prototype_block_solve!,A,b,work,rhs,1)
            println((;n,dense_ns=dense*1e6,block_ns=blocks*1e6,speedup=dense/blocks,allocation))
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && benchmark_block_prototype()
