using LinearAlgebra,Random
include("../src/refinement_utils/pu_tetra.jl")
BLAS.set_num_threads(1)

@noinline function try_pu_solve!(A,b,n)
    try
        pu3d_solve!(A,b,n)
    catch error
        error isa PosDefException || rethrow()
        throw(ArgumentError("Singular"))
    end
    return nothing
end

function solver_escape_probe()
    for n in (21,45,81,102)
        R=randn(n,n); K=R*R'+I; b=randn(n)
        A=zeros(n+13,n+13); rhs=zeros(n+13)
        for f in (pu3d_solve!,try_pu_solve!)
            for _ in 1:3
                A[1:n,1:n].=K;rhs[1:n].=b
                f(A,rhs,n)
            end
            A[1:n,1:n].=K;rhs[1:n].=b
            allocation=@allocated f(A,rhs,n)
            println((;n,method=string(f),allocation))
        end
    end
end
solver_escape_probe()
