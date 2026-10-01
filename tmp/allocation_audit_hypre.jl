using HYPRE, SparseArrays, LinearAlgebra, Statistics
BLAS.set_num_threads(1)
const HI = HYPRE.LibHYPRE.HYPRE_Int
const HB = HYPRE.LibHYPRE.HYPRE_BigInt
const HC = HYPRE.LibHYPRE.HYPRE_Complex

buffers() = (ncols=HI[], rows=HB[], cols=HB[], values=HC[], lastinds=Int[])
function refill!(buf, A)
    n = size(A,1)
    resize!(buf.ncols,n); resize!(buf.rows,n); resize!(buf.lastinds,n)
    resize!(buf.cols,nnz(A)); resize!(buf.values,nnz(A))
    fill!(buf.ncols,0)
    @inbounds for i in 1:n
        buf.rows[i] = i
    end
    r = rowvals(A); v = nonzeros(A)
    @inbounds for j in 1:size(A,2), k in nzrange(A,j)
        buf.ncols[r[k]] += 1
    end
    offset = 0
    @inbounds for i in 1:n
        buf.lastinds[i] = offset
        offset += buf.ncols[i]
    end
    @inbounds for j in 1:size(A,2), k in nzrange(A,j)
        row = r[k]
        pos = buf.lastinds[row] += 1
        buf.cols[pos] = j
        buf.values[pos] = v[k]
    end
    return HI(n),buf.ncols,buf.rows,buf.cols,buf.values
end

function fixture(n,band)
    diagonals = Pair{Int,Vector{Float64}}[0=>fill(Float64(2band+1),n)]
    for k in 1:band
        push!(diagonals,k=>fill(-1.0,n-k),-k=>fill(-1.0,n-k))
    end
    return spdiagm(diagonals...)
end

function report(label,f)
    f(); f()
    samples = [@timed f() for _ in 1:7]
    println((label=label,bytes=minimum(s.bytes for s in samples),
        median_ms=1000median(s.time for s in samples)))
end

function main()
    matrices = [fixture(20000,20),fixture(20000,14),fixture(19000,18)]
    buf = buffers()
    refill!(buf,first(matrices))
    # Refill all structural data, including equal-size but changed sparsity.
    for A in matrices
        reference = HYPRE.Internals.to_hypre_data(A,1,size(A,1))
        @assert reference == refill!(buf,A)
    end
    A = first(matrices)
    println((n=size(A,1),nnz=nnz(A),hypre_int_bytes=sizeof(HI),bigint_bytes=sizeof(HB)))
    report("existing CSC-to-HYPRE",()->HYPRE.Internals.to_hypre_data(A,1,size(A,1)))
    report("reused buffers, rebuilt contents",()->refill!(buf,A))
    report("existing changing structures",()->begin
        for B in matrices
            HYPRE.Internals.to_hypre_data(B,1,size(B,1))
        end
    end)
    report("reused buffers, changing structures",()->begin
        for B in matrices
            refill!(buf,B)
        end
    end)
end
main()
