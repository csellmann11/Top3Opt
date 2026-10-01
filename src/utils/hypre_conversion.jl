using SparseArrays
import HYPRE

"""
Reusable Julia storage for CSC-to-HYPRE conversion. Every conversion rebuilds
all entries from the current matrix, so changing topology, ordering, or matrix
size does not invalidate the workspace. Native matrices and AMG hierarchies
are still created and destroyed separately for every solve.
"""
struct HypreConversionWorkspace
    ncols::Vector{HYPRE.LibHYPRE.HYPRE_Int}
    rows::Vector{HYPRE.LibHYPRE.HYPRE_BigInt}
    cols::Vector{HYPRE.LibHYPRE.HYPRE_BigInt}
    values::Vector{HYPRE.LibHYPRE.HYPRE_Complex}
    lastinds::Vector{Int}
end

HypreConversionWorkspace() = HypreConversionWorkspace(
    HYPRE.LibHYPRE.HYPRE_Int[], HYPRE.LibHYPRE.HYPRE_BigInt[],
    HYPRE.LibHYPRE.HYPRE_BigInt[], HYPRE.LibHYPRE.HYPRE_Complex[], Int[])

const _HYPRE_CONVERSION_WORKSPACE_KEY = gensym(:toopt_hypre_conversion)

function _hypre_conversion_workspace()
    storage = task_local_storage()
    task = current_task()
    entry = get(storage,_HYPRE_CONVERSION_WORKSPACE_KEY,nothing)
    # Check ownership too, in case a caller propagates a task's storage.
    if entry !== nothing && entry[1] === task
        return entry[2]::HypreConversionWorkspace
    end
    workspace = HypreConversionWorkspace()
    storage[_HYPRE_CONVERSION_WORKSPACE_KEY] = (task,workspace)
    return workspace
end

"""
    hypre_conversion_data!(workspace, A)

Rebuild the package's row-packed input format using reusable buffers. Returned
arrays belong to `workspace` and are overwritten by the next conversion.
This retains buffer capacity, not connectivity or matrix values.
"""
function hypre_conversion_data!(workspace::HypreConversionWorkspace, A::SparseMatrixCSC)
    n = size(A,1)
    size(A,2) == n || throw(DimensionMismatch("HYPRE requires a square matrix"))
    n > 0 || throw(ArgumentError("HYPRE requires a nonempty matrix"))
    # Validate native index widths before allocating buffers or calling C.
    nrows = HYPRE.LibHYPRE.HYPRE_Int(n)
    HYPRE.LibHYPRE.HYPRE_BigInt(n)
    HYPRE.LibHYPRE.HYPRE_Int(nnz(A))

    ncols,rows,cols,values,lastinds = workspace.ncols,workspace.rows,
        workspace.cols,workspace.values,workspace.lastinds
    resize!(ncols,n); resize!(rows,n); resize!(lastinds,n)
    resize!(cols,nnz(A)); resize!(values,nnz(A))
    fill!(ncols,0)
    @inbounds for i in 1:n
        rows[i] = i
    end
    source_rows = rowvals(A)
    source_values = nonzeros(A)
    @inbounds for j in 1:n, k in nzrange(A,j)
        ncols[source_rows[k]] += 1
    end
    offset = 0
    @inbounds for i in 1:n
        lastinds[i] = offset
        offset += ncols[i]
    end
    @inbounds for j in 1:n, k in nzrange(A,j)
        row = source_rows[k]
        index = lastinds[row] += 1
        cols[index] = j
        values[index] = source_values[k]
    end
    return nrows,ncols,rows,cols,values
end

function _hypre_matrix(A::SparseMatrixCSC, workspace::HypreConversionWorkspace)
    nrows,ncols,rows,cols,values = hypre_conversion_data!(workspace,A)
    matrix = HYPRE.HYPREMatrix(HYPRE.MPI.COMM_SELF,1,size(A,1))
    try
        HYPRE.LibHYPRE.@check HYPRE.LibHYPRE.HYPRE_IJMatrixSetValues(
            matrix,nrows,ncols,rows,cols,values)
        HYPRE.Internals.assemble_matrix(matrix)
    catch
        Base.finalize(matrix)
        rethrow()
    end
    return matrix
end
