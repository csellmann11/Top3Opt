# Original generic numbering retained for equivalence tests.
function reference_pu3d_patch_number!(w,c,node)
    empty!(w.keys)
    empty!(w.masks)
    first,last = c.offsets[node],c.offsets[node+1]-1
    resize!(w.maps,last-first+1)
    for (localtet,index) in enumerate(first:last)
        ti,hat = c.incidence[index]
        t = c.geometry.tets[ti]
        w.maps[localtet] = SVector{10,Int}(ntuple(Val(10)) do i
            key = pu3d_local_key(t,i)
            slot = get(w.keys,key,0)
            mask = pu3d_local_fixed(c,t,i,hat)
            if slot == 0
                slot = length(w.masks)+1
                w.keys[key] = slot
                push!(w.masks,mask)
            else
                w.masks[slot] = w.masks[slot] .| mask
            end
            return slot
        end)
    end
    resize!(w.dofs,length(w.masks))
    n = 0
    for i in eachindex(w.masks)
        mask = w.masks[i]
        a = mask[1] ? 0 : (n += 1)
        b = mask[2] ? 0 : (n += 1)
        d = mask[3] ? 0 : (n += 1)
        w.dofs[i] = SA[a,b,d]
    end
    w.ndofs = n
    w.max_used = max(w.max_used,n)
    if size(w.matrix,1) < n
        capacity = max(n,ceil(Int,1.25size(w.matrix,1)))
        w.matrix = zeros(capacity,capacity)
        resize!(w.rhs,capacity)
        resize!(w.weighted,capacity)
    end
    @inbounds for j in 1:n
        w.rhs[j] = w.weighted[j] = 0.0
        for i in j:n
            w.matrix[i,j] = 0.0
        end
    end
    return first:last
end

function reference_pu3d_solve!(matrix,rhs,n)
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

include("pu_assembly_reference.jl")

function reference_pu3d_patch!(w,c,node,observer)
    incidence = reference_pu3d_patch_number!(w,c,node)
    contribution = c.nodal[node]
    for (localtet,index) in enumerate(incidence)
        ti,hat = c.incidence[index]
        contribution += reference_pu3d_add_tetra!(w,c,ti,hat,w.maps[localtet])
    end
    try
        reference_pu3d_solve!(w.matrix,w.rhs,w.ndofs)
    catch error
        error isa PosDefException || rethrow()
        throw(ArgumentError("PU patch $node has a singular elastic system; check positive densities and physical constraints"))
    end
    @inbounds for i in 1:w.ndofs
        contribution += w.weighted[i]*w.rhs[i]/2
    end
    c.nodal[node] = contribution
    if observer !== nothing
        for (localtet,index) in enumerate(incidence)
            ti,hat = c.incidence[index]
            pu3d_observe(observer,w,c,node,ti,hat,w.maps[localtet])
        end
    end
    return nothing
end

