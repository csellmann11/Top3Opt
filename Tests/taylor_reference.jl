# Comparison implementation of the supplied NEM description: edge neighbours,
# nine Taylor derivatives, ridge regularization, and the variable-beta product
# rule. Kept as a benchmark reference; it is not a conservative face scheme.
function compute_reference_taylor_mat(cv, states, pars;
        beta=2pars.β0 .* states.h_vec.^2, ridge=1e-4)
    neighbours, ghosts = create_neigh_list(states,cv)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for sid in eachindex(states.χ_vec)
        ns = neighbours[sid]
        h = states.h_vec[sid]
        x = states.x_vec[sid]
        A = Matrix{Float64}(undef,length(ns),9)
        mapped = [n < 0 ? ghosts[n] : n for n in ns]
        for (k,n) in enumerate(ns)
            dx,dy,dz = (get_location(n,sid,x,cv.mesh.topo,ghosts,states,false)-x)/h
            A[k,:] .= (dx,dy,dz,dx^2/2,dx*dy,dy^2/2,dx*dz,dy*dz,dz^2/2)
        end
        G = A'*A
        alpha = ridge*tr(G)
        # Contract the solve to the three gradients and the Laplacian: no need
        # to materialize all nine derivative rows or separate global matrices.
        selectors = zeros(9,4)
        for k in 1:3
            selectors[k,k] = 1/h
        end
        selectors[[4,6,9],4] .= 1/h^2
        D = (cholesky(Symmetric(G+alpha*I)) \ selectors)' * A'
        grad = D[1:3,:]
        grad_beta = grad*(beta[mapped] .- beta[sid])
        weights = beta[sid]*D[4,:] + grad'*grad_beta
        append!(rows,fill(sid,length(ns)+1))
        append!(cols,mapped); push!(cols,sid)
        append!(vals,weights); push!(vals,-sum(weights))
    end
    N = length(states.χ_vec)
    return sparse(rows,cols,vals,N,N)
end
