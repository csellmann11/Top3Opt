const HIERARCHICAL_TET_EDGES = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))

"""P1 vertex hats plus globally shared quadratic edge bubbles, without fitting.

Physical Dirichlet nodes/edges are removed. Every diagonal is a positive basis
energy; no patch rank test, artificial-boundary selection, or regularization is
needed. This accuracy prototype retains sparse K for correction iterations
and direct validation; it does not yet use a matrix-free implementation.
"""
function hierarchical_geometry(fan)
    nv,nt = size(fan.points,2),length(fan.volumes)
    edges = Tuple{Int,Int}[]
    lookup = Dict{Tuple{Int,Int},Int}()
    dofs = Matrix{Int}(undef,10,nt)
    rows,columns,values = Vector{Int}(undef,100nt),Vector{Int}(undef,100nt),Vector{Float64}(undef,100nt)
    cursor = 0
    for tet in 1:nt
        nodes = fan.tet_nodes[:,tet]
        dofs[1:4,tet] .= nodes
        for (edge,(a,b)) in enumerate(HIERARCHICAL_TET_EDGES)
            key = minmax(nodes[a],nodes[b])
            id = get!(lookup,key) do
                push!(edges,key)
                nv+length(edges)
            end
            dofs[4+edge,tet] = id
        end
        G = fan.gradients[tet]
        gradients = [hcat(G,[4 .* ((vertex==a ? G[:,b] : zeros(3))+
            (vertex==b ? G[:,a] : zeros(3))) for (a,b) in HIERARCHICAL_TET_EDGES]...)
            for vertex in 1:4]
        total = sum(gradients)
        K = fan.volumes[tet]/20*(transpose(total)*total+
            sum(transpose(g)*g for g in gradients))
        for j in 1:10,i in 1:10
            cursor += 1
            rows[cursor],columns[cursor],values[cursor] = dofs[i,tet],dofs[j,tet],K[i,j]
        end
    end
    nd = nv+length(edges)
    boundary = vcat(fan.boundary_nodes,[(a,b) in fan.boundary_edges for (a,b) in edges])
    K = sparse(rows,columns,values,nd,nd)
    dropzeros!(K)
    diagonal = diag(K)
    all(>(0),diagonal) || error("A hierarchical basis energy is not positive")
    free = findall(!,boundary)
    free_edges = filter(>(nv),free)
    return (;K,diagonal,free,free_edges,boundary,dofs,edges,vertex_count=nv)
end

"""Primal and linearized adjoint residuals in the same conforming basis."""
function hierarchical_residuals(mesh,fan,geometry,primal,dual,problem,goal;order=6,U=nothing,Z=nothing)
    U === nothing && (U = reconstruct_fan(mesh,fan,primal.u))
    Z === nothing && (Z = reconstruct_fan(mesh,fan,dual.u))
    length(U) == size(fan.points,2) && length(Z) == size(fan.points,2) ||
        throw(DimensionMismatch("Explicit reconstructions must contain all fan vertices"))
    bu,bz = zeros(size(geometry.K,1)),zeros(size(geometry.K,1))
    consistency,primal_work = 0.0,0.0
    rule = tetra_rule(order)
    shapes = vcat(rule.barycentric,
        reduce(vcat,[transpose(4 .* rule.barycentric[a,:].*rule.barycentric[b,:])
            for (a,b) in HIERARCHICAL_TET_EDGES]))
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:,tet]
        local_u,local_z = U[ids],Z[ids]
        G,volume = fan.gradients[tet],fan.volumes[tet]
        gu,gz = G*local_u,G*local_z
        xyz = fan.points[:,ids]*rule.barycentric
        average_gradient = hcat(G,[G[:,a]+G[:,b] for (a,b) in HIERARCHICAL_TET_EDGES]...)
        local_bu = -volume*transpose(average_gradient)*gu
        local_bz = -volume*transpose(average_gradient)*gz
        consistency -= volume*dot(gu,gz)
        primal_work -= volume*dot(gu,gu)
        for q in eachindex(rule.weights)
            N = view(rule.barycentric,:,q)
            x = view(xyz,:,q)
            u,z = dot(local_u,N),dot(local_z,N)
            f = problem.rhs(x)
            g = goal.derivative_value(x,u)
            gg = goal.derivative_gradient(x,gu)
            weight = volume*rule.weights[q]
            local_bu .+= (weight*f).*view(shapes,:,q)
            local_bz .+= (weight*g).*view(shapes,:,q)
            for a in 1:4
                local_bz[a] += weight*sum(gg[d]*G[d,a] for d in 1:3)
            end
            for (edge,(a,b)) in enumerate(HIERARCHICAL_TET_EDGES)
                local_bz[4+edge] += 4weight*sum(gg[d]*(N[a]*G[d,b]+N[b]*G[d,a]) for d in 1:3)
            end
            consistency += weight*f*z
            primal_work += weight*f*u
        end
        bu[geometry.dofs[:,tet]] .+= local_bu
        bz[geometry.dofs[:,tet]] .+= local_bz
    end
    return (;bu,bz,consistency,primal_work)
end

"""Embed original VEM nodal values in the hierarchical conforming fan space."""
function hierarchical_prolongation(mesh,fan,geometry)
    nv = fan.vertex_count
    rows,columns,values = Int[],Int[],Float64[]
    for node in 1:nv
        push!(rows,node); push!(columns,node); push!(values,1.0)
    end
    for (face,ids) in enumerate(mesh.faces),node in ids
        push!(rows,nv+face); push!(columns,node); push!(values,1/length(ids))
    end
    for (cell,ids) in enumerate(mesh.cells),node in ids
        push!(rows,nv+fan.face_count+cell); push!(columns,node); push!(values,1/length(ids))
    end
    return sparse(rows,columns,values,size(geometry.K,1),nv)
end

function hierarchical_pcg_checkpoints(A,b,precondition;checkpoints=(4,8,16),tolerance=1e-12)
    x,r = zeros(length(b)),copy(b)
    z = precondition(r)
    p = copy(z)
    rz = dot(r,z)
    threshold = tolerance*max(norm(b),eps())
    output = NamedTuple[]
    completed = 0
    for step in 1:maximum(checkpoints)
        if norm(r) > threshold
            Ap = A*p
            denominator = dot(p,Ap)
            denominator > 0 || error("PCG encountered a nonpositive direction")
            alpha = rz/denominator
            x .+= alpha.*p
            r .-= alpha.*Ap
            z = precondition(r)
            next_rz = dot(r,z)
            p .= z .+ (next_rz/rz).*p
            rz = next_rz
            completed = step
        end
        if step in checkpoints
            push!(output,(;step,completed,x=copy(x),relative_residual=norm(r)/max(norm(b),eps())))
        end
    end
    return output
end

"""PCG using diagonal smoothing plus the existing VEM factorization.

The additive preconditioner is SPD. No frequency information, new coarse
factorization, polynomial fit, or artificial patch boundary enters it.
"""
function hierarchical_pcg_estimates(mesh,fan,geometry,residuals,prepared;
        checkpoints=(4,8,16),remainder=delta->0.0,energy_goal=false,prolongation=nothing)
    free = geometry.free
    A = geometry.K[free,free]
    diagonal = geometry.diagonal[free]
    Pfull = prolongation === nothing ? hierarchical_prolongation(mesh,fan,geometry) : prolongation
    P = Pfull[free,prepared.free]
    precondition(r) = r./diagonal + P*(prepared.factor\(transpose(P)*r))
    us = hierarchical_pcg_checkpoints(A,residuals.bu[free],precondition;checkpoints)
    zs = hierarchical_pcg_checkpoints(A,residuals.bz[free],precondition;checkpoints)
    results = NamedTuple[]
    for (su,sz) in zip(us,zs)
        du,dz = zeros(size(geometry.K,1)),zeros(size(geometry.K,1))
        du[free],dz[free] = su.x,sz.x
        correction = remainder(du)
        symmetric = hierarchical_readout(residuals,du,dz)
        variational = residuals.consistency+dot(residuals.bu,dz)+dot(residuals.bz,du)-dot(su.x,A*sz.x)
        energy = residuals.primal_work+dot(residuals.bu,du)-dot(su.x,A*su.x)/2
        push!(results,(;method=:pcg_symmetric,steps=su.step,estimate=symmetric+correction,
            raw_estimate=symmetric,quadratic_correction=correction,
            primal_residual=su.relative_residual,dual_residual=sz.relative_residual))
        push!(results,(;method=:pcg_variational,steps=su.step,
            estimate=energy_goal ? energy : variational+correction,
            raw_estimate=energy_goal ? energy : variational,
            quadratic_correction=energy_goal ? 0.0 : correction,
            primal_residual=su.relative_residual,dual_residual=sz.relative_residual))
    end
    return results
end

function hierarchical_sweep!(x,residual,K,diagonal,indices)
    for order in (indices,reverse(indices))
        for j in order
            increment = residual[j]/diagonal[j]
            x[j] += increment
            for p in nzrange(K,j)
                residual[K.rowval[p]] -= K.nzval[p]*increment
            end
        end
    end
    return x
end

function hierarchical_readout(residuals,du,dz)
    return residuals.consistency+(dot(residuals.bu,dz)+dot(residuals.bz,du))/2
end

"""Half of J''[delta,delta] for the lab's goals of degree at most two.

Residual corrections reconstruct the adjoint linearized at U. The symmetric
readout alone estimates J'(U)[e], so quadratic goals require this term. For
linear goals it is zero; for integral(u²) it is integral(delta²); for energy
it is a(delta,delta)/2. No exact solution or problem-specific frequency enters.
"""
function hierarchical_goal_remainder(fan,geometry,delta,goal;order=4)
    rule = tetra_rule(order)
    result = 0.0
    for tet in eachindex(fan.volumes)
        G = fan.gradients[tet]
        local_delta = delta[geometry.dofs[:,tet]]
        xyz = fan.points[:,fan.tet_nodes[:,tet]]*rule.barycentric
        for q in eachindex(rule.weights)
            N = view(rule.barycentric,:,q)
            value = dot(local_delta[1:4],N)
            gradient = G*local_delta[1:4]
            for (edge,(a,b)) in enumerate(HIERARCHICAL_TET_EDGES)
                value += 4N[a]*N[b]*local_delta[4+edge]
                gradient .+= (4local_delta[4+edge]).*(N[a].*G[:,b]+N[b].*G[:,a])
            end
            x = view(xyz,:,q)
            dv = goal.derivative_value(x,value)-goal.derivative_value(x,0.0)
            dg = goal.derivative_gradient(x,gradient).-goal.derivative_gradient(x,(0.0,0.0,0.0))
            result += fan.volumes[tet]*rule.weights[q]*(dv*value+dot(dg,gradient))/2
        end
    end
    return result
end

function hierarchical_result(method,steps,residuals,du,dz,remainder)
    raw = hierarchical_readout(residuals,du,dz)
    correction = remainder(du)
    return (;method,steps,estimate=raw+correction,raw_estimate=raw,quadratic_correction=correction)
end

"""Diagonal corrections and progressively coupled controls for the same basis.

The all-free basis includes the fan-P1 residual omitted by edge-only enrichment.
The optional coarse solve is the fan-P1 operator, not the VEM operator. It is
factored once and shared between primal and dual residuals and all cycles.
"""
function hierarchical_estimates(geometry,residuals;sweeps=(1,2,4),coarse=true,direct=false,
        remainder=delta->0.0)
    K,d = geometry.K,geometry.diagonal
    results = NamedTuple[]
    for (name,free) in ((:edge_diagonal,geometry.free_edges),(:all_diagonal,geometry.free))
        du,dz = zeros(length(d)),zeros(length(d))
        du[free] .= residuals.bu[free]./d[free]
        dz[free] .= residuals.bz[free]./d[free]
        push!(results,hierarchical_result(name,0,residuals,du,dz,remainder))
    end
    for with_coarse in (false,true)
        with_coarse && !coarse && continue
        free = geometry.free
        coarse_dofs = filter(<=(geometry.vertex_count),free)
        factor = with_coarse ? cholesky(Symmetric(K[coarse_dofs,coarse_dofs])) : nothing
        du,dz = zeros(length(d)),zeros(length(d))
        ru,rz = copy(residuals.bu),copy(residuals.bz)
        for step in 1:maximum(sweeps)
            hierarchical_sweep!(du,ru,K,d,free)
            hierarchical_sweep!(dz,rz,K,d,free)
            if with_coarse
                cu,cz = factor\ru[coarse_dofs],factor\rz[coarse_dofs]
                du[coarse_dofs] .+= cu
                dz[coarse_dofs] .+= cz
                ru .-= K[:,coarse_dofs]*cu
                rz .-= K[:,coarse_dofs]*cz
            end
            if step in sweeps
                method = with_coarse ? :coarse_sweeps : :sweeps
                push!(results,hierarchical_result(method,step,residuals,du,dz,remainder))
            end
        end
    end
    if direct
        free = geometry.free
        factor = cholesky(Symmetric(K[free,free]))
        du,dz = zeros(length(d)),zeros(length(d))
        du[free] .= factor\residuals.bu[free]
        dz[free] .= factor\residuals.bz[free]
        push!(results,hierarchical_result(:direct,0,residuals,du,dz,remainder))
    end
    return results
end
