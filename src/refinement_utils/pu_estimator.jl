"""Borrowed output; valid until the cache is evaluated again."""
struct PU3DEstimate
    indicators::Vector{Float64}
    nodal::Vector{Float64}
    element_correction::Vector{Float64}
    signed::Float64
    magnitude::Float64
end
Base.values(e::PU3DEstimate) = Iterators.flatten((e.nodal,e.element_correction))

function pu3d_reconstruct!(c,u,eldata)
    g = c.geometry
    nv,nf = length(g.vertex_ids),length(g.face_ids)
    for i in eachindex(c.vertex_dofs)
        dofs = c.vertex_dofs[i]
        c.values[i] = SA[u[dofs[1]],u[dofs[2]],u[dofs[3]]]
    end
    for (face,ids) in enumerate(g.face_nodes)
        value = zero(SVector{3,Float64})
        for j in eachindex(ids)
            value += c.face_weights[face][j]*c.values[ids[j]]
        end
        c.values[nv+face] = value
    end
    for (cell,id) in enumerate(g.element_ids)
        data = eldata[id]
        value = zero(SVector{3,Float64})
        for (i,node) in enumerate(g.cell_nodes[cell])
            value += 4data.proj_s[1,i]*c.values[node]
        end
        for index in g.cell_offsets[cell]:g.cell_offsets[cell+1]-1
            t = g.tets[index]
            value -= (t.volume/data.volume)*(c.values[t.nodes[1]]+
                c.values[t.nodes[2]]+c.values[t.nodes[3]])
        end
        c.values[nv+nf+cell] = value
    end
    return nothing
end

@inline function pu3d_projected_gradient(c,cell,data)
    gradient = zero(SMatrix{3,3,Float64,9})
    for (i,node) in enumerate(c.geometry.cell_nodes[cell])
        gradient += c.values[node]*transpose(SA[data.proj_s[2,i],data.proj_s[3,i],data.proj_s[4,i]])
    end
    return gradient/data.hvol
end

function pu3d_prepare!(c,u,states,eldata,pars,density,q)
    isfinite(q) && q >= 0 || throw(ArgumentError("PU marking exponent must be finite and nonnegative"))
    c.lambda == pars.λ && c.mu == pars.μ || throw(ArgumentError("Rebuild PU cache after material changes"))
    length(density) == length(c.geometry.element_ids) || throw(DimensionMismatch("PU density size changed; rebuild the cache"))
    all(x -> isfinite(x) && 0 <= x <= 1,density) || throw(ArgumentError("Invalid physical PU density"))
    pu3d_reconstruct!(c,u,eldata)
    fill!(c.nodal,0.0)
    for (cell,id) in enumerate(c.geometry.element_ids)
        get_state_id(states,id) == cell || throw(ArgumentError("PU state ordering changed; rebuild the cache"))
        data = eldata[id]
        scale = density[cell]^3
        c.scale[cell] = scale
        c.marking_weight[cell] = data.volume*density[cell]^q
        gradient = pu3d_projected_gradient(c,cell,data)
        strain = (gradient+transpose(gradient))/2
        c.projected_energy[cell] = scale*dot(strain,pu3d_stress(strain,c.lambda,c.mu))
        stabilization = 0.0
        gamma = (9c.lambda+6c.mu)/4*scale*data.hvol
        for (i,node) in enumerate(c.geometry.cell_nodes[cell])
            difference = c.values[node]
            for (j,other) in enumerate(c.geometry.cell_nodes[cell])
                difference -= data.proj[i,j]*c.values[other]
            end
            value = gamma*dot(difference,difference)
            stabilization += value
            c.nodal[node] -= value/2
        end
        # This cell correction remains separate when magnitudes are formed.
        c.element_correction[cell] = data.volume*c.projected_energy[cell]+stabilization-
            pu3d_discrete_load_work(c.loads,cell,u)
    end
    for (index,t) in enumerate(c.geometry.tets)
        strain = zero(SMatrix{3,3,Float64,9})
        for i in 1:3
            strain += pu3d_strain(c.values[t.nodes[i]]-c.values[t.nodes[4]],t.gradients[i])
        end
        stress = c.scale[t.cell]*pu3d_stress(strain,c.lambda,c.mu)
        energy = dot(stress,strain)
        c.stress[index],c.energy[index] = stress,energy
        c.element_correction[t.cell] += pu3d_continuous_load_work(c.loads,index,t,c.values)-t.volume*energy
    end
    for w in c.workspaces
        w.max_used = 0
    end
    return nothing
end

"""Exact mean gradient outer product for one pair of scalar P2 shapes."""
@inline function pu3d_gradient_moment(t,i,j)
    g = t.gradients
    if i <= 4 && j <= 4
        return ((4(i == j)-1)/5)*(g[i]*transpose(g[j]))
    elseif i <= 4
        a,b = PU3D_P2_EDGES[j-4]
        return g[i]*transpose(((4(i == a)-1)*g[b]+(4(i == b)-1)*g[a])/5)
    elseif j <= 4
        a,b = PU3D_P2_EDGES[i-4]
        return (((4(j == a)-1)*g[b]+(4(j == b)-1)*g[a])/5)*transpose(g[j])
    end
    a,b = PU3D_P2_EDGES[i-4]
    k,l = PU3D_P2_EDGES[j-4]
    return 0.8*(g[b]*transpose((1+(a == k))*g[l]+(1+(a == l))*g[k])+
                g[a]*transpose((1+(b == k))*g[l]+(1+(b == l))*g[k]))
end

@inline function pu3d_elastic_block(t,i,j,lambda,mu)
    H = pu3d_gradient_moment(t,i,j)
    return lambda*H+mu*transpose(H)+(mu*tr(H))*one(H)
end

"""Scatter one unordered scalar-shape block into the active lower triangle."""
@inline function pu3d_scatter_block!(matrix,block,rows,columns,same_shape)
    @inbounds for column_component in 1:3
        column = columns[column_component]
        column == 0 && continue
        first_component = same_shape ? column_component : 1
        for row_component in first_component:3
            row = rows[row_component]
            row == 0 && continue
            if row >= column
                matrix[row,column] += block[row_component,column_component]
            else
                matrix[column,row] += block[row_component,column_component]
            end
        end
    end
    return nothing
end

function pu3d_add_tetra!(w,c,index,hat,map)
    t = c.geometry.tets[index]
    stress = c.stress[index]
    measure = c.scale[t.cell]*t.volume
    # A typical interior PU tetrahedron has four active scalar shapes. Compact
    # that set first, then form each 3x3 elastic block once, not per component.
    active = MVector{10,Int}(undef)
    count = 0
    @inbounds for i in 1:10
        rows = w.dofs[map[i]]
        iszero(rows) && continue
        count += 1
        active[count] = i
    end
    @inbounds for local_i in 1:count
        i = active[local_i]
        rows = w.dofs[map[i]]
        gradient,weighted_gradient = pu3d_internal_gradients(t,i,hat)
        internal = t.volume*(stress*gradient)
        weighted_internal = t.volume*(stress*weighted_gradient)
        for component in 1:3
            row = rows[component]
            row == 0 && continue
            localrow = 3i-3+component
            force,weighted_force = pu3d_load_entry(c.loads,index,localrow,hat)
            w.rhs[row] += force-internal[component]
            w.weighted[row] += weighted_force-weighted_internal[component]
        end
        for local_j in 1:local_i
            j = active[local_j]
            columns = w.dofs[map[j]]
            block = measure*pu3d_elastic_block(t,i,j,c.lambda,c.mu)
            pu3d_scatter_block!(w.matrix,block,rows,columns,i == j)
        end
    end
    return t.volume*(c.energy[index]-c.projected_energy[t.cell])/8
end

@inline pu3d_observe(::Nothing,w,c,node,index,hat,map) = nothing
@inline function pu3d_observe(observer,w,c,node,index,hat,map)
    delta = SVector{30,Float64}(ntuple(Val(30)) do i
        slot = w.dofs[map[div(i-1,3)+1]][mod1(i,3)]
        return slot == 0 ? 0.0 : w.rhs[slot]
    end)
    observer(node,index,hat,delta)
    return nothing
end

function pu3d_patch!(w,c,node,observer)
    incidence = pu3d_patch_number!(w,c,node)
    contribution = c.nodal[node]
    for (localtet,index) in enumerate(incidence)
        ti,hat = c.incidence[index]
        contribution += pu3d_add_tetra!(w,c,ti,hat,w.maps[localtet])
    end
    try
        pu3d_solve!(w.matrix,w.rhs,w.ndofs)
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

"""Distribute complete node magnitudes once, then add separate cell corrections.

Incidences are cell-ordered, so distinct adjacent cells need no hash/set lookup.
An interior face center distributes to two cells; a cell center to one.
"""
function pu3d_collect!(c,states)
    fill!(c.indicators,0.0)
    for node in eachindex(c.nodal)
        total,volume = 0.0,0.0
        previous = 0
        for index in c.offsets[node]:c.offsets[node+1]-1
            cell = c.geometry.tets[c.incidence[index][1]].cell
            cell == previous && continue
            previous = cell
            total += c.marking_weight[cell]
            volume += states.area_vec[cell]
        end
        previous = 0
        for index in c.offsets[node]:c.offsets[node+1]-1
            cell = c.geometry.tets[c.incidence[index][1]].cell
            cell == previous && continue
            previous = cell
            fraction = iszero(total) ? states.area_vec[cell]/volume : c.marking_weight[cell]/total
            c.indicators[cell] += abs(c.nodal[node])*fraction
        end
    end
    for cell in eachindex(c.indicators)
        c.indicators[cell] += abs(c.element_correction[cell])
    end
    return PU3DEstimate(c.indicators,c.nodal,c.element_correction,
        sum(c.nodal)+sum(c.element_correction),sum(abs,c.nodal)+sum(abs,c.element_correction))
end

"""Local P2 elasticity reconstruction and PU-DWR strain-energy localization.

`density` must be the physical density used to solve `u`, before any design
update. Geometry/loads/constraints stay fixed until the cache is rebuilt.
The optional observer forces serial execution and exposes local P2 corrections.
"""
function estimate_pu_dwr!(c,u,states,eldata,pars;q::Real=1.0,density=states.χ_vec,observer=nothing)
    pu3d_prepare!(c,u,states,eldata,pars,density,q)
    workers = length(c.workspaces)
    if observer !== nothing || workers == 1
        for node in eachindex(c.nodal)
            pu3d_patch!(first(c.workspaces),c,node,observer)
        end
    else
        Threads.@threads :static for worker in 1:workers
            # Mix vertex, face and cell stars so patch size does not correlate
            # with worker ownership. Collection remains deterministic.
            for node in worker:workers:length(c.nodal)
                pu3d_patch!(c.workspaces[worker],c,node,nothing)
            end
        end
    end
    return pu3d_collect!(c,states)
end

pu3d_cache_statistics(c) = (;node_count=length(c.nodal),tetrahedron_count=length(c.geometry.tets),
    cell_count=length(c.indicators),max_free_dofs=maximum(w.max_used for w in c.workspaces))
