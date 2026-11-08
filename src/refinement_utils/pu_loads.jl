using StaticArrays, LinearAlgebra
isdefined(@__MODULE__,:pu3d_body_load!) || include("../body_loads.jl")

"""Physical load and essential-boundary entities retained for PU reconstruction."""
struct PU3DBoundaryData
    records::Vector{Tuple{Symbol,String,Function}}
    dirichlet::Vector{Tuple{Symbol,String,SVector{3,Bool}}}
end
PU3DBoundaryData(records=Tuple{Symbol,String,Function}[]) =
    PU3DBoundaryData(records,Tuple{Symbol,String,SVector{3,Bool}}[])

function pu3d_record_dirichlet!(data,kind,name,components)
    isnothing(data) && return nothing
    mask = SVector{3,Bool}(ntuple(i -> i in components,3))
    push!(data.dirichlet,(kind,String(name),mask))
    return nothing
end

function pu3d_add_dirichlet!(data,ch,dh,faces,name,f;c_dofs=SA[1,2,3])
    add_dirichlet_bc!(ch,dh,faces,name,f;c_dofs)
    pu3d_record_dirichlet!(data,:face,name,c_dofs)
    return nothing
end

function pu3d_edge_dirichlet!(ch,dh,name,f,components)
    nodes = Set{Int}()
    for edge in ch.mesh.edge_sets[name]
        union!(nodes,ch.mesh.topo.connectivity[1,2][edge])
    end
    for node in nodes
        haskey(dh.dof_mapping,node) || continue
        values = f(ch.mesh.nodes[node].coords)
        length(values) == length(components) || throw(DimensionMismatch("Prescribed edge displacement"))
        dofs = dh.dof_mapping[node]
        for (k,component) in enumerate(components)
            ch.d_bcs[dofs[component]] = values[k]
        end
    end
    return nothing
end

function pu3d_add_dirichlet!(data,ch,dh,name,f;c_dofs=SA[1,2,3])
    if haskey(ch.mesh.node_sets,name)
        add_dirichlet_bc!(ch,dh,name,f;c_dofs)
        pu3d_record_dirichlet!(data,:node,name,c_dofs)
    elseif haskey(ch.mesh.edge_sets,name)
        pu3d_edge_dirichlet!(ch,dh,name,f,c_dofs)
        pu3d_record_dirichlet!(data,:edge,name,c_dofs)
    else
        throw(ArgumentError("Prescribed displacement set '$name' is neither a node nor an edge set"))
    end
    return nothing
end

"""Auxiliary constraints follow prescribed entities, never endpoint coincidence.

`edge_fixed` keys are pairs of compact original PU vertex indices. A prescribed
face fixes its center and perimeter; a point support fixes no auxiliary node.
"""
function pu3d_auxiliary_constraints(g,cv,boundary_data)
    free = SVector(false,false,false)
    face_fixed = fill(free,length(g.face_ids))
    edge_fixed = Dict{Tuple{Int,Int},SVector{3,Bool}}()
    isempty(boundary_data.dirichlet) && return face_fixed,edge_fixed
    face_index = Dict(id => i for (i,id) in enumerate(g.face_ids))
    vertex_index = any(record -> record[1] == :edge,boundary_data.dirichlet) ?
        Dict(id => i for (i,id) in enumerate(g.vertex_ids)) : nothing
    for (kind,name,mask) in boundary_data.dirichlet
        if kind == :face
            for id in cv.mesh.face_sets[name]
                face = get(face_index,id,0)
                face == 0 && continue
                g.face_cells[face][2] == 0 || throw(ArgumentError("Prescribed PU face $id is interior"))
                face_fixed[face] = face_fixed[face] .| mask
                nodes = g.face_nodes[face]
                for i in eachindex(nodes)
                    key = minmax(nodes[i],nodes[mod1(i+1,length(nodes))])
                    edge_fixed[key] = get(edge_fixed,key,free) .| mask
                end
            end
        elseif kind == :edge
            for id in cv.mesh.edge_sets[name]
                a,b = cv.mesh.topo.connectivity[1,2][id]
                a,b = get(vertex_index,a,0),get(vertex_index,b,0)
                (a == 0 || b == 0) && continue
                key = minmax(a,b)
                edge_fixed[key] = get(edge_fixed,key,free) .| mask
            end
        end
    end
    return face_fixed,edge_fixed
end

function pu3d_add_neumann!(data,ch,dh,faces,name,f)
    add_neumann_bc!(ch,dh,faces,name,f)
    isnothing(data) || push!(data.records,(:face,String(name),f))
    return nothing
end
function pu3d_add_neumann!(data,ch,dh,name,f)
    add_neumann_bc!(ch,dh,name,f)
    isnothing(data) || push!(data.records,(:edge,String(name),f))
    return nothing
end

struct PU3DTetraLoad
    force::SVector{30,Float64}
    weighted::SMatrix{30,4,Float64,120}
end
struct PU3DLoads
    slots::Vector{Int}
    tetra::Vector{PU3DTetraLoad}
    discrete::Vector{Vector{Tuple{Int,Float64}}}
end

# Degree-four simplex rules. Weights sum to one, so measures are physical.
const PU3D_TRIANGLE_POINTS = (
    SA[0.445948490915965,0.445948490915965,0.108103018168070],
    SA[0.445948490915965,0.108103018168070,0.445948490915965],
    SA[0.108103018168070,0.445948490915965,0.445948490915965],
    SA[0.091576213509771,0.091576213509771,0.816847572980458],
    SA[0.091576213509771,0.816847572980458,0.091576213509771],
    SA[0.816847572980458,0.091576213509771,0.091576213509771])
const PU3D_TRIANGLE_WEIGHTS = (0.223381589678011,0.223381589678011,0.223381589678011,
    0.109951743655322,0.109951743655322,0.109951743655322)

@inline function pu3d_accumulate_load!(force,weighted,N,value,measure)
    phi = pu3d_p2_values(N)
    @inbounds for i in 1:10,c in 1:3
        row = 3i-3+c
        v = measure*phi[i]*value[c]
        force[row] += v
        for hat in 1:4
            weighted[row,hat] += N[hat]*v
        end
    end
    return nothing
end

function pu3d_load_scratch!(scratch,index)
    get!(scratch,index) do
        (zeros(MVector{30,Float64}),zeros(MMatrix{30,4,Float64,120}))
    end
end

function pu3d_face_load!(scratch,index,t,g,f)
    a,b,c = (g.node_points[t.nodes[k]] for k in 1:3)
    area = norm(cross(b-a,c-a))/2
    for (N3,w) in zip(PU3D_TRIANGLE_POINTS,PU3D_TRIANGLE_WEIGHTS)
        value = SVector{3,Float64}(f(N3[1]*a+N3[2]*b+N3[3]*c))
        iszero(value) && continue
        force,weighted = pu3d_load_scratch!(scratch,index)
        pu3d_accumulate_load!(force,weighted,SA[N3[1],N3[2],N3[3],0.0],value,area*w)
    end
    return nothing
end

function pu3d_edge_load!(scratch,indices,g,f)
    points,weights = Ju3VEM.VEMGeo.get_gauss_legendre(4)
    for index in indices
        t = g.tets[index]
        a,b = g.node_points[t.nodes[1]],g.node_points[t.nodes[2]]
        half_length = norm(b-a)/(2length(indices))
        for (s,w) in zip(points,weights)
            x = (1-s)/2
            value = SVector{3,Float64}(f(x*a+(1-x)*b))
            iszero(value) && continue
            force,weighted = pu3d_load_scratch!(scratch,index)
            pu3d_accumulate_load!(force,weighted,SA[x,1-x,0.0,0.0],value,half_length*w)
        end
    end
    return nothing
end

# Use the mechanics' actual face projection and quadrature, which differ from
# the continuous P1 fan trace for non-affine data and variable traction.
function pu3d_discrete_face_load!(dest,cv,fd,f)
    ids = fd.face_node_ids.v.args[1]
    nodes2d = [Ju3VEM.VEMGeo.project_to_2d_abs(cv.mesh[id],fd.dΩ.plane) for id in ids]
    moments = Ju3VEM.VEMUtils.function_integration2d(nodes2d;order=1) do x,y
        point2d = SA[x,y]
        scaled = (point2d-fd.dΩ.bc)/fd.dΩ.hf
        point = Ju3VEM.VEMGeo.project_to_3d(point2d,fd.dΩ.plane)
        return SVector{3,Float64}(f(point))*transpose(SA[1.0,scaled[1],scaled[2]])
    end
    for (j,id) in enumerate(ids)
        dofs = cv.dh.dof_mapping[id]
        for component in 1:3
            value = sum(moments[component,k]*fd.ΠsL2[k,j] for k in 1:3)
            iszero(value) || push!(dest,(dofs[component],value))
        end
    end
    return nothing
end

function pu3d_load_incidence(g,needs_edges)
    face_tets = [Int[] for _ in g.face_ids]
    boundary_edges = Dict{Tuple{Int,Int},Vector{Int}}()
    for (index,t) in enumerate(g.tets)
        push!(face_tets[t.face],index)
        if needs_edges && g.face_cells[t.face][2] == 0
            key = minmax(g.vertex_ids[t.nodes[1]],g.vertex_ids[t.nodes[2]])
            push!(get!(() -> Int[],boundary_edges,key),index)
        end
    end
    return face_tets,boundary_edges
end

function pu3d_face_set_load!(scratch,discrete,g,cv,face_index,face_tets,name,f::F) where F
    for id in cv.mesh.face_sets[name]
        face = get(face_index,id,0)
        face == 0 && continue
        g.face_cells[face][2] == 0 || throw(ArgumentError("PU traction face $id is interior"))
        cell = g.face_cells[face][1]
        for index in face_tets[face]
            pu3d_face_load!(scratch,index,g.tets[index],g,f)
        end
        pu3d_discrete_face_load!(discrete[cell],cv,cv.facedata_col[id],f)
    end
    return nothing
end

function pu3d_edge_set_load!(scratch,discrete,g,cv,boundary_edges,name,f::F) where F
    for id in cv.mesh.edge_sets[name]
        node1,node2 = cv.mesh.topo.connectivity[1,2][id]
        indices = get(boundary_edges,minmax(node1,node2),nothing)
        isnothing(indices) && throw(ArgumentError("PU line load edge $id has no boundary tetrahedron"))
        pu3d_edge_load!(scratch,indices,g,f)
        half_length = norm(cv.mesh[node2].coords-cv.mesh[node1].coords)/(2length(indices))
        for index in indices,node in (node1,node2)
            value = f(cv.mesh[node].coords)*half_length
            for component in 1:3
                iszero(value[component]) || push!(discrete[g.tets[index].cell],
                    (cv.dh.dof_mapping[node][component],value[component]))
            end
        end
    end
    return nothing
end

function pu3d_boundary_loads!(scratch,discrete,g,cv,boundary_data)
    isempty(boundary_data.records) && return nothing
    needs_edges = any(record -> record[1] == :edge,boundary_data.records)
    face_tets,boundary_edges = pu3d_load_incidence(g,needs_edges)
    face_index = Dict(id => i for (i,id) in enumerate(g.face_ids))
    for (kind,name,f) in boundary_data.records
        if kind == :face
            pu3d_face_set_load!(scratch,discrete,g,cv,face_index,face_tets,name,f)
        else
            pu3d_edge_set_load!(scratch,discrete,g,cv,boundary_edges,name,f)
        end
    end
    return nothing
end

function pu3d_check_boundary_loads(discrete,ch)
    # Reject untracked nodal loads: reconstructing a physical traction from its
    # assembled vector is not unique and would invalidate the PU residual.
    assembled = Dict{Int,Float64}()
    for entries in discrete,(dof,value) in entries
        assembled[dof] = get(assembled,dof,0.0)+value
    end
    scale = max(1.0,maximum(abs,values(ch.n_bcs);init=0.0))
    all(abs(get(assembled,dof,0.0)-value) <= 1e-10*scale for (dof,value) in ch.n_bcs) &&
        all(abs(get(ch.n_bcs,dof,0.0)-value) <= 1e-10*scale for (dof,value) in assembled) ||
        throw(ArgumentError("PU-DWR needs physical load definitions; use pu3d_add_neumann! with PU3DBoundaryData"))
    return nothing
end

function pu3d_body_loads!(scratch,discrete,g,cv,eldata,rhs_fun::F) where F
    pu3d_body_iszero(rhs_fun) && return nothing
    moments = fill(zero(SMatrix{3,4,Float64,12}),length(g.element_ids))
    for (index,t) in enumerate(g.tets)
        data = eldata[g.element_ids[t.cell]]
        a,b,c,d = (g.node_points[i] for i in t.nodes)
        for (N,w) in zip(PU3D_TETRA_POINTS,PU3D_TETRA_WEIGHTS)
            point = N[1]*a+N[2]*b+N[3]*c+N[4]*d
            value = SVector{3,Float64}(rhs_fun(point))
            iszero(value) && continue
            force,weighted = pu3d_load_scratch!(scratch,index)
            pu3d_accumulate_load!(force,weighted,N,value,t.volume*w)
            moments[t.cell] += pu3d_body_moment(point,value,data.bc,data.hvol,t.volume*w)
        end
    end
    for (cell,id) in enumerate(g.element_ids)
        data = eldata[id]
        for (j,node) in enumerate(data.node_ids),component in 1:3
            value = sum(moments[cell][component,k]*data.proj_s[k,j] for k in 1:4)
            iszero(value) || push!(discrete[cell],(cv.dh.dof_mapping[node][component],value))
        end
    end
    return nothing
end

function pu3d_materialize_loads(g,scratch,discrete)
    slots = zeros(Int,length(g.tets))
    tetra = PU3DTetraLoad[]
    for index in sort!(collect(keys(scratch)))
        force,weighted = scratch[index]
        push!(tetra,PU3DTetraLoad(SVector(force),SMatrix(weighted)))
        slots[index] = length(tetra)
    end
    return PU3DLoads(slots,tetra,discrete)
end

function pu3d_build_loads(g,cv,ch,boundary_data,rhs_fun,eldata)
    scratch = Dict{Int,Tuple{MVector{30,Float64},MMatrix{30,4,Float64,120}}}()
    discrete = [Tuple{Int,Float64}[] for _ in g.element_ids]
    pu3d_boundary_loads!(scratch,discrete,g,cv,boundary_data)
    pu3d_check_boundary_loads(discrete,ch)
    pu3d_body_loads!(scratch,discrete,g,cv,eldata,rhs_fun)
    return pu3d_materialize_loads(g,scratch,discrete)
end

@inline function pu3d_load_entry(loads,tetindex,row,hat)
    slot = loads.slots[tetindex]
    slot == 0 && return 0.0,0.0
    load = loads.tetra[slot]
    return load.force[row],load.weighted[row,hat]
end

function pu3d_continuous_load_work(loads,tetindex,t,values)
    slot = loads.slots[tetindex]
    slot == 0 && return 0.0
    force = loads.tetra[slot].force
    work = 0.0
    @inbounds for i in 1:10
        value = if i <= 4
            values[t.nodes[i]]
        else
            a,b = PU3D_P2_EDGES[i-4]
            (values[t.nodes[a]]+values[t.nodes[b]])/2
        end
        for component in 1:3
            work += force[3i-3+component]*value[component]
        end
    end
    return work
end

function pu3d_discrete_load_work(loads,cell,u)
    value = 0.0
    @inbounds for (dof,force) in loads.discrete[cell]
        value += force*u[dof]
    end
    return value
end
