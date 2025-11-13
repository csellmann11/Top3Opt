using Ju3VEM
using Ju3VEM.FixedSizeArrays
using StaticArrays
using LinearAlgebra
using SparseArrays
using Chairmarks
import Ju3VEM.FR as FR
using Statistics
using Random
using Bumper
using TimerOutputs
using OrderedCollections
using Dates
using Bumper
using JLD2
using Infiltrator
using Dates: today
using Pardiso



include("../src/mat_states.jl")
include("../src/laplace_operator.jl")
include("../src/compute_displacement.jl")
include("../src/bisection.jl")
include("../src/mesh_processing_utils.jl")
include("../src/neighbor_search.jl")


const K = 1
const U = 3

function MBB_rhs(x)
    SA[0.0,0.0,0.0]
end

n = div(2,2)*2
n = 20

mesh = create_rectangular_mesh(
    n,n,n,
    1.0,1.0,1.0,StandardEl{K}
)

mesh2d = create_voronoi_mesh(
    (0.0,0.0),
    (1.0,1.0),
    n,n,StandardEl{K}
)

mesh = extrude_to_3d(n,mesh2d,1.0);

# for i in 1:2
#     for element in RootIterator{4}(mesh.topo)
#         _refine!(element,mesh.topo)
#     end
# end
# mesh = Mesh(mesh.topo,StandardEl{1}())


dx_cell = 1/n 
h_cell = find_maximal_cell_diameter(mesh.topo)

cv = CellValues{U}(mesh) 
ρ_init = 0.3
states = DesignVarInfo{U}(cv,ρ_init)

E = 210.e03; ν = 0.33
λ,μ = E_ν_to_lame(E,ν)
χ = 0.3
mat_law  = Helmholtz{3,3}(Ψlin_totopt,(λ,μ,χ))
mat_pars = (λ,μ)

χmin = 1e-03
η0   = 15.0 
β0   = 2*h_cell^2 * 0.5
ρ_init = 0.3 
sim_pars = SimPars(mat_law,λ,μ,χmin,η0,β0,ρ_init)


function find_distance_to_boundary(
    vol_id, 
    topo, 
    x0::SVector{3,Float64}, 
    d::SVector{3,Float64}
)
    min_rel_dist = typemax(Float64)

    face_ids = get_volume_area_ids(topo, vol_id)
    
    @inbounds for face_id in face_ids
        node_ids = get_area_node_ids(topo, face_id)
        
        length(node_ids) < 3 && continue
        
        p1 = topo.nodes[node_ids[1]]
        p2 = topo.nodes[node_ids[2]]
        p3 = topo.nodes[node_ids[3]]
        
        e1 = p2 - p1
        e2 = p3 - p1
        n = e1 × e2
        
        denom = n ⋅ d
        abs(denom) < eps(Float64) && continue
        
        t = (n ⋅ (p1 - x0)) / denom
        
        # Accept both positive AND negative t, take minimum absolute value
        abs_t = abs(t)
        if abs_t > eps(Float64) && abs_t < min_rel_dist
            min_rel_dist = abs_t
        end
    end
    
    return min_rel_dist
end


x0 = states.x_vec[1]
x1 = states.x_vec[14]
d = x1 - x0
min_rel_dist = find_distance_to_boundary(1,mesh.topo,x0,d)

@b find_distance_to_boundary(1,$mesh.topo,$x0,$d)
@show min_rel_dist
min_dist = min_rel_dist * norm(d)

x0 = states.x_vec[1]
x1 = states.x_vec[2]
d = x1 - x0
min_dist = find_distance_to_boundary(1,mesh.topo,x0,d)
@show min_dist



@time state_neights_col, b_face_id_to_state_id = create_neigh2_list(states,cv);
@time laplace_operator = compute_laplace_operator_mat(
                  cv.mesh.topo,state_neights_col,b_face_id_to_state_id,states,sim_pars)

state_neights_col[5] |> display 
# [b_face_id_to_state_id[key] for key in state_neights_col[37][4:11]]


f = x -> cos(pi*x[1])*cos(pi*x[2])*cos(pi*x[3])#-1/3*x[1]^3 + 1/2*x[1]^2

import Ju3VEM.FR.ForwardDiff as FD

@show FD.hessian(f,SA[1.0,1.0,1.0]) |> tr
@show FD.hessian(f,SA[0.0,0.0,0.0]) |> tr
@show FD.hessian(f,SA[0.5,0.0,0.0]) |> tr

for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    xbc = states.x_vec[sid] 
    states.χ_vec[sid] = f(xbc)
end


# ns = state_neights_col[14385]
# states.x_vec[ns] |> display 
# states.χ_vec[ns] |> display 
# states.x_vec[14385] |> display 


nels = length(RootIterator{4}(mesh.topo))
vals = Vector{Float64}(undef,nels)
for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    xbc = states.x_vec[sid] 
    vals[sid] = f(xbc)
end




lap = laplace_operator * states.χ_vec
for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    states.χ_vec[sid] = lap[sid]
end

@show maximum(states.χ_vec)
write_vtk(cv.mesh.topo,"Results/vtk/lap_test";cell_data_col = (states.χ_vec,))