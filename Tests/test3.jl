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
n = 4
mesh = create_rectangular_mesh(
    n,n,n,
    1.0,1.0,1.0,StandardEl{K}
)

mesh2d = create_voronoi_mesh(
    (0.0,0.0),
    (3.0,1.0),
    3n,n,StandardEl{K}
)

n_orig_nodes = length(mesh2d.topo.nodes)
println("n_orig_nodes: $n_orig_nodes")
topo = mesh2d.topo
topo = remove_short_edges(mesh2d.topo)
topo = remove_short_edges(topo)
n_new_nodes = length(topo.nodes)
println("n_new_nodes: $n_new_nodes")
n_active_nodes = count(x -> is_active(x,topo),topo.nodes)
println("n_active_nodes: $n_active_nodes")
# topo = remove_short_edges(topo)
mesh2d = Mesh(topo,StandardEl{1}())




mesh = extrude_to_3d(1,mesh2d,0.1);
n_active_nodes = count(x -> is_active(x,mesh.topo),mesh.topo.nodes)
cv = CellValues{1}(mesh);

k = get_sparsity_pattern(cv)
dk = diag(k)

length(keys(cv.dh.dof_mapping))

write_vtk(mesh.topo,"Results/vtk/lap_test")

# mesh2d = remove_short_edges(mesh2d)
# topo2d = Ju3VEM.VEMGeo.remove_short_edges(mesh2d.topo,factor=1/8)
# mesh2d = Mesh(topo2d,StandardEl{1}())

# mesh = extrude_to_3d(n,mesh2d,1.0);
rng = MersenneTwister(42)
for i in 1:2
    for element in RootIterator{4}(mesh.topo)
        rand(rng) < 0.5 && Ju3VEM.VEMGeo._refine!(element,mesh.topo)
    end
end

# element = get_volumes(mesh.topo)[681]
# Ju3VEM.VEMGeo._refine!(element,mesh.topo)
mesh = Mesh(mesh.topo,StandardEl{1}())


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
                  cv.mesh.topo,state_neights_col,b_face_id_to_state_id,states)




f = x -> cos(pi*x[1])*cos(pi*x[2])*cos(pi*x[3])#-1/3*x[1]^3 + 1/2*x[1]^2
# f = x -> -1/3*x[1]^3 + 1/2*x[1]^2

import Ju3VEM.FR.ForwardDiff as FD


FD.hessian(f,SA[1.0,1.0,1.0]) |> tr |> display 

for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    xbc = states.x_vec[sid] 
    states.χ_vec[sid] = f(xbc)
end


states.x_vec[1] |> display 

@time lap = laplace_operator * states.χ_vec;


# @time lap = comp_lap(cv,states);


for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    states.χ_vec[sid] = lap[sid]
end

@show maximum(states.χ_vec)
write_vtk(cv.mesh.topo,"Results/vtk/lap_test";cell_data_col = (states.χ_vec,))


#  write_vtk(mesh.topo,"Results/vtk/lap_test")