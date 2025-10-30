using Ju3VEM
using Ju3VEM.FixedSizeArrays
using StaticArrays
using LinearAlgebra
using SparseArrays
using Chairmarks
import Ju3VEM.FR as FR
using IterativeSolvers
using AlgebraicMultigrid
using IncompleteLU

struct SimParameter
    mat_law::Helmholtz
    λ::Float64
    μ::Float64 
    
    χmin::Float64
    η0::Float64 
    β0::Float64
    ρ_init::Float64
    h_min::Float64
end

@inline function Ψlin_totopt(∇u::M,λ,μ,χ) where {M<:AbstractMatrix}
    ε = 1/2*(∇u + ∇u')
    W = λ/2 * tr(ε)^2 + μ*tr(ε*ε)
    return W*χ[1]^3
end


include("mat_states.jl")
include("laplace_operator.jl")
include("compute_displacement.jl")
include("bisection.jl")
include("mesh_processing_utils.jl")

include("laplace_opterator2.jl")
const K = 1
const U = 3

function MBB_rhs(x)
    SA[0.0,0.0,0.0]
end

n = div(2,2)*2
n= 20
# l_beam = 3.0
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


dx_cell = 1/n 
h_cell = find_maximal_cell_diameter(mesh.topo)

cv = CellValues{U}(mesh) 
ρ_init = 0.3
states = TopStates{U}(cv,ρ_init)

E = 210.e03; ν = 0.33
λ,μ = E_ν_to_lame(E,ν)
χ = 0.3
mat_law  = Helmholtz{3,3}(Ψlin_totopt,(λ,μ,χ))
mat_pars = (λ,μ)

χmin = 1e-03
η0   = 15.0 
β0   = 2*h_cell^2 * 0.5
ρ_init = 0.3 
sim_pars = SimParameter(mat_law,λ,μ,χmin,η0,β0,ρ_init,h_cell)


# @time el_neighs, face_to_vols = Ju3VEM.VEMGeo.create_element_neighbour_list_fast(mesh.topo;n_neighs_min = 18);

@time el_neighs, face_to_vols = create_neighbor_list(cv);

@time "laplace_operator" lap_operator = compute_laplace_operator_mat(
    mesh.topo,el_neighs,face_to_vols,states,sim_pars) 



f = x -> cos(pi*x[1])#-1/3*x[1]^3 + 1/2*x[1]^2

import Ju3VEM.FR.ForwardDiff as FD

@show FD.hessian(f,SA[1.0,1.0,1.0]) |> tr
@show FD.hessian(f,SA[0.0,0.0,0.0]) |> tr
@show FD.hessian(f,SA[0.5,0.0,0.0]) |> tr

for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    xbc = states.x_vec[sid] 
    states.χ_vec[sid] = f(xbc)
end





nels = length(RootIterator{4}(mesh.topo))
vals = Vector{Float64}(undef,nels + length(idx_coords))
for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    xbc = states.x_vec[sid] 
    vals[sid] = f(xbc)
end

for (idx,node) in enumerate(idx_coords)
    full_idx = idx + nels
    vals[full_idx] = f(node)
end


lap = lap_operator * states.χ_vec
for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    states.χ_vec[sid] = lap[sid]
end

@show maximum(states.χ_vec)
write_vtk(cv.mesh.topo,"vtk/MBB_beam_sym_$(n)";cell_data = states.χ_vec)