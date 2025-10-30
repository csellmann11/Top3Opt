using Ju3VEM
using Ju3VEM.FixedSizeArrays
using StaticArrays
using LinearAlgebra
using SparseArrays
using Chairmarks
import Ju3VEM.FR as FR
using IterativeSolvers
using AlgebraicMultigrid
using Random
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

n = div(5,2)*2
l_beam = 3.0
mesh = create_rectangular_mesh(
    3n,div(n,2),n,
    l_beam,0.5,1.0,StandardEl{K}
);


left  = (0.0, 0.0, 0.0)
right = (l_beam, 0.5, 1.0)
rng   = Random.MersenneTwister(1)
# seeds = [SA[3*rand(rng), rand(rng), rand(rng)] for _ in 1:4000]
# seeds = relax_voronoi3d_seeds(left, right, seeds; maxiters=15, move_tol=1e-7, step=0.8)
# mesh = create_voronoi_mesh_3d(left, right, seeds, StandardEl{K}; dedup_tol=1e-10)

# mesh2d = create_voronoi_mesh(
#     (0.0,0.0),
#     (l_beam,0.5),
#     3n,div(n,2),StandardEl{K}
# )

# mesh = extrude_to_3d(n,mesh2d,1.0);



dx_cell = 1/n 
# h_cell = norm(SA[dx_cell,dx_cell,dx_cell])
h_cell = find_maximal_cell_diameter(mesh.topo)


E = 210.e03; ν = 0.33
λ,μ = E_ν_to_lame(E,ν)
χ = 0.3
mat_law  = Helmholtz{3,3}(Ψlin_totopt,(λ,μ,χ))
mat_pars = (λ,μ)

χmin = 1e-03
η0   = 15.0 
β0   = 2*h_cell^2 
ρ_init = 0.3 
sim_pars = SimParameter(mat_law,λ,μ,χmin,η0,β0,ρ_init,h_cell)


# @time el_neighs, face_to_vols = Ju3VEM.VEMGeo.create_element_neighbour_list_fast(mesh.topo;n_neighs_min = 15);
cv = CellValues{U}(mesh) 

@time "create_neighbor_list" el_neighs, face_to_vols = create_neighbor_list(cv);



ρ_init = 0.3
states = TopStates{U}(cv,ρ_init)


@time "laplace_operator" laplace_operator = compute_laplace_operator_mat(
    mesh.topo,el_neighs,face_to_vols,states,sim_pars) 


add_face_set!(mesh,"symmetry_bc",x -> x[1] ≈ 0.0)
add_face_set!(mesh,"symmetry_bc_2",x -> x[2] ≈ 0.5)
add_node_set!(mesh,"roller_bearing",x -> x[1] ≈ 3.0 && x[3] ≈ 0.0)
add_face_set!(mesh,"middle_traction",x -> (0 ≤ x[1] ≤ 0.4) && x[3] ≈ 1.0)

ch = ConstraintHandler{U}(mesh)


add_dirichlet_bc!(ch,cv.dh,cv.facedata_col,"symmetry_bc",x->SA[0.0],c_dofs = SA[1])
add_dirichlet_bc!(ch,cv.dh,cv.facedata_col,"symmetry_bc_2",x->SA[0.0],c_dofs = SA[2])
add_dirichlet_bc!(ch,cv.dh,"roller_bearing",x->SA[0.0,0.0],c_dofs = SA[2,3])

add_neumann_bc!(ch,cv.dh,cv.facedata_col,"middle_traction",x->SA[0.0,0.0,-1.0]) 

println("Computing initial displacement")
u0,k_global,eldata_col = compute_displacement(cv,ch,states,MBB_rhs,sim_pars);

Psi0 = 1/2 * u0' * k_global * u0
println("Initial strain energy: $Psi0")
u = u0

n_conv_until_stop = 2 


println("="^100)
println("Starting optimization")
println("="^100)
optimization_time = @elapsed let u = u0, Psi=Psi0, Psi0 = Psi0, eldata_col = eldata_col
    n_conv_count = 0
    Psi_step0 = Psi0
    for optimization_step in 1:2

        state_changed = state_update!(states,sim_pars,laplace_operator,u,eldata_col)
        u,k_global,eldata_col = compute_displacement(cv,ch,states,MBB_rhs,sim_pars)

        Psi = 1/2 * u' * k_global * u

        ΔPsi_rel = (Psi - Psi0) / Psi0 # no abs to show occilations in the convergence
        println("Optimization step: $optimization_step, Relative strain energy change: $ΔPsi_rel")
        improvement = round(Psi_step0/Psi * 100,sigdigits=2)
        println("Improvement: $improvement %")


        if abs(ΔPsi_rel) < 1e-6
            n_conv_count += 1
            if n_conv_count >= n_conv_until_stop
                break
            end
        else
            n_conv_count = 0
        end
        Psi0 = Psi



    end
end

println("Optimization time: $optimization_time")
write_vtk(cv.mesh.topo,"vtk/MBB_beam_sym_$(n)",cv.dh,u;cell_data = states.χ_vec)