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
include("../src/laplace_operator_gauss_kernel.jl")


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
#         Ju3VEM.VEMGeo._refine!(element,mesh.topo)
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

# @time laplace_operator = compute_laplace_operator_mat(
#     cv.mesh.topo,state_neights_col,b_face_id_to_state_id,states,sim_pars)                

state_neights_col[5] |> display 
# [b_face_id_to_state_id[key] for key in state_neights_col[5][15:end]]


f = x -> cos(pi*x[1])*cos(pi*x[2])*cos(pi*x[3])#-1/3*x[1]^3 + 1/2*x[1]^2

import Ju3VEM.FR.ForwardDiff as FD


@show FD.hessian(f,SA[0.0,0.0,0.0]) |> tr
@show FD.hessian(f,SA[0.5,0.0,0.0]) |> tr




for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    xbc = states.x_vec[sid] 
    states.χ_vec[sid] = f(xbc)
end



@fastmath function gauss_kernel(x,y;shape_parameter = 0.1)
    return exp(-(shape_parameter*norm(x-y))^2)
end

function laplacian_gauss_kernel(x, y; shape_parameter=0.1, d=3)
    r = norm(x - y)
    ε = shape_parameter
    # Δφ(r) = 4ε⁴r² exp(-ε²r²) - 2dε² exp(-ε²r²)
    return ε^2 * (4*ε^2*r^2 - 2*d) * exp(-ε^2 * r^2)
end


function kernel_matrix(state_ids, x_vec, χ_vec)
    n = length(state_ids)
    A = zeros(Float64, n, n)
    
    # Exploit symmetry: only compute upper triangle
    for i in 1:n
        A[i,i] = 1.0  # gauss_kernel(x,x) = 1
        for j in (i+1):n
            x = x_vec[state_ids[i]]
            y = x_vec[state_ids[j]]
            A[i,j] = gauss_kernel(x, y;shape_parameter = 0.01)
            A[j,i] = A[i,j]  # symmetric
        end
    end
    
    coeffs = A \ χ_vec[state_ids]
    return coeffs
end

function laplacian_weights(
    x,                    # evaluation point
    state_ids,           # neighbor indices
    x_vec;               # all node positions
    shape_parameter=0.1
)
    n = length(state_ids)
    d = length(x)  # spatial dimension
    
    # Build kernel matrix Φ
    Φ = zeros(Float64, n, n)
    for i in 1:n
        Φ[i,i] = 1.0 
        for j in (i+1):n
            xi = x_vec[state_ids[i]]
            xj = x_vec[state_ids[j]]
            Φ[i,j] = gauss_kernel(xi, xj; shape_parameter)
            Φ[j,i] = Φ[i,j]
        end
    end
    
    # Build Laplacian kernel vector at x
    b = zeros(Float64, n)
    ε = shape_parameter
    for i in 1:n
        r = norm(x - x_vec[state_ids[i]])
        # Δφ(r) = ε²(4ε²r² - 2d)exp(-ε²r²)
        b[i] = ε^2 * (4*ε^2*r^2 - 2*d) * exp(-ε^2 * r^2)
    end
    
    # Solve: Φ' * v = b  (transpose because v' * vals)
    v = qr(Φ) \ b
    
    return v
end


function get_ip(x,coeffs,state_ids,x_vec)
    val = sum(coeffs[i] * gauss_kernel(x,x_vec[state_ids[i]]) for i in eachindex(state_ids))
    return val
end


function get_laplacian(x, coeffs, state_ids, x_vec; d=3)
    return sum(coeffs[i] * laplacian_gauss_kernel(x, x_vec[state_ids[i]]) 
               for i in eachindex(state_ids))
end




nels = length(RootIterator{4}(mesh.topo))
vals = Vector{Float64}(undef,nels)
for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    xbc = states.x_vec[sid] 
    vals[sid] = f(xbc)
end


state_id = rand(1:length(states.x_vec))
@show state_id
@show states.x_vec[state_id]
h0 = states.h_vec[state_id]

n_ids = state_neights_col[state_id]


n_ids = n_ids[n_ids .> 0]

f_vals = states.χ_vec[n_ids]


f_true = f(states.x_vec[state_id])



# teststing custom kernel interpolation 




lap = laplace_operator * states.χ_vec





for element in RootIterator{4}(mesh.topo)
    sid = states.el_id_to_state_id[element.id]
    states.χ_vec[sid] = lap[sid]
end

@show maximum(states.χ_vec)
write_vtk(cv.mesh.topo,"Results/vtk/lap_test";cell_data_col = (states.χ_vec,))


 