using Test, Ju3VEM, Ju3VEM.FixedSizeArrays, StaticArrays, OrderedCollections
using LinearAlgebra, SparseArrays, Bumper
include("../src/mat_states.jl")
include("../src/laplace_operator.jl")
include("../src/neighbor_search.jl")
include("taylor_reference.jl")

@testset "Supplied Taylor product-rule reference" begin
    cv = CellValues{3}(create_rectangular_mesh(5,5,5,1.,1.,1.,StandardEl{1}))
    states = DesignVarInfo{3}(cv,0.3)
    n = length(states.χ_vec)
    beta = [1+dot(SA[0.2,0.3,-0.1],x) for x in states.x_vec]
    chi = [x^2+y^2/2+z^2/4+0.4x*y+0.2x*z+0.1y*z for (x,y,z) in states.x_vec]
    exact = [3.5b+0.2*(2x+0.4y+0.2z)+0.3*(y+0.4x+0.1z)-0.1*(z/2+0.2x+0.1y)
        for ((x,y,z),b) in zip(states.x_vec,beta)]
    R = compute_reference_taylor_mat(cv,states,(β0=1.,);beta)
    interior = findall(x->all(t->0.3-1e-10 <= t <= 0.7+1e-10,x),states.x_vec)
    @test norm(R*ones(n),Inf) < 1e-10
    @test maximum(abs.((R*chi)[interior]-exact[interior])) < 0.02
    neighbours,ghosts = create_neigh_list(states,cv)
    L = compute_laplace_operator_mat(cv.mesh.topo,neighbours,ghosts,states,false)
    Rconstant = compute_reference_taylor_mat(cv,states,(β0=1.,);beta=fill(2.,n))
    @test norm(Rconstant-2L,Inf) < 1e-10
end
