# Independent polynomial/quadrature references for the analytic tetra kernels.
using Test, LinearAlgebra, StaticArrays, Random
isdefined(@__MODULE__,:PU3DTetra) || include("../src/refinement_utils/pu_geometry.jl")
isdefined(@__MODULE__,:pu3d_solve!) || include("../src/refinement_utils/pu_tetra.jl")

pu_test_monomials(x) = SA[1.,x[1],x[2],x[3],x[1]^2,x[2]^2,x[3]^2,
    x[1]*x[2],x[1]*x[3],x[2]*x[3]]
pu_test_derivatives(x) = @SMatrix [
    0. 1. 0. 0. 2x[1] 0. 0. x[2] x[3] 0.;
    0. 0. 1. 0. 0. 2x[2] 0. x[1] 0. x[3];
    0. 0. 0. 1. 0. 0. 2x[3] 0. x[1] x[2]]

function pu_test_reference()
    # Vandermonde interpolation defines P2 independently of barycentric formulas.
    vertices = [SA[0.,0.,0.],SA[1.,0.,0.],SA[0.,1.,0.],SA[0.,0.,1.]]
    edge_pairs = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
    nodes = vcat(vertices,[(vertices[i]+vertices[j])/2 for (i,j) in edge_pairs])
    V = reduce(vcat,[transpose(pu_test_monomials(x)) for x in nodes])
    coefficients = inv(V)
    # Product Gauss-Legendre under the Duffy cube-to-tetrahedron transform.
    abscissae = [-0.9061798459386640,-0.5384693101056831,0.,
                  0.5384693101056831,0.9061798459386640]
    weights = [0.2369268850561891,0.4786286704993665,0.5688888888888889,
               0.4786286704993665,0.2369268850561891]
    quadrature = Tuple{SVector{3,Float64},Float64}[]
    for i in 1:5,j in 1:5,k in 1:5
        r,s,t = (abscissae[i]+1)/2,(abscissae[j]+1)/2,(abscissae[k]+1)/2
        x = SA[r,(1-r)*s,(1-r)*(1-s)*t]
        weight = weights[i]*weights[j]*weights[k]/8*(1-r)^2*(1-s)
        push!(quadrature,(x,weight))
    end
    return nodes,coefficients,quadrature
end

function check_pu_tetra(points)
    t = pu3d_tetra(SA[1,2,3,4],points,1,1)
    local_nodes,coefficients,quadrature = pu_test_reference()
    B = hcat(points[2]-points[1],points[3]-points[1],points[4]-points[1])
    nodes = [points[1]+B*x for x in local_nodes]
    lambda,mu = 7.3,2.1
    stiffness = zeros(30,30)
    gradient_integrals = zeros(3,10)
    product_integrals = zeros(3,10,4)
    @test sum(w for (_,w) in quadrature) ≈ 1/6 atol=1e-15
    @test t.volume ≈ abs(det(B))/6
    p2_values_ok = true; p2_gradients_ok = true
    for (x,wref) in quadrature
        phi = transpose(coefficients)*pu_test_monomials(x)
        gradients = transpose(inv(B))*pu_test_derivatives(x)*coefficients
        N = SA[1-sum(x),x[1],x[2],x[3]]
        computed_g = pu3d_p2_gradients(t,N)
        p2_values_ok &= isapprox(pu3d_p2_values(N),phi;atol=2e-14)
        p2_gradients_ok &= all(isapprox(computed_g[i],gradients[:,i];atol=2e-13) for i in 1:10)
        weight = abs(det(B))*wref
        strains = [begin
            G = zeros(3,3); G[c,:] = gradients[:,i]
            (G+G')/2
        end for i in 1:10 for c in 1:3]
        for i in 1:30,j in 1:30
            stiffness[i,j] += weight*(lambda*tr(strains[i])*tr(strains[j])+
                2mu*sum(strains[i].*strains[j]))
        end
        gradient_integrals .+= weight*gradients
        # The affine hats are recovered from an independent linear interpolant.
        linear_coeff = inv(hcat(ones(4),reduce(vcat,transpose.(points))))
        hat_gradients = linear_coeff[2:4,:]
        for hat in 1:4,i in 1:10
            product_integrals[:,i,hat] .+= weight*(N[hat]*gradients[:,i]+
                phi[i]*hat_gradients[:,hat])
        end
    end
    @test p2_values_ok
    @test p2_gradients_ok
    analytic = [t.volume*pu3d_stiffness_entry(t,cld(row,3),cld(col,3),
        mod1(row,3),mod1(col,3),lambda,mu) for row in 1:30,col in 1:30]
    @test analytic ≈ stiffness rtol=3e-13 atol=3e-13
    @test analytic ≈ transpose(analytic) atol=2e-14
    eigs = eigvals(Symmetric(analytic))
    @test minimum(eigs) >= -2e-12*maximum(eigs)
    @test count(x -> abs(x) < 1e-10*maximum(eigs),eigs) == 6
    for hat in 1:4,i in 1:10
        g,gweighted = pu3d_internal_gradients(t,i,hat)
        @test t.volume*g ≈ gradient_integrals[:,i] atol=3e-14
        @test t.volume*gweighted ≈ product_integrals[:,i,hat] atol=3e-14
    end
    for axis in (SA[1.,0.,0.],SA[0.,1.,0.],SA[0.,0.,1.])
        for f in (x -> axis,x -> cross(axis,x))
            u = reduce(vcat,[f(x) for x in nodes])
            @test norm(analytic*u) < 2e-12*norm(analytic)*norm(u)
        end
    end
    # Reproduce a general physical quadratic and its exact first derivative.
    H = SA[2. -0.4 0.6; -0.4 -0.8 0.3; 0.6 0.3 0.2]
    a = SA[0.2,-0.6,0.1]
    p(x) = 0.7+dot(a,x)+dot(x,H*x)/2
    values = p.(nodes)
    xref = SA[0.13,0.21,0.19]
    x = points[1]+B*xref
    N = SA[1-sum(xref),xref[1],xref[2],xref[3]]
    @test dot(pu3d_p2_values(N),values) ≈ p(x) atol=2e-13
    @test sum(values[i]*pu3d_p2_gradients(t,N)[i] for i in 1:10) ≈ a+H*x atol=2e-13
end

function pu_tetra_tests()
    @testset "3D P2 tetra kernels" begin
        @testset "Reference tetrahedron" begin
            check_pu_tetra([SA[0.,0.,0.],SA[1.,0.,0.],SA[0.,1.,0.],SA[0.,0.,1.]])
        end
        @testset "Skew translated tetrahedron" begin
            check_pu_tetra([SA[0.2,-0.3,0.4],SA[1.3,0.1,0.2],
                SA[0.6,0.7,0.5],SA[-0.1,0.2,1.2]])
        end
        @testset "Reversed orientation" begin
            check_pu_tetra([SA[0.2,-0.3,0.4],SA[0.6,0.7,0.5],
                SA[1.3,0.1,0.2],SA[-0.1,0.2,1.2]])
        end
        @testset "In-place local solve" begin
            rng = MersenneTwister(3201)
            for n in (1,3,14,50)
                R = randn(rng,n,n)
                A = R*R'+I
                rhs = randn(rng,n)
                matrix = fill(NaN,n+3,n+3)
                for j in 1:n,i in j:n
                    matrix[i,j] = A[i,j]
                end
                solution = vcat(rhs,fill(8.,3))
                pu3d_solve!(matrix,solution,n)
                @test solution[1:n] ≈ A\rhs rtol=1e-12
                @test all(==(8.),solution[n+1:end])
                @test all(isnan,matrix[n+1:end,:])
            end
            @test_throws PosDefException pu3d_solve!(zeros(2,2),ones(2),2)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    pu_tetra_tests()
end
