# Run from ToOpt3: julia --project=. --startup-file=no Tests/mechanics.jl
# Exercise the production VEM/assembly path against elastic invariants, not a
# second copy of its matrix-product implementation.
using Test, LinearAlgebra, SparseArrays, StaticArrays, OrderedCollections
using Ju3VEM, Ju3VEM.FixedSizeArrays, Bumper, TimerOutputs, Random
import Ju3VEM.FR as FR
import Ju3VEM.VEMUtils as VU

const U = 3
const to = TimerOutput()
BLAS.set_num_threads(1)
include("../src/mat_states.jl")
include("../src/compute_displacement.jl")
include("../src/bisection.jl")
include("../src/get_sparsity_pattern.jl")
include("../src/utils/mesh_processing_utils.jl")
include("../src/utils/temp_utils.jl")

function mechanics_parameters(rho=0.3)
    lam, mu = E_ν_to_lame(210.e3, 0.33)
    material = Helmholtz{3,3}(Ψlin_totopt, (lam, mu, 1.))
    return SimPars(material, lam, mu, 1e-3, 15., 1., rho)
end

function interpolate_displacement(cv, field)
    u = zeros(3length(cv.dh.dof_mapping))
    for (nid, dofs) in cv.dh.dof_mapping
        u[dofs] = field(cv.mesh.nodes[nid].coords)
    end
    return u
end

function check_projectors(cv)
    @testset "Face polynomial reproduction" begin
        for fd in values(cv.facedata_col)
            D = Matrix(VU.create_D_mat(cv.mesh, fd))
            @test Matrix(fd.ΠsL2)*D ≈ I atol=2e-10
        end
    end
    @testset "Volume polynomial reproduction" begin
        for element in RootIterator{4}(cv.mesh.topo)
            reinit!(element.id, cv)
            D = Matrix(VU.create_volume_dmat(element.id, cv.mesh,
                cv.facedata_col, cv.volume_data, cv.vnm))
            Ps, P = create_volume_vem_projectors(element.id, cv.mesh,
                cv.volume_data, cv.facedata_col, cv.vnm)
            @test Matrix(Ps)*D ≈ I atol=2e-10
            @test Matrix(P)*D ≈ D atol=2e-10
            @test Matrix(P)*Matrix(P) ≈ Matrix(P) atol=2e-10
        end
    end
end

function check_elastic_patch(mesh, pars)
    cv = CellValues{3}(mesh)
    @test validate_vem_projectors(cv) === nothing
    check_projectors(cv)
    states = DesignVarInfo{3}(cv, pars.ρ_init)
    stiffness, rhs, ed = assembly(cv, states, x -> SA[0.,0.,0.], pars)
    @test norm(stiffness-stiffness') <= 1e-12*norm(stiffness)
    @test iszero(rhs)
    @testset "Six rigid motions" begin
        for axis in (SA[1.,0.,0.], SA[0.,1.,0.], SA[0.,0.,1.])
            for field in (x -> axis, x -> cross(axis,x))
                u = interpolate_displacement(cv, field)
                @test norm(stiffness*u) <= 1e-11*norm(stiffness)*norm(u)
            end
        end
    end
    @testset "Affine strain and energy" begin
        # Independent closed-form isotropic elastic energy, including mixed
        # normal/shear strains and an arbitrary translation.
        for G in (SA[0.02 0.04 -0.01; -0.03 -0.01 0.06; 0.07 0.02 0.03],
                  SA[0.01 0.0 0.0; 0.0 -0.0033 0.0; 0.0 0.0 -0.0033])
            u = interpolate_displacement(cv, x -> G*x + SA[0.1,-0.2,0.3])
            strain = (G+G')/2
            density_energy = pars.λ/2*tr(strain)^2 + pars.μ*sum(abs2,strain)
            exact = sum(states.area_vec)*pars.ρ_init^3*density_energy
            @test dot(u,stiffness*u)/2 ≈ exact rtol=1e-9
            psi = compute_strain_energy(cv.dh,ed,u,states,pars)
            @test psi ≈ fill(density_energy,length(psi)) rtol=1e-9
        end
    end
end

function mechanics_tests()
    pars = mechanics_parameters()
    @testset "3D mechanics regression" begin
        @testset "Reject invalid dependency projectors" begin
            cv = CellValues{3}(create_rectangular_mesh(1,1,1,1.,1.,1.,StandardEl{1}))
            first(values(cv.facedata_col)).ΠsL2[2,:] .*= 0.25
            @test_throws ErrorException validate_vem_projectors(cv)
        end
        @testset "Unit cube" begin
            check_elastic_patch(create_rectangular_mesh(1,1,1,1.,1.,1.,StandardEl{1}),pars)
        end
        @testset "Small anisotropic cells" begin
            check_elastic_patch(create_rectangular_mesh(2,2,2,0.4,0.2,0.1,StandardEl{1}),pars)
        end
        @testset "Hanging nodes" begin
            mesh = create_rectangular_mesh(2,2,2,1.,1.,1.,StandardEl{1})
            Ju3VEM.VEMGeo._refine!(first(RootIterator{4}(mesh.topo)),mesh.topo)
            check_elastic_patch(Mesh(mesh.topo,StandardEl{1}()),pars)
        end
        @testset "Skew cells" begin
            mesh = create_rectangular_mesh(2,2,2,1.,1.,1.,StandardEl{1})
            transform = SA[1.0 0.25 0.1; 0.0 0.8 0.2; 0.0 0.0 0.6]
            for i in eachindex(mesh.topo.nodes)
                node = mesh.topo.nodes[i]
                mesh.topo.nodes[i] = Node(node.id,transform*node.coords + SA[0.2,-0.3,0.1])
            end
            check_elastic_patch(Mesh(mesh.topo,StandardEl{1}()),pars)
        end
        @testset "Extruded Voronoi" begin
            Random.seed!(1729)
            mesh2d = create_voronoi_mesh((0.,0.),(1.,1.),3,3,StandardEl{1})
            check_elastic_patch(extrude_to_3d(2,mesh2d,0.5),pars)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    mechanics_tests()
end
