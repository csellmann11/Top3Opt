# Run: julia --startup-file=no --project=. Tests/tpfa_geometry.jl
using Test, LinearAlgebra, SparseArrays, StaticArrays, OrderedCollections
using Ju3VEM, Ju3VEM.FixedSizeArrays
include("../src/mat_states.jl")
include("../src/laplace_operator.jl")
include("../src/utils/mesh_processing_utils.jl")
BLAS.set_num_threads(1)

function check_tpfa_mesh(mesh)
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.3)
    faces = build_tpfa_faces(cv,states)
    n = length(states.χ_vec)
    @test faces.area === states.area_vec
    @test all(>(0),faces.length)
    @test all(>(0),faces.distance_left)
    @test all(>(0),faces.distance_right)
    @test length(faces.left) == count(cells -> length(cells) == 2,values(regularization_faces(cv,states)))
    stiffness = diffusion_matrix(faces,2 .* states.h_vec.^2)
    flux = compute_flux_operator_mat(cv,states,(β0=1.,))
    @test stiffness ≈ stiffness'
    @test norm(stiffness*ones(n),Inf) < 1e-10
    @test norm(faces.area'*flux,Inf) < 1e-10
    @test norm(flux*ones(n),Inf) < 1e-9
    @test Diagonal(faces.area)*flux ≈ -stiffness
    @test minimum(eigvals(Symmetric(Matrix(stiffness)))) >= -1e-10
    @test iszero(compute_flux_operator_mat(cv,states,(β0=0.,)))
    @test_throws ArgumentError compute_flux_operator_mat(cv,states,(β0=1.,);scheme=:diamond)
    return faces,states
end

@testset "TPFA normal geometry" begin
    origin = SA[0.,0.,0.]
    n = SA[1.,0.,0.]
    left = SA[-1.,0.,0.]
    right = SA[0.5,0.3,-0.2]
    @test tpfa_face_distances(origin,n,left,right) == (1.,0.5)
    @test tpfa_face_distances(origin,-2n,left,right) == (1.,0.5)
    @test tpfa_face_distances(origin,n,right,left) == (0.5,1.)
    @test_throws ArgumentError tpfa_face_distances(origin,n,left,2left)
    @test_throws ArgumentError tpfa_face_distances(origin,zero(n),left,right)
    @test_throws ArgumentError tpfa_face_distances(origin,n,origin,right)
end

@testset "Uniform and hanging 3D TPFA faces" begin
    mesh = create_rectangular_mesh(3,3,3,1.,1.,1.,StandardEl{1})
    faces,states = check_tpfa_mesh(mesh)
    @test length(faces.left) == 54
    @test sum(faces.length) ≈ 6.0 atol=1e-10
    element = collect(RootIterator{4}(mesh.topo))[14]
    Ju3VEM.VEMGeo._refine!(element,mesh.topo)
    mesh = Mesh(mesh.topo,StandardEl{1}())
    faces,states = check_tpfa_mesh(mesh)
    @test length(states.χ_vec) == 34
    # Each of the central cell's six faces becomes four hanging subfaces;
    # the octants additionally share twelve interior faces.
    @test length(faces.left) == 54-6+24+12
end

@testset "Conservative coarse/fine TPFA interfaces" begin
    mesh = create_rectangular_mesh(2,1,1,2.,1.,1.,StandardEl{1})
    target = SA[1.,0.,0.]
    for level in 1:4
        cv = CellValues{3}(mesh)
        states = DesignVarInfo{3}(cv,0.3)
        candidates = findall(x -> x[1] < 1.,states.x_vec)
        sid = candidates[argmin([norm(states.x_vec[i]-target) for i in candidates])]
        element = only(el for el in RootIterator{4}(mesh.topo) if el.id == get_el_id(states,sid))
        Ju3VEM.VEMGeo._refine!(element,mesh.topo)
        mesh = Mesh(mesh.topo,StandardEl{1}())
        faces,states = check_tpfa_mesh(mesh)
        @test maximum(states.h_vec)/minimum(states.h_vec) ≈ 2.0^level
        interface_area = sum(faces.length[f] for f in eachindex(faces.left)
            if (states.x_vec[faces.left[f]][1] < 1.) != (states.x_vec[faces.right[f]][1] < 1.))
        @test interface_area ≈ 1.0 atol=1e-10
    end
end

@testset "Extruded Voronoi TPFA" begin
    mesh2d = create_voronoi_mesh((0.,0.),(1.,1.),3,3,StandardEl{1})
    topo = mesh2d.topo
    for _ in 1:3
        topo = remove_short_edges(topo)
    end
    mesh = extrude_to_3d(2,Mesh(topo,StandardEl{1}()),1.)
    check_tpfa_mesh(mesh)
end
