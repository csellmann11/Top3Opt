# Run: julia --startup-file=no --project=. Tests/pu_geometry.jl
isdefined(@__MODULE__,:mechanics_parameters) || include("mechanics.jl")
isdefined(@__MODULE__,:PU3DGeometry) || include("../src/refinement_utils/pu_geometry.jl")

function pu_geometry_fixture(mesh)
    cv = CellValues{3}(mesh)
    pars = mechanics_parameters()
    states = DesignVarInfo{3}(cv,pars.ρ_init)
    _,_,ed = assembly(cv,states,x -> SA[0.,0.,0.],pars)
    return cv,states,ed,build_pu_geometry(cv,states,ed)
end

function check_pu_geometry(mesh)
    cv,states,ed,g = pu_geometry_fixture(mesh)
    nv,nf,nc = length(g.vertex_ids),length(g.face_ids),length(g.element_ids)
    @test length(g.node_points) == nv+nf+nc
    @test g.element_ids == collect(keys(states.el_id_to_state_id))
    @test g.cell_offsets[1] == 1
    @test g.cell_offsets[end] == length(g.tets)+1
    @test all(diff(g.cell_offsets) .> 0)
    @test all(t.volume > 0 for t in g.tets)
    @test sum(t.volume for t in g.tets) ≈ sum(states.area_vec) rtol=1e-10
    @test all(g.vertex_ids[g.cell_nodes[c]] == collect(ed[eid].node_ids)
        for (c,eid) in enumerate(g.element_ids))
    @test all(g.node_points[nv+f] ≈ Ju3VEM.VEMGeo.project_to_3d(
        cv.facedata_col[fid].dΩ.bc,cv.facedata_col[fid].dΩ.plane)
        for (f,fid) in enumerate(g.face_ids))
    hats_ok = true; affine_ok = true; cell_means_ok = true
    triangles = Dict{Tuple{Int,Int,Int},Vector{Int}}()
    A = SA[0.8 0.2 -0.1; 0.4 -0.3 0.7; -0.2 0.1 0.5]
    offset = SA[0.1,-0.3,0.2]
    for (c,eid) in enumerate(g.element_ids)
        volume = 0.0; moment = zero(SVector{3,Float64})
        for ti in g.cell_offsets[c]:g.cell_offsets[c+1]-1
            t = g.tets[ti]
            @test t.cell == c && t.nodes[3] == nv+t.face && t.nodes[4] == nv+nf+c
            x = SA[g.node_points[t.nodes[1]],g.node_points[t.nodes[2]],
                   g.node_points[t.nodes[3]],g.node_points[t.nodes[4]]]
            volume += t.volume
            moment += t.volume*sum(x)/4
            for i in 1:4
                target = SVector{4,Float64}(ntuple(j -> Float64(i == j),4))
                hats_ok &= isapprox(pu3d_values(t,g,x[i]),target;atol=2e-12)
            end
            N = SA[0.1,0.2,0.3,0.4]
            p = sum(N[i]*x[i] for i in 1:4)
            hats_ok &= isapprox(pu3d_values(t,g,p),N;atol=2e-12)
            hats_ok &= norm(sum(t.gradients)) < 1e-11
            reconstructed = sum((A*x[i]+offset)*transpose(t.gradients[i]) for i in 1:4)
            affine_ok &= isapprox(reconstructed,A;atol=2e-11)
            key = (min(t.nodes[1],t.nodes[2]),max(t.nodes[1],t.nodes[2]),t.nodes[3])
            push!(get!(triangles,key,Int[]),c)
        end
        cell_means_ok &= isapprox(volume,ed[eid].volume;rtol=1e-10)
        cell_means_ok &= isapprox(moment/volume,ed[eid].bc;atol=2e-11)
    end
    @test hats_ok
    @test affine_ok
    @test cell_means_ok
    @test all(sort(cells) == sort(filter(!iszero,collect(g.face_cells[key[3]-nv])))
        for (key,cells) in triangles)
    @test all(length(cells) in (1,2) for cells in values(triangles))
    return cv,states,ed,g
end

function pu_geometry_tests()
    @testset "3D PU geometry" begin
        @testset "Cube and shared face" begin
            _,_,_,g = check_pu_geometry(create_rectangular_mesh(1,1,1,1.,1.,1.,StandardEl{1}))
            @test length(g.tets) == 24
            @test all(isapprox(t.volume,1/24) for t in g.tets)
            _,_,_,g = check_pu_geometry(create_rectangular_mesh(2,1,1,2.,1.,1.,StandardEl{1}))
            @test count(c -> c[2] != 0,g.face_cells) == 1
        end
        @testset "Hanging leaf faces and edges" begin
            mesh = create_rectangular_mesh(2,2,2,1.,1.,1.,StandardEl{1})
            Ju3VEM.VEMGeo._refine!(first(RootIterator{4}(mesh.topo)),mesh.topo)
            check_pu_geometry(Mesh(mesh.topo,StandardEl{1}()))
            mesh = create_rectangular_mesh(2,1,1,2.,1.,1.,StandardEl{1})
            for _ in 1:4
                cv = CellValues{3}(mesh)
                states = DesignVarInfo{3}(cv,0.3)
                ids = findall(x -> x[1] < 1.,states.x_vec)
                sid = ids[argmin([norm(states.x_vec[i]-SA[1.,0.,0.]) for i in ids])]
                eid = get_el_id(states,sid)
                Ju3VEM.VEMGeo._refine!(get_volumes(mesh.topo)[eid],mesh.topo)
                mesh = Mesh(mesh.topo,StandardEl{1}())
            end
            _,states,_,_ = check_pu_geometry(mesh)
            @test maximum(states.h_vec)/minimum(states.h_vec) ≈ 16.
        end
        @testset "Anisotropic affine transform" begin
            mesh = create_rectangular_mesh(2,2,2,1.,1.,1.,StandardEl{1})
            transform = SA[1.0 0.25 0.1; 0.0 0.8 0.2; 0.0 0.0 0.06]
            for i in eachindex(mesh.topo.nodes)
                node = mesh.topo.nodes[i]
                mesh.topo.nodes[i] = Node(node.id,transform*node.coords+SA[0.2,-0.3,0.1])
            end
            check_pu_geometry(Mesh(mesh.topo,StandardEl{1}()))
        end
        @testset "Extruded Voronoi" begin
            Random.seed!(1729)
            mesh2d = create_voronoi_mesh((0.,0.),(1.,1.),3,3,StandardEl{1})
            check_pu_geometry(extrude_to_3d(2,mesh2d,0.5))
        end
        @testset "Reject invalid tetrahedra and cell centers" begin
            points = [SA[0.,0.,0.],SA[1.,0.,0.],SA[0.,1.,0.],SA[1.,1.,0.]]
            @test_throws ArgumentError pu3d_tetra(SA[1,2,3,4],points,1,1)
            cv,states,ed,_ = pu_geometry_fixture(create_rectangular_mesh(1,1,1,1.,1.,1.,StandardEl{1}))
            eid = first(keys(ed)); data = ed[eid]
            ed[eid] = ElData(data.el_id,data.proj_s,data.proj,data.node_ids,
                data.γ_stab,data.hvol,data.volume,SA[2.,2.,2.])
            @test_throws ArgumentError build_pu_geometry(cv,states,ed)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    pu_geometry_tests()
end
