# Run from ToOpt3 with one BLAS thread:
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. --startup-file=no --threads=1 Tests/l_cantilever_voronoi.jl
# mechanics.jl provides the production imports, utilities, and elastic patch
# checks without running its own suite when included.
include("mechanics.jl")
using Ju3VEM.VEMGeo: _refine!
include("../src/utils/benchmark_meshes.jl")
include("../src/diamond_flux.jl")

const L_GEO = Ju3VEM.VEMGeo

function l_boundary_segment(points; atol=1e-10)
    on_line(d, value) = all(p -> abs(p[d]-value) <= atol, points)
    return on_line(1, 0.) || on_line(2, 0.) ||
        (on_line(1, 2.) && all(p -> p[2] <= 1.0 + atol, points)) ||
        (on_line(2, 2.) && all(p -> p[1] <= 1.0 + atol, points)) ||
        (on_line(1, 1.) && all(p -> p[2] >= 1.0 - atol, points)) ||
        (on_line(2, 1.) && all(p -> p[1] >= 1.0 - atol, points))
end

function l_mesh_signature(mesh)
    polygons = [sort([Tuple(mesh.topo.nodes[n].coords)
        for n in get_area_node_ids(mesh.topo, cell.id)])
        for cell in RootIterator{3}(mesh.topo)]
    return sort(Tuple.(polygons))
end

function check_l_voronoi_2d(mesh)
    topo = mesh.topo
    owners = Dict{Int,Int}()
    used = Set{Int}()
    total_area = 0.
    polygon_sizes = Int[]
    for cell in RootIterator{3}(topo)
        ids = collect(get_area_node_ids(topo, cell.id))
        union!(used, ids)
        push!(polygon_sizes, length(ids))
        @test length(ids) >= 3
        @test length(unique(ids)) == length(ids)
        points = [topo.nodes[n].coords for n in ids]
        signed_area = sum(points[i][1]*points[mod1(i+1,end)][2] -
            points[i][2]*points[mod1(i+1,end)][1] for i in eachindex(points))/2
        @test isfinite(signed_area) && abs(signed_area) > 1e-12
        total_area += abs(signed_area)
        for i in eachindex(points)
            a, b, c = points[i], points[mod1(i+1,end)], points[mod1(i+2,end)]
            cross_z = (b[1]-a[1])*(c[2]-b[2]) - (b[2]-a[2])*(c[1]-b[1])
            @test sign(signed_area)*cross_z >= -1e-10
        end
        L_GEO.iterate_element_edges(topo, cell.id) do _, eid, _
            owners[eid] = get(owners, eid, 0)+1
        end
    end
    perimeter = 0.
    for (eid, nowners) in owners
        a, b = topo.nodes[get_edge_node_ids(topo, eid)]
        boundary = l_boundary_segment((a, b))
        @test nowners == (boundary ? 1 : 2)
        boundary && (perimeter += norm(a-b))
    end
    @test total_area ≈ 3. atol=1e-10
    @test perimeter ≈ 8. atol=1e-10
    @test any(>(4), polygon_sizes) # The selected mesh must contain Voronoi polygons.
    @test used == Set(n.id for n in topo.nodes if is_active(n))
    coords = [Tuple(round.(topo.nodes[n].coords; digits=11)) for n in used]
    @test length(unique(coords)) == length(coords)
    for n in used
        x, y = topo.nodes[n]
        @test -1e-10 <= x <= 2.0 + 1e-10
        @test -1e-10 <= y <= 2.0 + 1e-10
        @test x <= 1.0 + 1e-10 || y <= 1.0 + 1e-10
    end
    for corner in ((0.,0.), (2.,0.), (2.,1.), (1.,1.), (1.,2.), (0.,2.))
        @test count(n -> norm(topo.nodes[n].coords-SVector(corner...)) < 1e-10, used) == 1
    end
end

function check_l_voronoi_3d(mesh)
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv, 0.15)
    @test all(v -> isfinite(v) && v > 0., states.area_vec)
    @test sum(states.area_vec) ≈ 1.5 atol=1e-9
    @test validate_vem_projectors(cv) === nothing
    coords = [n.coords for n in mesh.topo.nodes if is_active(n)]
    for d in 1:3
        @test minimum(p[d] for p in coords) ≈ 0. atol=1e-10
        @test maximum(p[d] for p in coords) ≈ (2.,2.,.5)[d] atol=1e-10
    end
    owners = regularization_faces(cv, states)
    boundary_area = 0.
    for (fid, cells) in owners
        fd = cv.facedata_col[fid]
        ids = fd.face_node_ids.v.args[1]
        points = [mesh.topo.nodes[n].coords for n in ids]
        boundary = all(p -> abs(p[3]) < 1e-10, points) ||
            all(p -> abs(p[3] - 0.5) < 1e-10, points) || l_boundary_segment(points)
        @test length(cells) == (boundary ? 1 : 2)
        anchor = sum(points)/length(points)
        area_vector = sum(cross(points[i]-anchor, points[mod1(i+1,end)]-anchor)
            for i in eachindex(points))/2
        area = norm(area_vector)
        @test isfinite(area) && area > 1e-12
        boundary && (boundary_area += area)
        for sid in cells
            eid = get_el_id(states, sid)
            normal = dot(area_vector, anchor-states.x_vec[sid]) > 0. ? area_vector : -area_vector
            @test all(n -> dot(mesh.topo.nodes[n].coords-anchor, normal) <= 1e-10*area,
                get_volume_node_ids(mesh.topo, eid))
        end
    end
    @test boundary_area ≈ 10. atol=1e-9 # Two L faces plus the extruded perimeter.
    R = compute_flux_operator_mat(cv, states, (β0=1.,))
    @test all(isfinite, nonzeros(R))
    @test norm(R*ones(length(states.χ_vec)), Inf) < 1e-8
    @test norm(states.area_vec'*R, Inf) < 1e-9
    return cv, states
end

@testset "L-cantilever Voronoi mesh" begin
    @testset "Conforming deterministic planar L" begin
        mesh2d = create_L_voronoi_mesh((0.,0.), (2.,2.), 8, 8, StandardEl{1})
        check_l_voronoi_2d(mesh2d)
        repeated = create_L_voronoi_mesh((0.,0.), (2.,2.), 8, 8, StandardEl{1})
        @test l_mesh_signature(mesh2d) == l_mesh_signature(repeated)
    end

    mesh = create_benchmark_mesh(:L_cantilever, :Voronoi, 2., 2., .5, 8, 8, 2, StandardEl{1})
    @testset "Production extrusion and mechanics" begin
        check_l_voronoi_3d(mesh)
        @test any(cell -> length(get_volume_node_ids(mesh.topo, cell.id)) > 8,
            RootIterator{4}(mesh.topo))
        check_elastic_patch(mesh, mechanics_parameters(0.15))
    end

    @testset "Refinement and L boundary conditions" begin
        base_cell_count = count(is_active_root, get_volumes(mesh.topo))
        refined, _ = refine_sets(mesh, get_sets_to_refine(:L_cantilever), 2)
        @test count(is_active_root, get_volumes(refined.topo)) > base_cell_count
        cv, states = check_l_voronoi_3d(refined)
        ch = create_constraint_handler(cv, :L_cantilever)
        @test length(refined.node_sets["top_clamp1"]) == 1
        @test length(refined.node_sets["top_clamp2"]) == 1
        @test !isempty(refined.face_sets["symmetry_bc"])
        @test !isempty(refined.face_sets["traction"])
        @test !isempty(ch.d_bcs)
        @test !isempty(ch.n_bcs)
        @test all(isfinite, values(ch.n_bcs))
        @test sum(values(ch.n_bcs)) < 0.
        for fid in refined.face_sets["traction"]
            @test all(n -> 1.749 <= refined.topo.nodes[n][1] <= 2. &&
                refined.topo.nodes[n][2] ≈ 1. && refined.topo.nodes[n][3] >= .374,
                get_area_node_ids(refined.topo, fid))
        end

        # Refining a cell at the reentrant corner also exercises hanging faces
        # away from the force-refinement strip.
        sid = argmin([norm(x-SA[1.,1.,.25]) for x in states.x_vec])
        eid = get_el_id(states, sid)
        target = only(cell for cell in RootIterator{4}(refined.topo) if cell.id == eid)
        L_GEO._refine!(target, refined.topo)
        check_l_voronoi_3d(Mesh(refined.topo, StandardEl{1}()))
    end

    @testset "Hexahedral L selection remains available" begin
        structured = create_benchmark_mesh(:L_cantilever, :Hexahedra, 2., 2., .5, 8, 8, 2, StandardEl{1})
        @test count(is_active_root, get_volumes(structured.topo)) == 96
        @test all(cell -> length(get_volume_node_ids(structured.topo, cell.id)) == 8,
            RootIterator{4}(structured.topo))
    end
end
