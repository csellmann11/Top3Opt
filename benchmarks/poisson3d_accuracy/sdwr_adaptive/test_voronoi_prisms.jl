using Test
include("load.jl")
include("voronoi_prisms.jl")

function sdwr_prism_physical_face(points,domain;tol=2e-10)
    plane(d,value) = all(abs(x-value)<=tol for x in points[d,:])
    plane(3,0.0) || plane(3,1.0) || if domain == :cube
        plane(1,0.0) || plane(1,1.0) || plane(2,0.0) || plane(2,1.0)
    else
        plane(1,-1.0) || plane(2,1.0) ||
        (plane(1,1.0) && minimum(points[2,:]) >= -tol) ||
        (plane(2,-1.0) && maximum(points[1,:]) <= tol) ||
        (plane(1,0.0) && maximum(points[2,:]) <= tol) ||
        (plane(2,0.0) && minimum(points[1,:]) >= -tol)
    end
end

@testset "Production Voronoi prisms: topology and scalar consistency" begin
    for domain in (:cube,:lshape),n in (2,3)
        mesh = sdwr_voronoi_prism_mesh(domain,n)
        fan = adaptive_build_fan(mesh)
        @test mesh.mesh_family == :voronoi_prisms
        @test sum(mesh.volumes) ≈ (domain==:cube ? 1.0 : 3.0) atol=2e-10
        @test sum(fan.volumes) ≈ sum(mesh.volumes) atol=2e-10
        @test all(>(0.0),fan.volumes)
        @test any(length(ids)>8 for ids in mesh.cells)
        @test size(mesh.points,2) == length(mesh.original_vertices)
        @test length(mesh.cells) == length(mesh.original_cells)
        @test length(mesh.faces) == length(mesh.original_faces)
        @test minimum(mesh.points[3,:]) ≈ 0.0
        @test maximum(mesh.points[3,:]) ≈ 1.0
        if domain == :lshape
            @test all(mesh.points[1,node]<=2e-10 || mesh.points[2,node]>=-2e-10
                for node in axes(mesh.points,2))
        end
        triangles = Dict{NTuple{3,Int},Int}()
        for tet in axes(fan.tet_nodes,2),opposite in 1:4
            key = Tuple(sort(fan.tet_nodes[[i for i in 1:4 if i!=opposite],tet]))
            triangles[key] = get(triangles,key,0)+1
        end
        @test all(count in (1,2) for count in values(triangles))
        for (key,count) in triangles
            @test (count==1) == sdwr_prism_physical_face(fan.points[:,collect(key)],domain)
        end
        for (face,owners) in enumerate(mesh.face_cells)
            @test (owners[2]==0) == sdwr_prism_physical_face(mesh.points[:,mesh.faces[face]],domain)
        end
        coefficients = [0.7,-0.3,1.1]
        affine = vec(transpose(coefficients)*mesh.points) .+ 0.4
        U = adaptive_reconstruct(mesh,fan,affine)
        @test U ≈ vec(transpose(coefficients)*fan.points) .+ 0.4 atol=3e-14
        for cell in eachindex(mesh.cells)
            ids = mesh.cells[cell]
            data = adaptive_cell_operator(mesh,fan,cell)
            @test data.P*data.P ≈ data.P atol=3e-12
            @test data.G*ones(length(ids)) ≈ zeros(3) atol=3e-12
            @test data.G*transpose(mesh.points[:,ids]) ≈ Matrix{Float64}(I,3,3) atol=3e-12
            @test data.P*affine[ids] ≈ affine[ids] atol=3e-12
            @test data.G*affine[ids] ≈ coefficients atol=3e-12
            @test dot(affine[ids],data.K*affine[ids]) ≈ mesh.volumes[cell]*sum(abs2,coefficients) atol=3e-12
            @test eigmin(Symmetric(data.K)) > -3e-12
            # Independent prism area-times-height volume, using its bottom face.
            zmin,zmax = extrema(mesh.points[3,ids])
            bottom = only(face for face in mesh.cell_faces[cell]
                if all(abs(mesh.points[3,node]-zmin)<1e-12 for node in mesh.faces[face]))
            polygon = mesh.points[:,mesh.faces[bottom]]
            area = abs(sum(polygon[1,j]*polygon[2,mod1(j+1,size(polygon,2))]-
                polygon[2,j]*polygon[1,mod1(j+1,size(polygon,2))]
                for j in axes(polygon,2)))/2
            @test mesh.volumes[cell] ≈ area*(zmax-zmin) atol=3e-13
            @test data.h ≈ cbrt(mesh.volumes[cell])
        end
        problem = manufactured_problem(domain==:cube ? :cube_smooth : :lshape_smooth)
        assembled = adaptive_assemble(mesh,fan,problem;quadrature_order=6)
        solution = adaptive_solve(assembled,mesh)
        free,boundary = solution.free_dofs,findall(mesh.boundary_vertices)
        @test solution.algebraic_residual < 3e-12
        @test norm((assembled.K*affine)[free]) < 3e-11
        recovered = copy(affine)
        recovered[free] = solution.factor\(-assembled.K[free,boundary]*affine[boundary])
        @test recovered ≈ affine atol=3e-11
        @test all(iszero,adaptive_reconstruct(mesh,fan,solution.u)[fan.boundary_nodes])
        @test isposdef(Symmetric(assembled.K[free,free]))
    end
end

@testset "Production generator determinism" begin
    for domain in (:cube,:lshape)
        first = sdwr_voronoi_prism_mesh(domain,2)
        second = sdwr_voronoi_prism_mesh(domain,2)
        @test first.points == second.points
        @test first.cells == second.cells
        @test first.faces == second.faces
        @test first.volumes == second.volumes
    end
end

@testset "Production local refinement preserves conforming scalar reconstruction" begin
    for domain in (:cube,:lshape)
        mesh = sdwr_voronoi_prism_mesh(domain,2)
        target = domain == :cube ? [0.5,0.5,0.5] : [0.0,0.0,0.5]
        for step in 1:2
            old_count = length(mesh.cells)
            selected = argmin([norm(mesh.centers[:,cell]-target) for cell in eachindex(mesh.cells)])
            mesh = sdwr_refine_prism_mesh(mesh,[selected])
            fan = adaptive_build_fan(mesh)
            @test length(mesh.cells) > old_count
            @test sum(mesh.volumes) ≈ (domain==:cube ? 1.0 : 3.0) atol=2e-10
            @test sum(fan.volumes) ≈ sum(mesh.volumes) atol=2e-10
            @test maximum(leaf.level for leaf in mesh.leaves) > minimum(leaf.level for leaf in mesh.leaves)
            triangles = Dict{NTuple{3,Int},Int}()
            for tet in axes(fan.tet_nodes,2),opposite in 1:4
                key = Tuple(sort(fan.tet_nodes[[i for i in 1:4 if i!=opposite],tet]))
                triangles[key] = get(triangles,key,0)+1
            end
            @test all(count in (1,2) for count in values(triangles))
            for (key,count) in triangles
                @test (count==1) == sdwr_prism_physical_face(fan.points[:,collect(key)],domain)
            end
            coefficients = [0.7,-0.3,1.1]
            affine = vec(transpose(coefficients)*mesh.points) .+ 0.4
            U = adaptive_reconstruct(mesh,fan,affine)
            @test U ≈ vec(transpose(coefficients)*fan.points) .+ 0.4 atol=3e-13
            for cell in eachindex(mesh.cells)
                data = adaptive_cell_operator(mesh,fan,cell)
                @test data.G*affine[mesh.cells[cell]] ≈ coefficients atol=3e-11
                @test data.P*data.P ≈ data.P atol=3e-11
            end
            problem = manufactured_problem(domain==:cube ? :cube_smooth : :lshape_smooth)
            assembled = adaptive_assemble(mesh,fan,problem;quadrature_order=6)
            solution = adaptive_solve(assembled,mesh)
            free,boundary = solution.free_dofs,findall(mesh.boundary_vertices)
            @test solution.algebraic_residual < 3e-12
            @test norm((assembled.K*affine)[free]) < 3e-11
            recovered = copy(affine)
            recovered[free] = solution.factor\(-assembled.K[free,boundary]*affine[boundary])
            @test recovered ≈ affine atol=3e-11
        end
    end
end
