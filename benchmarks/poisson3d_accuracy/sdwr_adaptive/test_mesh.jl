using Test
include("load.jl")

@testset "Adaptive face conformity and polynomial consistency" begin
    for domain in (:cube,:lshape)
        leaves = adaptive_initial_leaves(domain,2)
        leaves = adaptive_refine(leaves,[1])
        leaves = adaptive_refine(leaves,[1,2])
        mesh = adaptive_build_mesh(leaves;domain,n=2)
        fan = adaptive_build_fan(mesh)
        @test sum(mesh.volumes) ≈ (domain == :cube ? 1.0 : 3.0)
        @test sum(fan.volumes) ≈ sum(mesh.volumes)
        @test minimum(fan.volumes) > 0
        @test maximum(length,mesh.cells) > 8
        @test maximum(length,mesh.faces) > 4
        # Every triangular face is shared exactly twice or is a physical face.
        triangle_counts = Dict{NTuple{3,Int},Int}()
        for tet in axes(fan.tet_nodes,2),opposite in 1:4
            face = Tuple(sort(fan.tet_nodes[[i for i in 1:4 if i!=opposite],tet]))
            triangle_counts[face] = get(triangle_counts,face,0)+1
        end
        expected_boundary = Set{NTuple{3,Int}}()
        for (face,owners) in enumerate(mesh.face_cells)
            owners[2] == 0 || continue
            ids = mesh.faces[face]
            for i in eachindex(ids)
                push!(expected_boundary,Tuple(sort([ids[i],ids[mod1(i+1,length(ids))],
                    fan.vertex_count+face])))
            end
        end
        @test all(count in (1,2) for count in values(triangle_counts))
        @test Set(face for (face,count) in triangle_counts if count==1) == expected_boundary
        coefficients = [0.7,-0.3,1.1]
        u = vec(transpose(coefficients)*mesh.points) .+ 0.4
        U = adaptive_reconstruct(mesh,fan,u)
        @test U ≈ vec(transpose(coefficients)*fan.points) .+ 0.4 atol=3e-15
        for cell in eachindex(mesh.cells)
            ids = mesh.cells[cell]
            data = adaptive_cell_operator(mesh,fan,cell)
            @test data.P*data.P ≈ data.P atol=2e-13
            @test data.G*ones(length(ids)) ≈ zeros(3) atol=3e-13
            @test data.G*transpose(mesh.points[:,ids]) ≈ Matrix{Float64}(I,3,3) atol=3e-13
            @test data.P*u[ids] ≈ u[ids] atol=2e-13
            @test data.G*u[ids] ≈ coefficients atol=2e-13
            @test dot(u[ids],data.K*u[ids]) ≈ mesh.volumes[cell]*sum(abs2,coefficients) atol=2e-13
            @test eigmin(Symmetric(data.K)) > -3e-13
            average_gradient = zeros(3)
            for tet in fan.cell_tets[cell]
                @test fan.gradients[tet]*U[fan.tet_nodes[:,tet]] ≈ coefficients atol=3e-13
                average_gradient .+= fan.volumes[tet] .* (fan.gradients[tet]*U[fan.tet_nodes[:,tet]])
            end
            @test average_gradient ≈ mesh.volumes[cell] .* (data.G*u[ids]) atol=2e-14
        end
        problem = manufactured_problem(domain == :cube ? :cube_smooth : :lshape_smooth)
        assembled = adaptive_assemble(mesh,fan,problem;quadrature_order=4)
        solution = adaptive_solve(assembled,mesh)
        # Global affine Dirichlet patch test: interior fluxes must cancel even
        # at the nonuniform coarse/fine interfaces.
        free,boundary = solution.free_dofs,findall(mesh.boundary_vertices)
        @test norm((assembled.K*u)[free]) < 2e-12
        affine_solution = copy(u)
        affine_solution[free] = solution.factor\(-assembled.K[free,boundary]*u[boundary])
        @test affine_solution ≈ u atol=2e-12
        fan_load = sin.(collect(1:length(U)))
        @test dot(fan_load,U) ≈ dot(sdwr_reconstruction_transpose(mesh,fan,fan_load),u) atol=2e-12
        @test solution.algebraic_residual < 1e-12
        @test all(iszero,solution.u[mesh.boundary_vertices])
        reconstruction = adaptive_reconstruct(mesh,fan,solution.u)
        @test all(iszero,reconstruction[fan.boundary_nodes])
        doubled = adaptive_solve(merge(assembled,(;rhs=2 .* assembled.rhs)),mesh;factor=solution.factor)
        @test doubled.u ≈ 2 .* solution.u
    end
end

@testset "Unrefined adaptive operator agrees with original cube benchmark" begin
    for domain in (:cube,:lshape)
        oldmesh = build_mesh(domain,2)
        oldfan = build_fan(oldmesh)
        mesh = adaptive_build_mesh(adaptive_initial_leaves(domain,2);domain,n=2)
        fan = adaptive_build_fan(mesh)
        @test length(mesh.cells) == length(oldmesh.cells)
        @test size(mesh.points,2) == size(oldmesh.points,2)
        @test length(fan.volumes) == length(oldfan.volumes)
        for cell in eachindex(mesh.cells)
            oldcell = findfirst(c->isapprox(oldmesh.centers[:,c],mesh.centers[:,cell]),eachindex(oldmesh.cells))
            oldids,newids = oldmesh.cells[oldcell],mesh.cells[cell]
            permutation = [findfirst(j->oldmesh.points[:,oldids[j]]==mesh.points[:,node],eachindex(oldids))
                for node in newids]
            original = poisson_cell_operator(oldmesh,oldcell)
            adaptive = adaptive_cell_operator(mesh,fan,cell)
            @test adaptive.G ≈ original.G[:,permutation] atol=2e-14
            @test adaptive.P ≈ original.P[permutation,permutation] atol=2e-14
            @test adaptive.K ≈ original.K[permutation,permutation] atol=2e-14
        end
        problem = manufactured_problem(domain == :cube ? :cube_smooth : :lshape_smooth)
        # Order six integrates the L-domain polynomial load exactly; lower
        # rules retain orientation-dependent quadrature error on reordered tets.
        oldsolution = solve_primal(oldmesh,oldfan,problem;quadrature_order=6)
        solution = adaptive_solve(adaptive_assemble(mesh,fan,problem;quadrature_order=6),mesh)
        permutation = [findfirst(i->oldmesh.points[:,i]==mesh.points[:,j],axes(oldmesh.points,2))
            for j in axes(mesh.points,2)]
        @test solution.u ≈ oldsolution.u[permutation] atol=2e-14
        @test solution.rhs ≈ oldsolution.rhs[permutation] atol=2e-14
        @test solution.K ≈ oldsolution.K[permutation,permutation] atol=2e-14
    end
end
