# Structural regression checks, including changing coarse/fine connectivity.
# Run: julia --project=. --startup-file=no Tests/sparsity_pattern.jl
using Test, SparseArrays, LinearAlgebra, StaticArrays
using Ju3VEM, Ju3VEM.FixedSizeArrays
using Ju3VEM.VEMGeo: _coarsen!
include("../src/get_sparsity_pattern.jl")
include("../src/utils/mesh_processing_utils.jl")

function legacy_block_expansion(pattern, components)
    expanded = kron(pattern, sparse(ones(Bool, components, components)))
    return SparseMatrixCSC(expanded.m, expanded.n, expanded.colptr,
        expanded.rowval, zeros(Float64, nnz(expanded)))
end

function check_same_structure(actual, expected)
    @test actual isa SparseMatrixCSC{Float64,Int}
    @test size(actual) == size(expected)
    @test actual.colptr == expected.colptr
    @test actual.rowval == expected.rowval
    @test actual.nzval == expected.nzval
    @test all(iszero, actual.nzval)
end

# Assemble independent node/cell incidence through the public cell/DOF mapping,
# instead of duplicating get_sparsity_pattern's face traversal.
function reference_mesh_sparsity(cv::CellValues{D,U}) where {D,U}
    rows = Int[]
    columns = Int[]
    count = 0
    for (cell, element) in enumerate(RootIterator{4}(cv.mesh.topo))
        count = cell
        reinit!(element.id, cv)
        for node_id in cv.vnm.map.keys
            first_dof = first(get_dofs(cv.dh, node_id))
            push!(rows, (first_dof - 1) ÷ U + 1)
            push!(columns, cell)
        end
    end
    incidence = sparse(rows, columns, ones(Bool, length(rows)),
        length(cv.dh.dof_mapping), count)
    return legacy_block_expansion(incidence * incidence', U)
end

@testset "Direct CSC block expansion" begin
    # Rectangular patterns, empty columns/rows, non-count values and explicitly
    # stored zeros must retain precisely the previous structural convention.
    patterns = (
        sparse([1,3,2], [1,1,4], [2,4,7], 3,5),
        SparseMatrixCSC(2,3,Int32[1,2,2,3],Int32[2,1],[0.0,-2.0]),
        spzeros(Int, 0,0), spzeros(Int, 0,3), spzeros(Int, 3,0),
        spzeros(Int, 3,4), sparse(reshape([true], 1,1)),
    )
    for pattern in patterns, components in (1,2,3,4)
        check_same_structure(_expand_sparsity_blocks(pattern, components),
            legacy_block_expansion(pattern, components))
    end
    @test_throws ArgumentError _expand_sparsity_blocks(first(patterns), 0)

    # Equal dimensions and nnz do not imply equal connectivity.
    left = sparse([1,2,3], [1,2,3], ones(Int,3), 3,3)
    right = sparse([2,3,1], [1,2,3], ones(Int,3), 3,3)
    first_result = _expand_sparsity_blocks(left, 3)
    second_result = _expand_sparsity_blocks(right, 3)
    check_same_structure(second_result, legacy_block_expansion(right, 3))
    @test first_result.rowval != second_result.rowval
    check_same_structure(first_result, legacy_block_expansion(left, 3))
end

@testset "Uniform, hanging-face and coarsened meshes" begin
    mesh = create_rectangular_mesh(3,3,3,1.,1.,1.,StandardEl{1})
    cv = CellValues{3}(mesh)
    uniform = get_sparsity_pattern(cv)
    check_same_structure(uniform, reference_mesh_sparsity(cv))

    # Refining only the central cell also changes the neighboring coarse cells'
    # face connectivity without changing those cells' IDs.
    parent = collect(RootIterator{4}(mesh.topo))[14]
    Ju3VEM.VEMGeo._refine!(parent, mesh.topo)
    mesh = Mesh(mesh.topo, StandardEl{1}())
    cv = CellValues{3}(mesh)
    refined = get_sparsity_pattern(cv)
    check_same_structure(refined, reference_mesh_sparsity(cv))
    @test size(refined,1) > size(uniform,1)
    @test length(collect(RootIterator{4}(mesh.topo))) > 27

    child = Ju3VEM.VEMGeo.get_volumes(mesh.topo)[first(parent.childs)]
    Ju3VEM.VEMGeo._coarsen!(child, mesh.topo)
    clear_up_topo!(mesh.topo)
    mesh = Mesh(mesh.topo, StandardEl{1}())
    cv = CellValues{3}(mesh)
    coarsened = get_sparsity_pattern(cv)
    check_same_structure(coarsened, reference_mesh_sparsity(cv))
    @test length(collect(RootIterator{4}(mesh.topo))) == 27
    check_same_structure(coarsened, uniform)

    # Different refinement locations can have equal matrix dimensions and nnz.
    # The topology is still rebuilt, rather than inferred from either count.
    corner_patterns = map((1,27)) do corner
        corner_mesh = create_rectangular_mesh(3,3,3,1.,1.,1.,StandardEl{1})
        element = collect(RootIterator{4}(corner_mesh.topo))[corner]
        Ju3VEM.VEMGeo._refine!(element, corner_mesh.topo)
        corner_cv = CellValues{3}(Mesh(corner_mesh.topo, StandardEl{1}()))
        actual = get_sparsity_pattern(corner_cv)
        check_same_structure(actual, reference_mesh_sparsity(corner_cv))
        actual
    end
    @test size(corner_patterns[1]) == size(corner_patterns[2])
    @test nnz(corner_patterns[1]) == nnz(corner_patterns[2])
    @test corner_patterns[1].rowval != corner_patterns[2].rowval
end
