using Test, LinearAlgebra
isdefined(@__MODULE__,:sdwr_cut_tetra_rule) || include("cut_goal_quadrature.jl")

function sdwr_test_cut_moments(vertices,box)
    rule = sdwr_cut_tetra_rule(vertices,box)
    xyz = vertices*rule.barycentric
    first = xyz*rule.weights
    second = xyz*Diagonal(rule.weights)*transpose(xyz)
    return (;rule,xyz,first,second)
end

function sdwr_test_rule_admissible(vertices,box,rule)
    @test all(isfinite,rule.weights)
    @test all(>(0),rule.weights)
    @test all(rule.barycentric .>= -2e-12)
    @test vec(sum(rule.barycentric;dims=1)) ≈ ones(length(rule.weights)) atol=2e-13
    xyz = vertices*rule.barycentric
    for axis in 1:3
        @test all(xyz[axis,:] .>= box[axis][1]-2e-12)
        @test all(xyz[axis,:] .<= box[axis][2]+2e-12)
    end
    @test sum(rule.weights) ≈ rule.volume atol=1e-14
end

@testset "Exact clipped simplex moments" begin
    vertices = [0.0 1 0 0;0 0 1 0;0 0 0 1]
    for lower in ((0.0,0.0,0.0),(0.5,0.0,0.0),(0.2,0.3,0.1))
        box = ntuple(d->(lower[d],2.0),3)
        moments = sdwr_test_cut_moments(vertices,box)
        scale = 1-sum(lower)
        volume = scale^3/6
        mean = collect(lower).+scale/4
        second = [lower[i]*lower[j]+scale*(lower[i]+lower[j])/4+
            scale^2*(i==j ? 1/10 : 1/20) for i in 1:3,j in 1:3]
        @test moments.rule.volume ≈ volume rtol=2e-13
        @test moments.first ≈ volume.*mean rtol=2e-13
        @test moments.second ≈ volume.*second rtol=3e-13
        sdwr_test_rule_admissible(vertices,box,moments.rule)
    end
    full = sdwr_test_cut_moments(vertices,((-1.,2.),(-1.,2.),(-1.,2.)))
    upper = sdwr_test_cut_moments(vertices,((-1.,0.5),(-1.,2.),(-1.,2.)))
    lower = sdwr_test_cut_moments(vertices,((0.5,2.),(-1.,2.),(-1.,2.)))
    @test upper.rule.volume+lower.rule.volume ≈ full.rule.volume atol=2e-14
    @test upper.first+lower.first ≈ full.first atol=2e-14
    @test upper.second+lower.second ≈ full.second atol=2e-14
    @test !full.rule.cut
    @test size(full.rule.barycentric) == (4,4)
    for box in (((2.,3.),(0.,1.),(0.,1.)),((1.,2.),(0.,1.),(0.,1.)))
        empty = sdwr_cut_tetra_rule(vertices,box)
        @test empty.volume == 0
        @test size(empty.barycentric) == (4,0)
        @test isempty(empty.weights)
    end
    @test_throws ArgumentError sdwr_cut_tetra_rule(zeros(3,4),((0.,1.),(0.,1.),(0.,1.)))
    @test_throws ArgumentError sdwr_cut_tetra_rule(vertices,((1.,0.),(0.,1.),(0.,1.)))
end

@testset "Box moments and original-hat products" begin
    vertices = [0.0 1 0 0;0 0 1 0;0 0 0 1]
    box = ((0.05,0.2),(0.1,0.25),(0.15,0.3))
    moments = sdwr_test_cut_moments(vertices,box)
    volume = prod(bounds[2]-bounds[1] for bounds in box)
    mean = [(bounds[1]+bounds[2])/2 for bounds in box]
    second = [i==j ? (box[i][1]^2+prod(box[i])+box[i][2]^2)/3 : mean[i]*mean[j]
        for i in 1:3,j in 1:3]
    @test moments.rule.volume ≈ volume rtol=2e-13
    @test moments.first ≈ volume.*mean rtol=2e-13
    @test moments.second ≈ volume.*second rtol=3e-13
    sdwr_test_rule_admissible(vertices,box,moments.rule)
    # Original tetra hat N1=1-x-y-z and an arbitrary affine error:
    # integrate their product through analytic Cartesian box moments.
    coefficient = [0.7,-0.2,0.4]
    constant = -0.11
    expected = volume*(constant+dot(coefficient,mean)-constant*sum(mean)-
        sum(coefficient[j]*second[i,j] for i in 1:3,j in 1:3))
    error_values = constant .+ transpose(moments.xyz)*coefficient
    actual = dot(moments.rule.weights,moments.rule.barycentric[1,:].*error_values)
    @test actual ≈ expected rtol=3e-13 atol=1e-16
    nodal_values = constant .+ transpose(vertices)*coefficient
    @test transpose(moments.rule.barycentric)*nodal_values ≈ error_values atol=2e-15
    # A reversed tetra orientation changes its barycentric labeling only.
    reversed = sdwr_test_cut_moments(vertices[:,[1,3,2,4]],box)
    @test reversed.rule.volume ≈ moments.rule.volume rtol=2e-13
    @test reversed.first ≈ moments.first rtol=2e-13
    @test reversed.second ≈ moments.second rtol=3e-13
end

@testset "Partitioned cube and scaled cut geometry" begin
    permutations = ((1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1))
    for (shift,scale) in (([0.,0.,0.],[1.,1.,1.]),([2.,-1.,0.3],[0.2,1.7,0.6]))
        low,high = [0.13,0.21,0.34],[0.73,0.82,0.91]
        box = ntuple(d->(shift[d]+scale[d]*low[d],shift[d]+scale[d]*high[d]),3)
        volume,first,second = 0.0,zeros(3),zeros(3,3)
        for permutation in permutations
            coordinates = zeros(3,4)
            coordinates[permutation[1],2] = 1
            coordinates[:,3] = coordinates[:,2]
            coordinates[permutation[2],3] = 1
            coordinates[:,4] .= 1
            vertices = shift .+ scale.*coordinates
            moments = sdwr_test_cut_moments(vertices,box)
            volume += moments.rule.volume
            first += moments.first
            second += moments.second
            sdwr_test_rule_admissible(vertices,box,moments.rule)
        end
        expected_volume = prod(bounds[2]-bounds[1] for bounds in box)
        mean = [(bounds[1]+bounds[2])/2 for bounds in box]
        expected_second = [i==j ? (box[i][1]^2+prod(box[i])+box[i][2]^2)/3 : mean[i]*mean[j]
            for i in 1:3,j in 1:3]
        @test volume ≈ expected_volume rtol=5e-13
        @test first ≈ expected_volume.*mean rtol=5e-13
        @test second ≈ expected_volume.*expected_second rtol=5e-13
    end
end

@testset "Localized goal wrapper" begin
    vertices = [0.0 1 0 0;0 0 1 0;0 0 0 1]
    goal = (;name=:localized_u,power=1,box=((0.,0.2),(0.,0.2),(0.,0.2)))
    @test sdwr_goal_tetra_rule(vertices,goal).volume ≈ 0.2^3 rtol=2e-13
    @test_throws ArgumentError sdwr_goal_tetra_rule(vertices,merge(goal,(;power=2)))
    fan = (;points=vertices,tet_nodes=reshape(1:4,4,1),volumes=[1/6])
    rules = sdwr_cut_goal_rules(fan,goal)
    @test length(rules) == 1
    @test rules[1].volume ≈ 0.2^3 rtol=2e-13
end

@testset "Oblique tetrahedra and cut additivity" begin
    # All four tetrahedron faces are oblique; the box is strictly inside it.
    vertices = [2. 2 -2 -2;2 -2 2 -2;2 -2 -2 2]
    box = ((-0.1,0.2),(-0.2,0.15),(-0.15,0.1))
    moments = sdwr_test_cut_moments(vertices,box)
    volume = prod(bounds[2]-bounds[1] for bounds in box)
    mean = [(bounds[1]+bounds[2])/2 for bounds in box]
    second = [i==j ? (box[i][1]^2+prod(box[i])+box[i][2]^2)/3 : mean[i]*mean[j]
        for i in 1:3,j in 1:3]
    @test moments.rule.volume ≈ volume rtol=5e-13
    @test moments.first ≈ volume.*mean rtol=5e-13 atol=1e-16
    @test moments.second ≈ volume.*second rtol=5e-13 atol=1e-16
    sdwr_test_rule_admissible(vertices,box,moments.rule)
    # Clipping commutes with splitting an irregular tetrahedron into four
    # children; each child has its own barycentric coordinate system.
    vertices = [0.1 1.2 -0.1 0.2;0.2 0.1 1.3 0.1;-0.1 0.2 0.3 1.4]
    box = ((0.15,0.65),(0.2,0.8),(0.05,0.7))
    whole = sdwr_test_cut_moments(vertices,box)
    center = vertices*[0.13,0.27,0.21,0.39]
    volume,first,second = 0.0,zeros(3),zeros(3,3)
    for missing in 1:4
        child = hcat(center,vertices[:,[i for i in 1:4 if i != missing]])
        moments = sdwr_test_cut_moments(child,box)
        volume += moments.rule.volume
        first += moments.first
        second += moments.second
        sdwr_test_rule_admissible(child,box,moments.rule)
    end
    @test volume ≈ whole.rule.volume rtol=5e-13
    @test first ≈ whole.first rtol=5e-13
    @test second ≈ whole.second rtol=5e-13
end
