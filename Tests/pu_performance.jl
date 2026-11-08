include("pu_benchmark.jl")

function pu_warm_allocations(n)
    cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(n)
    for _ in 1:3
        estimate_pu_dwr!(c,u,states,ed,pars)
    end
    return minimum(@allocated(estimate_pu_dwr!(c,u,states,ed,pars)) for _ in 1:3)
end

@testset "PU-DWR scratch reuse and allocation scaling" begin
    small = pu_warm_allocations(2)
    large = pu_warm_allocations(4)
    @info "Warm PU-DWR allocations" small large
    # Small task-launch/result objects are permitted; rebuilding mesh-sized
    # dictionaries, patch matrices, or load arrays each estimate is not.
    @test small < 32_768
    @test large <= small+8_192
end

@testset "PU marker thresholds independent of vertex count" begin
    cv,states,pars = pu_fixture(2)
    indicators = zeros(length(states.χ_vec))
    indicators[1] = 1.
    markers = falses(length(get_volumes(cv.mesh.topo)))
    ref,_ = mark_elements_for_adaption(cv,indicators,states,zeros(length(indicators)),
        2,markers,false,2.,0.125;vertex_scaling=false)
    @test ref[get_el_id(states,1)]
    @test count(ref) == 1
end
