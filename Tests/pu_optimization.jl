include("pu_dwr.jl")
include("pu_stage_profile.jl")

@testset "Exact optimized patches agree with original implementation" begin
    for n in (1,2), hanging in (false,true)
        cv,states,pars,ch,loads,u,K,ed,c=pu_clamped_problem(n;hanging)
        old=reference_estimate_pu!(c,u,states,ed,pars)
        old_nodal,old_indicators,old_correction=copy(old.nodal),copy(old.indicators),copy(old.element_correction)
        old_signed=old.signed
        new=estimate_pu_dwr!(c,u,states,ed,pars)
        @test new.nodal ≈ old_nodal rtol=2e-10 atol=1e-17
        @test new.indicators ≈ old_indicators rtol=2e-10 atol=1e-17
        @test new.element_correction == old_correction
        @test new.signed ≈ old_signed rtol=2e-10 atol=1e-17
        for node in eachindex(c.nodal)
            w=first(c.workspaces)
            reference_pu3d_patch_number!(w,c,node)
            old_count=w.ndofs
            pu3d_patch_number!(w,c,node)
            @test w.ndofs == old_count
        end
    end
end
