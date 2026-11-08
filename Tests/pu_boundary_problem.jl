isdefined(@__MODULE__, :run_optimization) || include("regularization_benchmark.jl")

@testset "PU-DWR production line load and isolated supports" begin
    mesh = create_rectangular_mesh(6,2,2,3.,0.5,1.,StandardEl{1})
    lambda,mu = E_ν_to_lame(210.e3,0.33)
    pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lambda,mu,1.)),lambda,mu,1e-3,15.,1.,0.3)
    result = run_optimization(mesh,ZeroBodyForce(),pars;
        vtk_folder_name=joinpath(mktempdir(),"bending_beam"),
        MAX_OPT_STEPS=2,MAX_REF_LEVEL=1,error_estimator=:pu_dwr,
        b_case=:Bending_Beam_sym,do_adaptivity=false,take_snapshots_at=[1])
    @test length(result.error_history) == 2
    @test all(e -> isfinite(e.signed) && e.magnitude > 0 && abs(e.signed) <= e.magnitude,
        result.error_history)
    @test isfinite(result.final_strain_energy) && result.final_strain_energy > 0
end
