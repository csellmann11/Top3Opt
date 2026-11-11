using Test, StaticArrays, TimerOutputs
include("../src/utils/general_utils.jl")
include("../src/hash_output.jl")

@testset "Density CLI and run identity" begin
    defaults = parse_commandline(String[])
    @test defaults["flux_scheme"] == :tpfa
    @test defaults["regularization"] == :mean
    @test !defaults["projection"]
    @test defaults["filter_radius"] == 0.0
    @test defaults["density_substeps"] == 0
    @test defaults["beta0"] == 1.0 && defaults["eta0"] == 15.0
    scaled = parse_commandline(["--beta0","0.5","--eta0","30"])
    @test scaled["beta0"] == 0.5 && scaled["eta0"] == 30.0
    for regularization in ("mean","local_max","helmholtz"), projection in ("false","true")
        args = parse_commandline(["--regularization",regularization,"--projection",projection])
        @test args["regularization"] == Symbol(regularization)
        @test args["projection"] == (projection == "true")
    end
    for regularization in ("mean","local_max")
        args = parse_commandline(["--regularization",regularization,"--update_mode","implicit"])
        @test args["update_mode"] == :implicit
    end
    @test_throws ArgumentError parse_commandline(["--regularization","helmholtz","--update_mode","implicit"])
    @test generate_setting_hash(defaults) == generate_setting_hash(parse_commandline(String[]))
    @test generate_setting_hash(defaults) == generate_setting_hash(parse_commandline(["--laplace_rescale","false"]))
    original = generate_setting_hash(defaults)
    for (key,value) in ("regularization"=>:local_max,"update_mode"=>:implicit,
            "projection"=>true,"filter_radius"=>0.02,"density_substeps"=>2,
            "filter_tolerance"=>1e-10,"density_activity_tolerance"=>0.01,
            "solver"=>:hypre,"b_case"=>:Cantilever_sym,"beta0"=>0.5,"eta0"=>30.0)
        args = copy(defaults)
        args[key] = value
        @test generate_setting_hash(args) != original
    end
    @test length(original) < 80
end
