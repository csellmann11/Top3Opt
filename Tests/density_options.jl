using Test, StaticArrays, TimerOutputs
include("../src/utils/general_utils.jl")
include("../src/hash_output.jl")

@testset "Density CLI and run identity" begin
    defaults = parse_commandline(String[])
    @test defaults["flux_scheme"] == :tpfa
    @test defaults["regularization"] == :gradient
    @test defaults["scaling"] == :mean && defaults["integrator"] == :explicit
    @test !defaults["diag"] && defaults["move_limit"] == Inf
    @test !defaults["projection"]
    # Sharpening settings: shared tolerances, then those of the projection and the obstacle.
    @test (defaults["sharpen_imbalance_tolerance"],defaults["sharpen_gray_tolerance"]) == (0.1,0.01)
    @test (defaults["projection_growth_rate"],defaults["projection_beta_initial"]) == (0.3,1.0)
    @test defaults["obstacle_max_weight"] == Inf
    tuned = parse_commandline(["--sharpen_imbalance_tolerance","0.2","--sharpen_gray_tolerance","0.05",
        "--projection_growth_rate","0.5","--projection_beta_initial","2","--obstacle_max_weight","8"])
    @test (tuned["sharpen_imbalance_tolerance"],tuned["sharpen_gray_tolerance"]) == (0.2,0.05)
    @test (tuned["projection_growth_rate"],tuned["projection_beta_initial"],tuned["obstacle_max_weight"]) == (0.5,2.0,8.0)
    for argv in (["--sharpen_imbalance_tolerance","0"],["--sharpen_imbalance_tolerance","1"],
            ["--sharpen_gray_tolerance","0"],["--projection_growth_rate","0"],
            ["--projection_beta_initial","Inf"],["--obstacle_max_weight","0"])
        @test_throws ArgParseError parse_commandline(argv;exc_handler=ArgParse.debug_handler)
    end
    @test defaults["filter_radius"] == 0.0
    @test defaults["density_substeps"] == 0
    @test defaults["beta0"] == 3.0 && defaults["eta0"] == 15.0
    scaled = parse_commandline(["--beta0","0.5","--eta0","30"])
    @test scaled["beta0"] == 0.5 && scaled["eta0"] == 30.0
    for scaling in ("mean","local_max","local_lambda","local_free"), diag in ("false","true"), projection in ("false","true")
        args = parse_commandline(["--scaling",scaling,"--diag",diag,"--projection",projection])
        @test args["regularization"] == :gradient && args["scaling"] == Symbol(scaling)
        @test args["diag"] == (diag == "true") && args["projection"] == (projection == "true")
    end
    for scaling in ("mean","local_max"), diag in ("false","true")
        args = parse_commandline(["--scaling",scaling,"--integrator","semi_implicit","--diag",diag])
        @test args["integrator"] == :semi_implicit
    end
    @test parse_commandline(["--regularization","helmholtz"])["regularization"] == :helmholtz
    @test parse_commandline(["--move_limit","0.05"])["move_limit"] == 0.05
    @test parse_commandline(["--move_limit","Inf"])["move_limit"] == Inf
    @test parse_commandline(["--scaling","local_lambda","--integrator","explicit"])["integrator"] == :explicit
    @test parse_commandline(["--scaling","local_free","--integrator","explicit"])["integrator"] == :explicit
    for argv in (["--regularization","helmholtz","--integrator","semi_implicit"],
            ["--regularization","helmholtz","--scaling","local_max"],
            ["--regularization","helmholtz","--diag","true"],
            ["--scaling","local_lambda","--integrator","semi_implicit"],
            ["--scaling","local_free","--integrator","semi_implicit"],
            ["--integrator","implicit"],["--scaling","helmholtz"],
            # Removed names: the former regularization values and --update_mode.
            ["--regularization","mean"],["--regularization","local_max"],["--regularization","local_lambda"],
            ["--regularization","local_free"])
        @test_throws ArgumentError parse_commandline(argv)
    end
    @test_throws ArgParseError parse_commandline(["--update_mode","implicit"];exc_handler=ArgParse.debug_handler)
    @test_throws ArgParseError parse_commandline(["--move_limit","0"];exc_handler=ArgParse.debug_handler)
    @test generate_setting_hash(defaults) == generate_setting_hash(parse_commandline(String[]))
    @test generate_setting_hash(defaults) == generate_setting_hash(parse_commandline(["--laplace_rescale","false"]))
    original = generate_setting_hash(defaults)
    # Arguments without a beta0 key get the identity of the CLI default.
    @test generate_setting_hash(delete!(copy(defaults),"beta0")) == original
    # Damping off and an infinite move limit keep the identity of the former runs.
    @test generate_setting_hash(delete!(delete!(copy(defaults),"diag"),"move_limit")) == original
    # The default (explicit, zero substeps) now takes its substeps from the measured
    # CFL limit only: it must not reuse the directory of the former fixed count.
    @test original != "s300r3mHa0gmeanp0_0b9d1f554bbc69f0"
    # Semi-implicit runs and requested substep counts keep their former identity.
    semi = generate_setting_hash(merge(copy(defaults),Dict("integrator"=>:semi_implicit)))
    requested = generate_setting_hash(merge(copy(defaults),Dict("density_substeps"=>2)))
    @test semi != original && requested != original && semi != requested
    # A finite move limit is the per-cell box now; it must not reuse the
    # directory of the former eta-increase rule (identity of ";M=0.1").
    limited = generate_setting_hash(merge(copy(defaults),Dict("move_limit"=>0.1)))
    @test limited != "s300r3mHa0gmeanp0_fa0d45a8dafcfeb9"
    @test limited != original
    @test generate_setting_hash(merge(copy(defaults),Dict("move_limit"=>0.2))) ∉ (limited,original)
    for (key,value) in ("regularization"=>:helmholtz,"scaling"=>:local_max,"integrator"=>:semi_implicit,
            "diag"=>true,"move_limit"=>0.1,
            "projection"=>true,"filter_radius"=>0.02,"density_substeps"=>2,
            "filter_tolerance"=>1e-10,"density_activity_tolerance"=>0.01,
            "solver"=>:hypre,"b_case"=>:Cantilever_sym,"beta0"=>0.5,"eta0"=>30.0)
        args = copy(defaults)
        args[key] = value
        @test generate_setting_hash(args) != original
    end
    @test length(original) < 80
    # Sharpening is imbalance based now and its mechanism follows the regularization:
    # neither mechanism reuses the directory of the former objective-based beta rule
    # (identities of ";C=stall") nor the other one's.
    sharpened(regularization;kw...) = generate_setting_hash(merge(copy(defaults),
        Dict("projection"=>true,"regularization"=>regularization),Dict(string(k)=>v for (k,v) in kw)))
    obstacle,projected = sharpened(:gradient),sharpened(:helmholtz)
    @test obstacle != "s300r3mHa0gmeanp1_88af6d621e052730"
    @test projected != "s300r3mHa0ghelmholtzp1_48e422bcff3fd595"
    @test startswith(obstacle,"s300r3mHa0gmeanp1_") && startswith(projected,"s300r3mHa0ghelmholtzp1_")
    # A changed setting of the active mechanism is a new run; one of the other is ignored.
    @test sharpened(:gradient;obstacle_max_weight=8.0) != obstacle
    @test sharpened(:gradient;sharpen_imbalance_tolerance=0.2) != obstacle
    @test sharpened(:gradient;sharpen_gray_tolerance=0.05) != obstacle
    @test sharpened(:gradient;projection_growth_rate=0.5,projection_beta_initial=2.0) == obstacle
    @test sharpened(:helmholtz;projection_growth_rate=0.5) != projected
    @test sharpened(:helmholtz;projection_beta_initial=2.0) != projected
    @test sharpened(:helmholtz;sharpen_imbalance_tolerance=0.2) != projected
    @test sharpened(:helmholtz;obstacle_max_weight=8.0) == projected
    # Without sharpening every sharpening setting is ignored.
    @test generate_setting_hash(merge(copy(defaults),Dict("obstacle_max_weight"=>8.0,
        "sharpen_imbalance_tolerance"=>0.2,"projection_growth_rate"=>0.5))) == original
    # local_free runs get their own directory, apart from local_lambda.
    @test generate_setting_hash(merge(copy(defaults),Dict("scaling"=>:local_free))) ∉
        (original,generate_setting_hash(merge(copy(defaults),Dict("scaling"=>:local_lambda))))
end
