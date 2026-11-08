using Test, StaticArrays, TimerOutputs
include("../src/utils/general_utils.jl")
include("../src/hash_output.jl")

@testset "PU-DWR options and run identity" begin
    defaults = parse_commandline(String[])
    @test defaults["error_estimator"] == :sdwr
    @test defaults["pu_marking_exponent"] == 1.0
    for estimator in ("sdwr", "pu_dwr"), q in (0.0, 1.0, 3.0)
        parsed = parse_commandline(["--error_estimator", estimator,
            "--pu_marking_exponent", string(q)])
        @test parsed["error_estimator"] == Symbol(estimator)
        @test parsed["pu_marking_exponent"] == q
    end
    for invalid in ("dwr", "recovery", "PU_DWR")
        @test_throws ArgumentError parse_commandline(["--error_estimator", invalid])
    end
    for invalid in ("-1", "NaN", "Inf", "-Inf")
        @test_throws ArgumentError parse_commandline(["--pu_marking_exponent", invalid])
    end
    default_hash = generate_setting_hash(defaults)
    @test default_hash == generate_setting_hash(parse_commandline(["--error_estimator", "sdwr"]))
    @test default_hash != generate_setting_hash(parse_commandline(["--error_estimator", "pu_dwr"]))
    @test default_hash != generate_setting_hash(parse_commandline(["--pu_marking_exponent", "3"]))
    @test length(default_hash) < 80
end
