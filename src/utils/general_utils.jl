using ArgParse 


function def_rhs_fun(x)
    SA[0.0, 0.0, 0.0]
end
pu3d_body_iszero(::typeof(def_rhs_fun)) = true

function parse_commandline(argv=ARGS)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--max_opt_steps","-s"
            help = "Maximum number of optimization steps"
            arg_type = Int
            default  = 300
        "--max_ref_level","-r"
            help = "Maximum refinement level"
            arg_type = Int
            default  = 3
        "--mesh_type","-m"
            help = "Mesh type"
            arg_type = Symbol
            default  = :Hexahedra # :Hexahedra # :Voronoi
        "--do_adaptivity","-a"
            help = "Boolean to determine if adaptivity is enabled"
            arg_type = Bool
            default  = false
        "--do_adaptivity_at_the_start","-b"
            help = "if true, the mesh is refined to the finest level at the start"
            arg_type = Bool
            default  = true
        "--b_case","-c"
            help = "Case name"
            arg_type = Symbol
            default  = :MBB_sym #[:simple_lever,:MBB_sym,:Cantilever_sym,:Bending_Beam_sym,:pressure_plate,:L_cantilever]
        "--density_marking","-d"
            help = "Toogle if elements are marked for refinement if density is growing"
            arg_type = Bool
            default  = true
        "--error_estimator"
            help = "Adaptive error indicator: sdwr or pu_dwr"
            arg_type = Symbol
            default = :sdwr
        "--pu_marking_exponent"
            help = "PU-DWR marking weight: cell volume times physical density to this power"
            arg_type = Float64
            default = 1.0
        "--laplace_rescale","-l"
            help = "Deprecated compatibility option; TPFA does not use Laplace rescaling"
            arg_type = Bool
            default  = true
        "--flux_scheme"
            help = "Face-gradient approximation (tpfa only)"
            arg_type = Symbol
            default = :tpfa
            range_tester = x -> x == :tpfa
        "--regularization"
            help = "Density regularization: mean, local_max, or helmholtz"
            arg_type = Symbol
            default = :mean
            range_tester = x -> x in (:mean, :local_max, :helmholtz)
        "--update_mode"
            help = "Density update: explicit or implicit (implicit is available for gradient penalties only)"
            arg_type = Symbol
            default = :explicit
            range_tester = x -> x in (:explicit, :implicit)
        "--projection"
            help = "Enable volume-preserving smooth projection with automatic continuation"
            arg_type = Bool
            default = false
        "--beta0"
            help = "Regularization strength beta0; the per-cell coefficient is 2*beta0*h_i^2"
            arg_type = Float64
            default = 1.0
            range_tester = x -> isfinite(x) && x >= 0
        "--eta0"
            help = "Viscosity eta0 of the density update; larger values give smaller density steps"
            arg_type = Float64
            default = 15.0
            range_tester = x -> isfinite(x) && x > 0
        "--filter_radius"
            help = "Uniform Helmholtz radius override; zero uses r²_i = 2*beta0*h_i^2 like the gradient penalties"
            arg_type = Float64
            default = 0.0
            range_tester = x -> isfinite(x) && x >= 0
        "--density_substeps"
            help = "Requested density substeps; zero uses the legacy gradient count or one Helmholtz design step"
            arg_type = Int
            default = 0
            range_tester = x -> x >= 0
        "--filter_tolerance"
            help = "Relative tolerance for Helmholtz forward/adjoint solves"
            arg_type = Float64
            default = 1e-12
            range_tester = x -> isfinite(x) && 0 < x < 1
        "--density_activity_tolerance"
            help = "Absolute physical-density change used by adaptive mesh marking"
            arg_type = Float64
            default = 1e-3
            range_tester = x -> isfinite(x) && 0 <= x <= 1
        "--rhs_fun","-f"
            help = "Right hand side function"
            arg_type = Function
            default  = def_rhs_fun 
        "--solver"
            help = "Linear solver backend: petsc (CG + GAMG with rigid-body near-null space) or hypre (PCG + BoomerAMG)"
            arg_type = Symbol
            default  = :petsc
    end
    args = parse_args(argv,s)
    args["error_estimator"] in (:sdwr, :pu_dwr) ||
        throw(ArgumentError("error_estimator must be sdwr or pu_dwr"))
    isfinite(args["pu_marking_exponent"]) && args["pu_marking_exponent"] >= 0 ||
        throw(ArgumentError("pu_marking_exponent must be finite and nonnegative"))
    args["regularization"] == :helmholtz && args["update_mode"] != :explicit &&
        throw(ArgumentError("Helmholtz uses an explicit design update; choose --update_mode explicit"))
    return args
end


function get_simulation_times(to::TimerOutput)
    sim_times = SimulationTimes() 
    sim_times.solve_time = TimerOutputs.time(to["compute_displacement"]["solver"])/(1e09)
    sim_times.assembly_time = TimerOutputs.time(to["compute_displacement"]["assembly"])/(1e09)
    sim_times.state_update_time = TimerOutputs.time(to["state_update"])/(1e09)
    try
        sim_times.adaptivity_time = TimerOutputs.time(to["adaptivity"])/(1e09)
    catch
        sim_times.adaptivity_time = 0.0
    end
    return sim_times
end
