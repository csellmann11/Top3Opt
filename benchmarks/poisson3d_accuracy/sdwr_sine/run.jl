include("../sdwr_adaptive/run.jl")
include("cases.jl")
include("uniform.jl")

"""Evaluate the unchanged projected sDWR through its exact global identity.

All unknowns are obtained by sparse primal/adjoint solves. Local matrices
represent the same continuous P1 fan as the original benchmark. Reusing the
uniform element tensors avoids constructing millions of identical tetrahedra.
The original PU-integrated estimator is checked independently in validate.jl.
"""
function sine_uniform_evaluate(system,reference,problem,source,u,name)
    h = system.h
    if name == :integral_u
        rhs = sine_uniform_load(system,reference.lT,h^3)
        reconstructed = dot(rhs,u)
        correction = dot(sine_uniform_load(system,reference.lP-reference.lT,h^3),u)/2
    elseif name == :integral_u2
        mass_u = sine_uniform_apply(system,reference.M,h^3,u)
        rhs = 2mass_u
        reconstructed = dot(u,mass_u)
        correction = dot(u,sine_uniform_apply(system,reference.B-reference.M,h^3,u))
    elseif name == :energy
        rhs = sine_uniform_apply(system,reference.H,h,u)
        reconstructed = dot(u,rhs)/2
        correction = dot(u,sine_uniform_apply(system,reference.C-reference.H,h,u))/2
    else
        throw(ArgumentError("unknown sine goal $name"))
    end
    z = sine_uniform_solve(system,rhs).u
    stabilization = dot(u,sine_uniform_apply(system,reference.S,h,z))
    load_correction = dot(source.reconstructed-source.projected,z)/2
    estimate = stabilization+load_correction+correction
    exact = sine_reference(problem,name)
    error = exact-reconstructed
    free = system.free
    dual_residual = norm((system.K*z-rhs)[free])/max(norm(rhs[free]),eps())
    return (;z,rhs,exact,reconstructed,error,estimate,stabilization,load_correction,
        goal_correction=correction,dual_residual)
end

function run_sine_sweep(;modes=((1,1,1),(3,1,1),(5,1,1)),
        levels=(8,12,16,24,32,48),goals=(:integral_u,:integral_u2,:energy),order=8,
        output=joinpath(@__DIR__,"..","results","sdwr_sine"))
    BLAS.set_num_threads(1)
    mkpath(output)
    open(joinpath(output,"settings.toml"),"w") do io
        TOML.print(io,Dict("modes"=>[collect(mode) for mode in modes],
            "levels"=>collect(levels),"goals"=>string.(collect(goals)),"quadrature_order"=>order,
            "pde"=>"-Delta u = pi^2*(m^2+n^2+p^2)*u; zero Dirichlet; unit cube",
            "target"=>"J(u)-J(U); U is original continuous P1 fan reconstruction",
            "operator"=>"original alpha=h VEM-style operator and projected primal load",
            "estimator"=>"unchanged affine projected sDWR; exact global algebraic reduction",
            "solve"=>"actual sparse Cholesky primal and adjoint solves; shared uniform tensors",
            "production_changes"=>false,"patch_solves"=>false))
    end
    reference = sine_uniform_reference(;order)
    rows = NamedTuple[]
    for n in levels
        println("SINE_PREPARING ",(;n,cells=n^3)); flush(stdout)
        system = sine_uniform_system(n,reference)
        for mode in modes
            problem = sine_problem(mode)
            source = sine_uniform_source(system,reference,mode)
            u = sine_uniform_solve(system,source.projected).u
            free = system.free
            residual = norm((system.K*u-source.projected)[free])/
                max(norm(source.projected[free]),eps())
            nodal_exact = [problem.exact(view(system.points,:,i)) for i in axes(system.points,2)]
            amplitude = dot(u,nodal_exact)/dot(nodal_exact,nodal_exact)
            mode_defect = norm(u-amplitude*nodal_exact)/max(norm(u),eps())
            for name in goals
                evaluated = sine_uniform_evaluate(system,reference,problem,source,u,name)
                row = (;problem=String(problem.name),m=mode[1],k=mode[2],p=mode[3],goal=String(name),
                    n,cells=n^3,vertices=(n+1)^3,free_dofs=length(free),
                    exact_goal=evaluated.exact,reconstructed_goal=evaluated.reconstructed,
                    signed_goal_error=evaluated.error,
                    relative_goal_error=abs(evaluated.error/evaluated.exact),
                    signed_estimate=evaluated.estimate,
                    signed_effectivity=evaluated.estimate/evaluated.error,
                    stabilization=evaluated.stabilization,load_correction=evaluated.load_correction,
                    goal_correction=evaluated.goal_correction,primal_residual=residual,
                    dual_residual=evaluated.dual_residual,nodal_amplitude=amplitude,
                    relative_nonsine_component=mode_defect,quadrature_order=order)
                push!(rows,row)
                write_csv(joinpath(output,"accuracy.csv"),rows)
                println("SINE ",(;mode,goal=name,n,relative_error=row.relative_goal_error,
                    effectivity=row.signed_effectivity)); flush(stdout)
            end
        end
        # Release each sparse factor before constructing the next mesh.
        system = nothing
        GC.gc()
    end
    return rows
end

function sine_sweep_cli(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument,"--") && occursin('=',argument) || error("Use --key=value")
        key,value = split(argument[3:end],'=';limit=2)
        key in ("levels","modes","goals","order","output") || error("Unknown option $key")
        options[key] = value
    end
    modes = Tuple(Tuple(parse.(Int,split(mode,','))) for mode in
        split(get(options,"modes","1,1,1;3,1,1;5,1,1"),';'))
    return run_sine_sweep(;modes,
        levels=Tuple(parse.(Int,split(get(options,"levels","8,12,16,24,32,48"),','))),
        goals=Tuple(Symbol.(split(get(options,"goals","integral_u,integral_u2,energy"),','))),
        order=parse(Int,get(options,"order","8")),
        output=get(options,"output",joinpath(@__DIR__,"..","results","sdwr_sine")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    sine_sweep_cli(ARGS)
end
