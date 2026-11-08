using CairoMakie, CSV, Printf

const SDWR_PLOT_PROBLEMS = (:cube_smooth, :lshape_smooth, :lshape_singular)
const SDWR_PLOT_GOALS = (:integral_u, :integral_u2, :localized_u)
const SDWR_PLOT_PROBLEM_NAMES = ("Cube: smooth", "L-beam: smooth", "L-beam: singular")
const SDWR_PLOT_GOAL_NAMES = ("Integral of u", "Integral of u²", "Localized integral of u")

function sdwr_plot_value(row, names; default=NaN)
    for name in names
        hasproperty(row, name) || continue
        value = getproperty(row, name)
        (ismissing(value) || isempty(string(value))) && continue
        return value isa Number ? Float64(value) : parse(Float64, string(value))
    end
    return Float64(default)
end

"""Normalize the old and new CSV schemas; preserve every recorded mesh point."""
function sdwr_plot_rows(path, series)
    isfile(path) || error("Missing convergence data: $path")
    rows = NamedTuple[]
    for row in CSV.File(path)
        problem, goal = Symbol(row.problem), Symbol(row.goal)
        problem in SDWR_PLOT_PROBLEMS || error("Unknown problem $problem")
        goal in SDWR_PLOT_GOALS || error("Unknown goal $goal")
        dofs = sdwr_plot_value(row, (:free_dofs,))
        effectivity = sdwr_plot_value(row, (:signed_effectivity, :effectivity))
        dofs > 0 || error("A logarithmic DOF axis requires positive free_dofs")
        isfinite(effectivity) || error("Nonfinite effectivity in $path for $problem/$goal")
        relative_error = sdwr_plot_value(row, (:relative_goal_error,))
        isfinite(relative_error) && relative_error >= 0 || error("Invalid relative goal error")
        push!(rows, (;series=String(series), problem=String(problem), goal=String(goal),
            free_dofs=Int(dofs),
            total_vertex_dofs=sdwr_plot_value(row, (:vertices,)),
            cells=Int(sdwr_plot_value(row, (:cells,))),
            n=sdwr_plot_value(row, (:n, :base_n)),
            cycle=sdwr_plot_value(row, (:cycle,)),
            layers=sdwr_plot_value(row, (:layers,)),
            signed_effectivity=effectivity, relative_goal_error=relative_error,
            signed_goal_error=sdwr_plot_value(row, (:signed_goal_error,)),
            signed_estimate=sdwr_plot_value(row, (:signed_estimate, :eta)),
            source_file=abspath(path)))
    end
    return rows
end

function sdwr_plot_series(uniform, adaptive, voronoi, adaptive_prism, combined)
    specifications = [
        (;name=:uniform_hex, label="Uniform hex", color="#0072B2", marker=:circle,
            linestyle=:solid, path=uniform),
        (;name=:adaptive_hex, label="Adaptive hex: separated", color="#D55E00", marker=:rect,
            linestyle=:solid, path=adaptive)]
    if voronoi !== nothing
        push!(specifications, (;name=:voronoi_prism, label="Voronoi prisms", color="#009E73",
            marker=:diamond, linestyle=:solid, path=voronoi))
    end
    if adaptive_prism !== nothing
        push!(specifications, (;name=:adaptive_prism, label="Adaptive prism-derived", color="#CC79A7",
            marker=:star5, linestyle=:dash, path=adaptive_prism))
    end
    if combined !== nothing
        push!(specifications, (;name=:adaptive_hex_combined, label="Adaptive hex: combined",
            color="#777777", marker=:utriangle, linestyle=:dot, path=combined))
    end
    return specifications
end

"""All nine panels use the same axes and show every actual sampled mesh."""
function sdwr_convergence_figure(rows, specifications; metric=:effectivity, title=nothing)
    metric in (:effectivity, :relative_error) || error("Unknown plot metric")
    effectivity = metric == :effectivity
    figure = Figure(size=(1520, 1160), fontsize=18, backgroundcolor=:white)
    title === nothing && (title = effectivity ? "sDWR effectivity convergence" : "sDWR goal-error convergence")
    Label(figure[1, 1:3], title; fontsize=27, font=:bold, tellwidth=false)
    elements = [[LineElement(color=spec.color, linestyle=spec.linestyle, linewidth=2.4),
        MarkerElement(color=spec.color, marker=spec.marker, markersize=11)]
        for spec in specifications]
    Legend(figure[2, 1:3], elements, [spec.label for spec in specifications];
        orientation=:horizontal, framevisible=false, labelsize=16, colgap=23)
    series_names = Set(String(spec.name) for spec in specifications)
    plot_rows = filter(row->row.series in series_names, rows)
    max_dofs = maximum(row.free_dofs for row in plot_rows)
    xmin, xmax = 0.85, max_dofs*1.18
    if effectivity
        values = [row.signed_effectivity for row in plot_rows]
        lower, upper = min(0.0, minimum(values)), max(1.2, maximum(values))
        pad = 0.06*(upper-lower)
        ymin, ymax = lower-pad, upper+pad
    else
        positive = [row.relative_goal_error for row in plot_rows if row.relative_goal_error > 0]
        isempty(positive) && error("No positive errors for logarithmic error plot")
        ymin, ymax = minimum(positive)/1.7, maximum(positive)*1.7
    end
    x_ticks = 10.0 .^ (0:ceil(Int,log10(xmax)))
    for (i, problem) in enumerate(SDWR_PLOT_PROBLEMS)
        Label(figure[i+2, 0], SDWR_PLOT_PROBLEM_NAMES[i]; rotation=pi/2,
            fontsize=20, font=:bold, tellheight=false)
        for (j, goal) in enumerate(SDWR_PLOT_GOALS)
            axis = Axis(figure[i+2, j];
                title=i == 1 ? SDWR_PLOT_GOAL_NAMES[j] : "",
                xlabel=i == 3 ? "Unknown DOFs after\nDirichlet constraints" : "",
                ylabel=j == 1 ? (effectivity ? "Signed effectivity" : "Relative goal error") : "",
                xscale=log10, yscale=effectivity ? identity : log10,
                xticks=(x_ticks, [@sprintf("%.0f",x) for x in x_ticks]),
                xgridcolor=(:black,0.09), ygridcolor=(:black,0.09),
                topspinevisible=false, rightspinevisible=false,
                xticklabelsvisible=i == 3, yticklabelsvisible=j == 1)
            effectivity && hlines!(axis, [1.0]; color=(:black,0.65), linewidth=1.3, linestyle=:dash)
            for spec in specifications
                selected = sort([row for row in plot_rows if row.series == String(spec.name) &&
                    row.problem == String(problem) && row.goal == String(goal)]; by=row->row.free_dofs)
                isempty(selected) && continue
                x = [row.free_dofs for row in selected]
                y = effectivity ? [row.signed_effectivity for row in selected] :
                    [row.relative_goal_error > 0 ? row.relative_goal_error : NaN for row in selected]
                lines!(axis, x, y; color=spec.color, linestyle=spec.linestyle, linewidth=2.3)
                scatter!(axis, x, y; color=spec.color, marker=spec.marker,
                    markersize=9, strokecolor=:white, strokewidth=0.7)
            end
            xlims!(axis, xmin, xmax)
            ylims!(axis, ymin, ymax)
        end
    end
    footer = effectivity ?
        "η / [J(u) − J(U)]   •   Every marker is a computed mesh; lines only connect samples   •   Dashed reference: 1" :
        "|J(u) − J(U)| / |J(u)|   •   Every marker is a computed mesh; lines only connect samples"
    footer *= "\nDOFs count one scalar field; prescribed boundary values are excluded."
    Label(figure[6, 1:3], footer; fontsize=15, color="#444444", tellwidth=false)
    if any(spec->spec.name == :adaptive_hex_combined, specifications)
        Label(figure[7, 1:3], "Combined marking was tested only for the cube and singular L-beam.";
            fontsize=14, color="#555555", tellwidth=false)
    end
    rowgap!(figure.layout, 18)
    colgap!(figure.layout, 24)
    return figure
end

function plot_sdwr_convergence(;uniform=joinpath(@__DIR__,"..","results","sdwr_uniform_control.csv"),
        uniform_fine=nothing,
        adaptive=joinpath(@__DIR__,"..","results","sdwr_adaptive","accuracy.csv"),
        voronoi=nothing, adaptive_prism=nothing,
        combined=joinpath(@__DIR__,"..","results","sdwr_adaptive_combined","accuracy.csv"),
        output=joinpath(@__DIR__,"..","results","sdwr_convergence"))
    specifications = sdwr_plot_series(uniform,adaptive,voronoi,adaptive_prism,combined)
    rows = reduce(vcat,[sdwr_plot_rows(spec.path,spec.name) for spec in specifications])
    uniform_fine === nothing || append!(rows,sdwr_plot_rows(uniform_fine,:uniform_hex))
    mkpath(output)
    CSV.write(joinpath(output,"all_points.csv"),rows)
    main_specs = filter(spec->spec.name != :adaptive_hex_combined,specifications)
    for metric in (:effectivity,:relative_error)
        figure = sdwr_convergence_figure(rows,main_specs;metric)
        save(joinpath(output,"$(metric).png"),figure;px_per_unit=1.5)
        save(joinpath(output,"$(metric).svg"),figure)
    end
    if combined !== nothing
        comparison_specs = filter(spec->spec.name in
            (:uniform_hex,:adaptive_hex,:adaptive_hex_combined),specifications)
        figure = sdwr_convergence_figure(rows,comparison_specs;
            title="sDWR effectivity: hex marking comparison")
        save(joinpath(output,"effectivity_marking_comparison.png"),figure;px_per_unit=1.5)
        save(joinpath(output,"effectivity_marking_comparison.svg"),figure)
    end
    for spec in specifications
        count = sum(row.source_file == abspath(spec.path) for row in rows)
        println("PLOT_INPUT ", (;series=spec.name,rows=count,path=abspath(spec.path)))
    end
    if uniform_fine !== nothing
        count = sum(row.source_file == abspath(uniform_fine) for row in rows)
        println("PLOT_INPUT ", (;series=:uniform_hex,rows=count,path=abspath(uniform_fine)))
    end
    println("PLOT_OUTPUT ",abspath(output))
    return (;rows,output=abspath(output))
end

function sdwr_plot_cli(args)
    options = Dict{Symbol,String}()
    for argument in args
        startswith(argument,"--") && occursin('=',argument) || error("Use --key=value arguments")
        key,value = split(argument[3:end],'=';limit=2)
        key = replace(key,"-"=>"_")
        Symbol(key) in (:uniform,:uniform_fine,:adaptive,:voronoi,:adaptive_prism,:combined,:output) || error("Unknown option $key")
        options[Symbol(key)] = value
    end
    kwargs = (;(key=>(value == "none" && key in (:combined,:voronoi,:adaptive_prism,:uniform_fine) ? nothing : value)
        for (key,value) in options)...)
    return plot_sdwr_convergence(;kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    sdwr_plot_cli(ARGS)
end
