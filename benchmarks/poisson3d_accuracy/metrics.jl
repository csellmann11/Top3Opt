"""Independent exact-field integration for interpreting the cell indicators.

The positive reconstruction error is a spatial diagnostic. Its sum is not the
signed VEM goal error: the latter also contains consistency/stabilization terms.
"""
function exact_cell_diagnostics(mesh,fan,primal,problem;quadrature_order=10)
    rule = tetra_rule(quadrature_order)
    U = reconstruct_fan(mesh,fan,primal.u)
    exact_energy = zeros(length(mesh.cells))
    reconstruction_error = similar(exact_energy); fill!(reconstruction_error,0)
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:,tet]
        gradient = fan.gradients[tet]*U[ids]
        quadrature = tetra_quadrature(fan.points[:,ids],rule)
        cell = fan.tet_cells[tet]
        for q in eachindex(quadrature.weights)
            exact_gradient = collect(problem.gradient(quadrature.points[:,q]))
            weight = quadrature.weights[q]/2
            exact_energy[cell] += weight*dot(exact_gradient,exact_gradient)
            difference = exact_gradient-gradient
            reconstruction_error[cell] += weight*dot(difference,difference)
        end
    end
    discrete_energy = [dot(primal.u[mesh.cells[e]],
        primal.cells[e].K*primal.u[mesh.cells[e]])/2 for e in eachindex(mesh.cells)]
    return (;exact_energy,discrete_energy,goal_difference=exact_energy-discrete_energy,
        reconstruction_error,integrated_exact_energy=sum(exact_energy))
end

function safe_ratio(numerator,denominator;scale=1.0)
    abs(denominator) > 100eps(Float64)*abs(scale) || return NaN
    return numerator/denominator
end

"""Compare marking with positive reconstructed-field energy error, not with ΔJ."""
function marking_diagnostics(indicators,truth;fraction=0.2)
    all(>=(0),indicators) || error("Negative marking indicator")
    count = max(1,ceil(Int,fraction*length(indicators)))
    selected = sortperm(indicators;rev=true)[1:count]
    ideal = sortperm(truth;rev=true)[1:count]
    indicator_sum,truth_sum = sum(indicators),sum(truth)
    distribution_l1 = indicator_sum > 0 && truth_sum > 0 ?
        sum(abs,indicators/indicator_sum-truth/truth_sum) : NaN
    captured = safe_ratio(sum(truth[selected]),truth_sum;scale=truth_sum)
    optimal = safe_ratio(sum(truth[ideal]),truth_sum;scale=truth_sum)
    return (;distribution_l1,captured_error_fraction=captured,
        optimal_captured_fraction=optimal,marked_cells=count)
end

function accuracy_row(problem,mesh,primal,truth,method,estimate;
                      quadrature_order,truth_order,mark_fraction=0.2)
    goal_error = problem.reference_energy-primal.energy
    marking = marking_diagnostics(estimate.indicators,truth.reconstruction_error;
        fraction=mark_fraction)
    return (;problem=String(problem.name),domain=String(problem.domain),n=mesh.n,
        cells=length(mesh.cells),vertices=size(mesh.points,2),free_dofs=length(primal.free_dofs),
        method=String(method),exact_goal=problem.reference_energy,discrete_goal=primal.energy,
        signed_goal_error=goal_error,
        relative_goal_error=abs(goal_error)/problem.reference_energy,
        signed_estimate=estimate.signed,indicator_sum=estimate.magnitude,
        signed_effectivity=safe_ratio(estimate.signed,goal_error;scale=problem.reference_energy),
        magnitude_effectivity=safe_ratio(estimate.magnitude,abs(goal_error);scale=problem.reference_energy),
        reconstruction_energy_error=sum(truth.reconstruction_error),
        exact_energy_integration_discrepancy=abs(truth.integrated_exact_energy-problem.reference_energy),
        reference_energy_uncertainty=problem.reference_uncertainty,
        primal_relative_residual=primal.algebraic_residual,
        patch_total_dofs=estimate.statistics.total_dofs,
        patch_max_dofs=estimate.statistics.max_dofs,
        compressed_patches=get(estimate.statistics,:reduced_patches,0),
        sdwr_stabilization_part=get(estimate.statistics,:stabilization_part,NaN),
        sdwr_load_consistency_part=get(estimate.statistics,:load_consistency_part,NaN),
        quadrature_order,truth_order,mark_fraction,marking...)
end

function csv_value(value)
    value isa AbstractFloat && return @sprintf("%.17g",value)
    text = string(value)
    return occursin(r"[,\"\n]",text) ? "\""*replace(text,"\""=>"\"\"")*"\"" : text
end

function write_csv(path,rows)
    isempty(rows) && return
    open(path,"w") do io
        names = propertynames(first(rows))
        println(io,join(string.(names),','))
        for row in rows
            propertynames(row) == names || error("CSV row schema changed")
            println(io,join((csv_value(getproperty(row,name)) for name in names),','))
        end
    end
end

function write_cell_results(path,mesh,primal,truth,estimates)
    rows = NamedTuple[]
    for cell in eachindex(mesh.cells)
        x = mesh.centers[:,cell]
        push!(rows,(;cell,x=x[1],y=x[2],z=x[3],volume=mesh.volumes[cell],
            exact_energy=truth.exact_energy[cell],discrete_energy=truth.discrete_energy[cell],
            signed_goal_difference=truth.goal_difference[cell],
            reconstruction_energy_error=truth.reconstruction_error[cell],
            full_pu_indicator=estimates.full.indicators[cell],
            reduced_pu_indicator=estimates.reduced.indicators[cell],
            sdwr_indicator=estimates.sdwr.indicators[cell]))
    end
    write_csv(path,rows)
end

function write_node_results(path,fan,estimates)
    rows = NamedTuple[]
    for node in axes(fan.points,2)
        x = fan.points[:,node]
        kind = node <= fan.vertex_count ? "vertex" :
            node <= fan.vertex_count+fan.face_count ? "face_center" : "cell_center"
        push!(rows,(;node,kind,x=x[1],y=x[2],z=x[3],
            physical_boundary=fan.boundary_nodes[node],adjacent_cells=length(fan.node_cells[node]),
            full_dofs=estimates.full.patchdofs[node],reduced_dofs=estimates.reduced.patchdofs[node],
            full_signed_contribution=estimates.full.nodal[node],
            reduced_signed_contribution=estimates.reduced.nodal[node],
            sdwr_signed_contribution=estimates.sdwr.nodal[node]))
    end
    write_csv(path,rows)
end

"""Legacy VTK is readable in ParaView; it contains every cell diagnostic."""
function write_cell_vtk(path,mesh,primal,truth,estimates)
    open(path,"w") do io
        println(io,"# vtk DataFile Version 3.0\nPoisson PU accuracy comparison\nASCII\nDATASET UNSTRUCTURED_GRID")
        println(io,"POINTS $(size(mesh.points,2)) double")
        for x in eachcol(mesh.points)
            println(io,join(x,' '))
        end
        println(io,"CELLS $(length(mesh.cells)) $(9length(mesh.cells))")
        for cell in mesh.cells
            println(io,"8 ",join(cell.-1,' '))
        end
        println(io,"CELL_TYPES $(length(mesh.cells))")
        for _ in mesh.cells
            println(io,12)
        end
        println(io,"POINT_DATA $(length(primal.u))\nSCALARS solution double 1\nLOOKUP_TABLE default")
        foreach(value -> println(io,value),primal.u)
        println(io,"CELL_DATA $(length(mesh.cells))")
        fields = (;exact_energy=truth.exact_energy,discrete_energy=truth.discrete_energy,
            signed_goal_difference=truth.goal_difference,
            reconstruction_energy_error=truth.reconstruction_error,
            full_pu_indicator=estimates.full.indicators,
            reduced_pu_indicator=estimates.reduced.indicators,sdwr_indicator=estimates.sdwr.indicators)
        for (name,values) in pairs(fields)
            println(io,"SCALARS $name double 1\nLOOKUP_TABLE default")
            foreach(value -> println(io,value),values)
        end
    end
end

function write_accuracy_report(path,rows,comparisons)
    open(path,"w") do io
        println(io,"# Poisson 3D accuracy experiment\n")
        println(io,"Signed effectivity is eta / (J_exact - J_h). Values near 1 indicate agreement, including sign. Indicator sum is a separate nonnegative marking quantity. No runtime comparison is made.\n")
        println(io,"| Problem | n | Cells | Method | Relative goal error | Signed effectivity | Magnitude effectivity | Error captured by top 20% |")
        println(io,"| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
        for row in rows
            @printf(io,"| %s | %d | %d | %s | %.4g | %.5f | %.5f | %.1f%% |\n",
                row.problem,row.n,row.cells,row.method,row.relative_goal_error,
                row.signed_effectivity,row.magnitude_effectivity,100row.captured_error_fraction)
        end
        println(io,"\nThe captured-error metric uses the positive energy error of the continuous P1 fan reconstruction. It is a spatial diagnostic and does not sum to the signed VEM goal error. Rankings with tied indicators use cell order.\n")
        println(io,"| Problem | n | Relative change in signed estimate | Relative L1 indicator change |")
        println(io,"| --- | ---: | ---: | ---: |")
        for row in comparisons
            @printf(io,"| %s | %d | %.4g | %.4g |\n",row.problem,row.n,
                row.relative_signed_change,row.relative_indicator_change)
        end
        println(io,"\nsDWR substitutes the elementwise linear VEM projection into the same volume estimator, retaining the goal and cell load-consistency corrections. With this projected-load discretization its signed total is (S + F(U)-F_h(u_h))/2; it is not the former unscaled S proxy. Boundary patches in the reduced method enforce the same zero P2 Dirichlet DOFs before rank reduction. See README.md for the construction and scope.")
        println(io,"\n## Independent truth integration\n")
        println(io,"The global reference uses analytic or independent polar integration. The cell diagnostics use finite-order tetrahedral integration. The following discrepancy is relative to the absolute goal error, not to the much larger exact energy. It diagnoses integration accuracy but is not a certified local-error bound.\n")
        println(io,"| Problem | n | Exact-energy integration discrepancy / abs(goal error) | Reference uncertainty / abs(goal error) |")
        println(io,"| --- | ---: | ---: | ---: |")
        for row in rows
            row.method == "full" || continue
            @printf(io,"| %s | %d | %.4g | %.4g |\n",row.problem,row.n,
                safe_ratio(row.exact_energy_integration_discrepancy,abs(row.signed_goal_error);scale=row.exact_goal),
                safe_ratio(row.reference_energy_uncertainty,abs(row.signed_goal_error);scale=row.exact_goal))
        end
    end
end
