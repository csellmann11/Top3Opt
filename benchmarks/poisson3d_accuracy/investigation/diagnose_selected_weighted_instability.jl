isdefined(@__MODULE__, :poisson_estimator_data) || include(joinpath(@__DIR__, "..", "load.jl"))
isdefined(@__MODULE__, :irregular_tetrahedral_mesh) || include("irregular_tetrahedra.jl")
isdefined(@__MODULE__, :selected_weighted_patch_system) || include("selected_weighted_patches.jl")

"""Direct quadrature check independent of the cubic-moment operator assembly."""
function weighted_instability_quadrature(fan, system, U, forcing; order=6)
    n = length(system.edges)
    A, b = zeros(n,n), zeros(n)
    rule = tetra_rule(order)
    for tet in system.tets
        ids, G = fan.tet_nodes[:,tet], fan.gradients[tet]
        hat = findfirst(==(system.node), ids)
        rows = [get(system.edge_rows, minmax(ids[a],ids[c]),0) for (a,c) in POISSON_P2_EDGES]
        xyz = fan.points[:,ids] * rule.barycentric
        gradU = G * U[ids]
        for q in eachindex(rule.weights)
            N = rule.barycentric[:,q]
            values = [4N[a]*N[c] for (a,c) in POISSON_P2_EDGES]
            gradients = [4 .* (N[a].*G[:,c] + N[c].*G[:,a]) for (a,c) in POISSON_P2_EDGES]
            weight = fan.volumes[tet] * rule.weights[q]
            for j in 1:6
                row=rows[j]
                row==0 && continue
                gradtest=N[hat].*gradients[j] + values[j].*G[:,hat]
                b[row] += weight*(forcing(view(xyz,:,q))*N[hat]*values[j]-dot(gradU,gradtest))
                for k in 1:6
                    rows[k]==0 && continue
                    A[row,rows[k]] += weight*dot(gradients[k],gradtest)
                end
            end
        end
    end
    return (;A,b)
end

function diagnose_selected_weighted_instability(;n=6, amplitude=.08,
        output_directory=joinpath(@__DIR__,"..","results","selected_weighted_instability"))
    BLAS.set_num_threads(1)
    control=irregular_tetrahedral_mesh(:cube,n;amplitude)
    mesh,fan=control.mesh,control.fan
    problem=manufactured_problem(:cube_smooth)
    primal=solve_tetrahedral_p1(fan,problem)
    data=tetrahedral_p1_estimator_data(mesh,fan,primal,problem)
    truth=tetrahedral_p1_truth(mesh,fan,primal,problem)
    goalerror=sum(truth.goal_difference)
    println("CASE ", (;n,amplitude,goalerror,maximum_primal=maximum(primal.u),
        common_correction=data.common_correction,geometry=control.geometry_stats))
    mkpath(output_directory)
    selected_nodes=n==6 ? (998,521,429,961,761) : (358,143,312)
    nodefile=open(joinpath(output_directory,"nodes_n$(n).csv"),"w")
    println(nodefile,"node,tolerance,rank,nullity,condition_estimate,coefficient_norm,raw_readout,effectivity_contribution,relative_residual")
    modesfile=open(joinpath(output_directory,"modes_n$(n).csv"),"w")
    println(modesfile,"node,mode,sigma_ratio,rhs_projection,readout_projection,coefficient_amplitude,raw_readout,effectivity_contribution")
    for node in selected_nodes
        system=selected_weighted_patch_system(fan,data,node)
        base=selected_weighted_representatives(system)
        direct=weighted_instability_quadrature(fan,system,data.U,problem.rhs)
        lu_delta=system.A\system.b
        lu_quad_delta=direct.A\direct.b
        svd_delta=base.minimum_coefficient.coefficients
        vertices=unique(vec(fan.tet_nodes[:,system.tets]))
        physical_vertices=[v for v in vertices if fan.boundary_nodes[v]]
        println("PATCH ",(;node,coordinate=fan.points[:,node],physical=fan.boundary_nodes[node],
            tetrahedra=length(system.tets),physical_vertices,
            physical_coordinates=fan.points[:,physical_vertices],
            unknown_physical_edges=count(e->e in fan.boundary_edges,system.edges),
            operator_relative_difference=norm(system.A-direct.A)/norm(system.A),
            rhs_relative_difference=norm(system.b-direct.b)/norm(system.b),
            readout=dot(system.g,svd_delta),direct_readout=dot(direct.b,svd_delta),
            lu_readout=dot(system.g,lu_delta),quadrature_lu_readout=dot(direct.b,lu_quad_delta),
            lu_svd_relative_difference=norm(lu_delta-svd_delta)/norm(svd_delta),
            lu_relative_residual=norm(system.A*lu_delta-system.b)/norm(system.b)))
        for tolerance in (1e-13,1e-11,1e-9,1e-8,1e-6)
            selected=selected_weighted_representatives(system;rank_tolerance=tolerance)
            state=selected.minimum_coefficient
            row=(node,tolerance,selected.rank,selected.nullity,selected.condition_estimate,
                state.coefficient_norm,state.readout,.5state.readout/goalerror,state.relative_residual)
            println(nodefile,join(row,','))
            println("TOL ",row)
        end
        decomposition=svd(system.A)
        rhs_projection=decomposition.U'*system.b
        readout_projection=decomposition.V'*system.g
        contributions=rhs_projection.*readout_projection./decomposition.S
        for k in eachindex(contributions)
            println(modesfile,join((node,k,decomposition.S[k]/decomposition.S[1],
                rhs_projection[k],readout_projection[k],rhs_projection[k]/decomposition.S[k],
                contributions[k],.5contributions[k]/goalerror),','))
        end
        for k in sortperm(abs.(contributions);rev=true)[1:min(3,length(contributions))]
            println("MODE ",(;node,k,ratio=decomposition.S[k]/decomposition.S[1],
                rhs_projection=rhs_projection[k],readout_projection=readout_projection[k],
                amplitude=rhs_projection[k]/decomposition.S[k],readout=contributions[k],
                effectivity_contribution=.5contributions[k]/goalerror))
        end
        flush(stdout)
    end
    close(nodefile)
    close(modesfile)
    globalfile=open(joinpath(output_directory,"global_n$(n).csv"),"w")
    println(globalfile,"n,tolerance,selector,signed_estimate,effectivity,singular_patches,maximum_relative_patch_residual")
    for tolerance in (1e-11,1e-8,1e-6)
        result=selected_weighted_patch_estimates(mesh,fan,data;rank_tolerance=tolerance)
        for selector in propertynames(result.estimates)
            estimate=getproperty(result.estimates,selector)
            row=(n,tolerance,selector,estimate.signed,estimate.signed/goalerror,
                count(r->r.nullity>0,result.diagnostics),maximum(r.relative_residual for r in result.diagnostics))
            println(globalfile,join(row,','))
            println("GLOBAL ",row)
        end
        flush(stdout)
    end
    close(globalfile)
    return nothing
end

if abspath(PROGRAM_FILE)==@__FILE__
    diagnose_selected_weighted_instability()
end
