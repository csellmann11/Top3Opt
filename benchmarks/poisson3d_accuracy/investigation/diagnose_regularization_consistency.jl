# Reuse the exact patches, fields, and input-data construction already checked
# by the selected-representative tests. Including this file also runs that suite.
isdefined(@__MODULE__, :selected_test_polynomial_data) || include("test_selected_weighted_patches.jl")
isdefined(@__MODULE__, :regularized_weighted_patch_system) || include("regularized_weighted_patches.jl")

function diagnose_regularization_consistency(;
        output_file=joinpath(@__DIR__,"..","results","patch_regularization_consistency.csv"))
    BLAS.set_num_threads(1)
    problem=manufactured_problem(:cube_smooth)
    regular=irregular_tetrahedral_mesh(:cube,3;amplitude=0.0)
    perturbed=irregular_tetrahedral_mesh(:cube,3;amplitude=.08)
    refined=locally_refined_tetrahedral_mesh(perturbed;stride=7)
    quadratic(x)=0.4+0.7x[1]-0.3x[2]+0.2x[3]+x[1]^2+2x[2]^2+3x[3]^2+
        0.5x[1]*x[2]-0.4x[1]*x[3]+0.6x[2]*x[3]
    affine(x)=0.7+0.2x[1]-0.3x[2]+0.4x[3]
    specs=regularized_weighted_default_specs()
    rows=NamedTuple[]
    for (geometry,control) in ((:regular,regular),(:perturbed,perturbed),(:refined,refined))
        mesh,fan=control.mesh,control.fan
        primal=solve_tetrahedral_p1(fan,problem)
        node=selected_test_interior_node(fan)
        qdata=selected_test_polynomial_data(mesh,fan,primal,quadratic,-12.0)
        qsystem=regularized_weighted_patch_system(fan,qdata,node)
        exact=[quadratic((fan.points[:,a]+fan.points[:,b])/2)-(qdata.U[a]+qdata.U[b])/2
            for (a,b) in qsystem.edges]
        exact_readout=dot(qsystem.g,exact)
        equation_defect=norm(qsystem.A*exact-qsystem.b)
        @assert equation_defect <= 5e-13
        qprepared=regularized_weighted_prepare(qsystem)
        adata=selected_test_polynomial_data(mesh,fan,primal,affine,0.0)
        asystem=regularized_weighted_patch_system(fan,adata,node)
        aprepared=regularized_weighted_prepare(asystem)
        for spec in specs
            state=regularized_weighted_solution(qprepared,spec)
            astate=regularized_weighted_solution(aprepared,spec)
            defect=state.coefficients-exact
            @assert abs(astate.readout) <= 5e-13
            push!(rows,(;geometry,node,method=spec.name,normalization=spec.normalization,
                filter=spec.filter,parameter=spec.parameter,dofs=length(exact),
                equation_defect,exact_readout,readout=state.readout,
                readout_defect=state.readout-exact_readout,
                relative_readout_defect=abs(state.readout-exact_readout)/abs(exact_readout),
                coefficient_defect=norm(defect),
                energy_norm_defect=norm(qprepared.Lk'*defect),
                exact_coefficient_norm=norm(exact),exact_energy_norm=norm(qprepared.Lk'*exact),
                retained_rank=state.retained_rank,chosen_alpha=state.chosen_alpha,
                actual_gain=state.actual_gain,gain_cap_active=state.gain_cap_active,
                normalized_equation_residual=state.normalized_residual_norm,
                affine_coefficient_norm=astate.coefficient_norm,affine_readout=astate.readout))
        end
        localrows=filter(row->row.geometry==geometry,rows)
        println("REGULARIZATION_CONSISTENCY ",(;geometry,node,exact_readout,equation_defect,
            maximum_affine_readout=maximum(abs(row.affine_readout) for row in localrows),
            methods=[(;method=row.method,relative_readout_defect=row.relative_readout_defect,
                coefficient_defect=row.coefficient_defect,gain_cap_active=row.gain_cap_active)
                for row in localrows]))
    end
    mkpath(dirname(output_file))
    open(output_file,"w") do io
        names=propertynames(first(rows))
        println(io,join(string.(names),','))
        for row in rows
            println(io,join((getproperty(row,name) for name in names),','))
        end
    end
    @assert length(rows)==3length(specs)
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    diagnose_regularization_consistency()
end
