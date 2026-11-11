# Optional diagnostics; these kernels run only when requested.
"""Normalized regularization force -b_i*(K chi)_i/area_i, including the local boost."""
function regularization_forces!(work,cache,chi)
    mul!(work.reg_before,cache.stiffness,cache.old)
    mul!(work.reg_after,cache.stiffness,chi)
    for i in eachindex(chi)
        factor = -cache.beta_relative[i]/cache.faces.area[i]
        work.reg_before[i] *= factor
        work.reg_after[i] *= factor
    end
end

function diagnostic_record!(work,cache,chi,p,scale,projection,options;
        eta0,rho,lower,substep,dt,elapsed,rate,system_updated,start_time,
        coefficient_seconds,system_seconds,solve_seconds,projection_seconds,
        physical=chi,physical_driving=p,density_projection=nothing)
    area, old = cache.faces.area, cache.old
    a, s = dt/eta0, scale.value
    linear_residual = gradient_linear_residual!(work,cache,chi,a,options)
    stationarity,split_defect = gradient_stationarity!(work,cache,chi,p,s,a,rho,lower,options)
    lambda = projection.shift*s/a
    (;total,work_J,physical_work,R_before,R_after,move2,move_max,design_move,reg2,net2,
      boosted,boost_mean,normalized_max,bound_area,beta_min,beta_max) =
        gradient_balance_metrics(work,cache,chi,p,s,lambda,physical,physical_driving,lower)
    pmax = maximum(abs,p)
    return (substep=substep,dt=dt,pseudo_time=elapsed+dt,density_projection_metadata(density_projection)...,
        p_avg=s,p_avg_raw=scale.raw,p_max=pmax,peak_to_mean=pmax/s,
        scale_fallback=scale.fallback,scale_floored=scale.floored,gray_weight=scale.gray_weight,
        eta=eta0*s,beta_min=beta_min,beta_max=beta_max,
        beta_boost_max=maximum(cache.beta_relative),beta_boost_mean=boost_mean/total,
        boosted_area_fraction=boosted/total,normalized_drive_max=normalized_max,
        volume_error=dot(area,physical)/total-rho,design_volume_error=projection.error,
        chi_min=minimum(physical),chi_max=maximum(physical),
        design_min=minimum(chi),design_max=maximum(chi),
        move_linf=move_max,move_rms=sqrt(move2/total),design_move_linf=design_move,
        grayness_metrics(physical,area,lower)...,design_grayness=grayness_metrics(chi,area,lower).grayness,
        bound_area_fraction=bound_area/total,lambda=lambda,projection_iterations=projection.iterations,
        regularization_before=R_before,regularization_after=R_after,
        regularization_field=density_projection === nothing ? "physical" : "design",
        linearized_design_objective_change=work_J,linearized_objective_change=physical_work,
        projection_linearization_defect=abs(physical_work-work_J),
        linearized_merit_change=work_J+R_after-R_before,
        reg_force_rms=s*sqrt(reg2/total),net_drive_rms=sqrt(net2/total),
        frozen_stationarity_linf=stationarity,split_equation_defect_linf=split_defect,
        linear_residual=linear_residual,diffusion_cfl=dt*rate,
        system_updated=system_updated,coefficient_seconds=coefficient_seconds,
        system_seconds=system_seconds,solve_seconds=solve_seconds,
        projection_seconds=projection_seconds,density_seconds=(time_ns()-start_time)*1e-9)
end

function filter_residual!(scratch,cache,output,input)
    mul!(scratch,cache.system,output)
    residual = rhs_norm = 0.0
    for i in eachindex(input)
        rhs = cache.faces.area[i]*input[i]
        residual = max(residual,abs(scratch[i]-rhs))
        rhs_norm = max(rhs_norm,abs(rhs))
    end
    return residual/max(rhs_norm,eps(Float64))
end

function filter_record!(work,cache,design,chi,p,scale,projection,options;
        eta0,rho,lower,substep,dt,adjoint_iterations,forward_iterations,
        adjoint_seconds,forward_seconds,projection_seconds,start_time,
        filtered=chi,filtered_driving=p,density_projection=nothing)
    area, old = cache.faces.area, cache.old
    adjoint_residual = filter_residual!(work.product,cache,cache.q,filtered_driving)
    forward_residual = filter_residual!(work.product,cache,filtered,design)
    @. work.trial = cache.design_before+cache.q/scale.value
    project_volume!(work.projected,work.trial,area,rho,lower;tolerance=options.volume_tolerance)
    stationarity = maximum(i -> abs(work.projected[i]-cache.design_before[i]),eachindex(design))
    (;total,move2,move_max,design_move,work_J,filtered_work,design_work,bound_area) =
        filter_balance_metrics(work,cache,design,chi,p,filtered,filtered_driving,lower)
    pmax = maximum(abs,p)
    return (substep=substep,dt=dt,pseudo_time=substep*dt,density_projection_metadata(density_projection)...,
        p_avg=scale.value,p_avg_raw=scale.raw,p_max=pmax,peak_to_mean=pmax/scale.value,
        scale_fallback=scale.fallback,scale_floored=scale.floored,gray_weight=scale.gray_weight,
        eta=eta0*scale.value,radius_min=sqrt(minimum(cache.radius2)),radius_max=sqrt(maximum(cache.radius2)),
        volume_error=dot(area,chi)/total-rho,design_volume_error=projection.error,
        filter_volume_defect=dot(area,filtered)/total-dot(area,design)/total,
        projection_volume_defect=dot(area,chi)/total-dot(area,filtered)/total,
        chi_min=minimum(chi),chi_max=maximum(chi),design_min=minimum(design),design_max=maximum(design),
        move_linf=move_max,move_rms=sqrt(move2/total),design_move_linf=design_move,
        grayness_metrics(chi,area,lower)...,design_grayness=grayness_metrics(design,area,lower).grayness,
        bound_area_fraction=bound_area/total,lambda=projection.shift*eta0*scale.value/dt,
        projection_iterations=projection.iterations,linearized_objective_change=work_J,
        linearized_design_objective_change=design_work,
        projection_linearization_defect=abs(work_J-filtered_work),
        adjoint_work_defect=abs(filtered_work-design_work),frozen_stationarity_linf=stationarity,
        linear_residual=max(adjoint_residual,forward_residual),
        adjoint_residual=adjoint_residual,forward_residual=forward_residual,
        adjoint_iterations=adjoint_iterations,forward_iterations=forward_iterations,
        adjoint_seconds=adjoint_seconds,forward_seconds=forward_seconds,
        projection_seconds=projection_seconds,density_seconds=(time_ns()-start_time)*1e-9)
end

function gradient_linear_residual!(work,cache,chi,a,options)
    area = cache.faces.area
    linear_residual = 0.0
    if options.integrator == :semi_implicit
        mul!(work.trial,cache.stiffness,cache.trial)
        rhs_norm = 0.0
        for i in eachindex(chi)
            linear_residual = max(linear_residual,abs(area[i]*cache.trial[i]+a*work.trial[i]-cache.rhs[i]))
            rhs_norm = max(rhs_norm,abs(cache.rhs[i]))
        end
        linear_residual /= max(rhs_norm,eps(Float64))
    end
    return linear_residual
end

"""Check projected stationarity and the semi-implicit split equation independently."""
function gradient_stationarity!(work,cache,chi,p,s,a,rho,lower,options)
    area,old = cache.faces.area,cache.old
    # These extra projections are diagnostics, not part of the actual update.
    @. work.trial = old+p/s+work.reg_before
    project_volume!(work.projected,work.trial,area,rho,lower;tolerance=options.volume_tolerance)
    stationarity = maximum(i -> abs(work.projected[i]-old[i]),eachindex(chi))
    split_defect = 0.0
    if options.integrator == :semi_implicit
        # Fully implicit constrained step; reg_after carries b, so dividing by b
        # leaves baseline diffusion and the boosted-down force and multiplier.
        boost = cache.beta_relative
        @. work.trial = old+(a/boost)*(p/s+work.reg_after)
        weights = all(isone,boost) ? nothing : (@. work.weights = 1/boost)
        project_volume!(work.projected,work.trial,area,rho,lower;
            tolerance=options.volume_tolerance,weights)
        split_defect = maximum(i -> abs(work.projected[i]-chi[i]),eachindex(chi))
    end
    return stationarity,split_defect
end

"""Accumulate energy, work, movement and coefficient diagnostics in one cell pass."""
function gradient_balance_metrics(work,cache,chi,p,s,lambda,physical,physical_driving,lower)
    area,old = cache.faces.area,cache.old
    total = sum(area)
    work_J = physical_work = R_before = R_after = move2 = move_max = design_move = reg2 = net2 = 0.0
    boosted = boost_mean = normalized_max = bound_area = 0.0
    beta_min, beta_max = Inf, 0.0
    for i in eachindex(chi)
        delta = chi[i]-old[i]
        physical_delta = physical[i]-work.physical_before[i]
        work_J -= area[i]*p[i]*delta
        physical_work -= area[i]*physical_driving[i]*physical_delta
        R_before -= area[i]*old[i]*work.reg_before[i]*s/2
        R_after -= area[i]*chi[i]*work.reg_after[i]*s/2
        move2 += area[i]*physical_delta^2
        move_max = max(move_max,abs(physical_delta))
        design_move = max(design_move,abs(delta))
        reg2 += area[i]*work.reg_before[i]^2
        net2 += area[i]*(p[i]-lambda)^2
        boost = cache.beta_relative[i]
        boost_mean += area[i]*boost
        boost > 1+1e-12 && (boosted += area[i])
        normalized_max = max(normalized_max,abs(p[i])/(s*boost))
        beta = s*cache.base_beta[i]*boost
        beta_min = min(beta_min,beta); beta_max = max(beta_max,beta)
        (chi[i]<=lower+1e-12 || chi[i]>=1-1e-12) && (bound_area += area[i])
    end
    return (;total,work_J,physical_work,R_before,R_after,move2,move_max,design_move,reg2,net2,boosted,boost_mean,normalized_max,bound_area,beta_min,beta_max)
end

"""Compare physical, filtered and design work in the same area metric."""
function filter_balance_metrics(work,cache,design,chi,p,filtered,filtered_driving,lower)
    area,old = cache.faces.area,cache.old
    total = sum(area)
    move2 = move_max = design_move = work_J = filtered_work = design_work = bound_area = 0.0
    for i in eachindex(chi)
        delta = chi[i]-old[i]
        delta_a = design[i]-cache.design_before[i]
        move2 += area[i]*delta^2
        move_max = max(move_max,abs(delta))
        design_move = max(design_move,abs(delta_a))
        work_J -= area[i]*p[i]*delta
        filtered_work -= area[i]*filtered_driving[i]*(filtered[i]-work.filtered_before[i])
        design_work -= area[i]*cache.q[i]*delta_a
        (design[i] <= lower+1e-12 || design[i] >= 1-1e-12) && (bound_area += area[i])
    end
    return (;total,move2,move_max,design_move,work_J,filtered_work,design_work,bound_area)
end
