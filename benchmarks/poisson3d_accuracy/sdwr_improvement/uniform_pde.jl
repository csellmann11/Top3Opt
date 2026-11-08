include("../sdwr_sine/run.jl")
using Test

"""Numerical integral of the known source squared; no solution truth is used."""
function pde_source_square(system,rhs;order=4)
    rule = gauss_legendre01(order)
    value = 0.0
    for cell in axes(system.cells,2),i in eachindex(rule.nodes),j in eachindex(rule.nodes),k in eachindex(rule.nodes)
        indices = (i,j,k)
        x = ntuple(d->system.origins[d,cell]+system.h*rule.nodes[indices[d]],3)
        value += system.h^3*rule.weights[i]*rule.weights[j]*rule.weights[k]*rhs(x)^2
    end
    return value
end

"""Uniform-cube asymptotic completion using only available PDE/goal data.

This function never receives exact fields, modal frequencies or reference
errors. Coefficients follow the uniform-cube consistency expansion; they are
not proposed as geometry-independent constants for irregular/adaptive meshes.
"""
function pde_completed_estimate(system,reference,source,rhs,u,z,goal;
        source_square=nothing,square_order=4)
    if goal == :energy
        stabilization = dot(u,sine_uniform_apply(system,reference.S,system.h,u))
        source_square === nothing && (source_square=pde_source_square(system,rhs;order=square_order))
        kernel_part = 5stabilization/12
        source_part = system.h^2*source_square/24
        retained_correction = dot(source.reconstructed-source.projected,u)
    else
        stabilization = dot(u,sine_uniform_apply(system,reference.S,system.h,z))
        pairing = goal == :integral_u ? sum(source.reconstructed) :
            goal == :integral_u2 ? 2dot(source.reconstructed,u) :
            throw(ArgumentError("unsupported goal $goal"))
        kernel_part = stabilization/3
        source_part = system.h^2*pairing/12
        goal_correction = goal == :integral_u ?
            dot(sine_uniform_load(system,reference.lP-reference.lT,system.h^3),u)/2 :
            dot(u,sine_uniform_apply(system,reference.B-reference.M,system.h^3,u))
        retained_correction = dot(source.reconstructed-source.projected,z)/2+goal_correction
    end
    estimate = kernel_part+source_part
    return (;estimate,kernel_part,source_part,retained_correction,
        retained_estimate=estimate+retained_correction)
end

"""Original global sDWR identity, with supplied analytic reference used only afterwards."""
function pde_original_evaluate(system,reference,source,u,goal)
    h = system.h
    if goal == :integral_u
        rhs = sine_uniform_load(system,reference.lT,h^3)
        value = dot(rhs,u)
        correction = dot(sine_uniform_load(system,reference.lP-reference.lT,h^3),u)/2
    elseif goal == :integral_u2
        mass_u = sine_uniform_apply(system,reference.M,h^3,u)
        rhs = 2mass_u
        value = dot(u,mass_u)
        correction = dot(u,sine_uniform_apply(system,reference.B-reference.M,h^3,u))
    elseif goal == :energy
        rhs = sine_uniform_apply(system,reference.H,h,u)
        value = dot(u,rhs)/2
        correction = dot(u,sine_uniform_apply(system,reference.C-reference.H,h,u))/2
    else
        throw(ArgumentError("unsupported goal $goal"))
    end
    dual = sine_uniform_solve(system,rhs)
    stabilization = dot(u,sine_uniform_apply(system,reference.S,h,dual.u))
    estimate = stabilization+dot(source.reconstructed-source.projected,dual.u)/2+correction
    return (;z=dual.u,rhs,value,estimate,stabilization,dual_residual=dual.algebraic_residual)
end

"""Problem generation and analytic truth are separate from estimator inputs."""
function pde_sine_mixture(name,parts)
    fields = [(coefficient,sine_problem(mode)) for (coefficient,mode) in parts]
    exact(x) = sum(c*p.exact(x) for (c,p) in fields)
    rhs(x) = sum(c*p.rhs(x) for (c,p) in fields)
    references = Dict(:integral_u=>sum(c*sine_reference(p,:integral_u) for (c,p) in fields),
        :integral_u2=>sum(c^2/8 for (c,p) in fields),
        :energy=>sum(c^2*p.reference_energy for (c,p) in fields))
    return (;name,exact,rhs,references,parts,kind=:sine_mixture)
end

function pde_cases()
    cases = NamedTuple[pde_sine_mixture(Symbol("sine_$(m)_1_1"),((1.0,(m,1,1)),)) for m in (1,3,5)]
    push!(cases,pde_sine_mixture(:mixture_positive,((1.0,(1,1,1)),(0.3,(3,5,1)))))
    push!(cases,pde_sine_mixture(:mixture_signed,((1.0,(3,1,1)),(-0.4,(1,5,1)))))
    polynomial = manufactured_problem(:cube_smooth)
    push!(cases,(;name=:cube_polynomial,exact=polynomial.exact,rhs=polynomial.rhs,
        references=Dict(:integral_u=>1/216,:integral_u2=>1/27000,
            :energy=>polynomial.reference_energy),parts=(),kind=:polynomial))
    return cases
end

"""Reference quadrature moments for degree-two polynomials in each coordinate."""
function pde_polynomial_moments(reference)
    powers = [(i,j,k) for k in 0:2 for j in 0:2 for i in 0:2]
    products = [reference.weights[q]*prod(reference.points[d,q]^power[d] for d in 1:3)
        for power in powers,q in eachindex(reference.weights)]
    return (;powers,projected=reference.shapeP*transpose(products),
        reconstructed=reference.shapeT*transpose(products))
end

"""Assemble the old polynomial cube source through reference quadrature moments."""
function pde_polynomial_source(system,moments)
    bP,bT = zeros(size(system.points,2)),zeros(size(system.points,2))
    coefficients,localP,localT = zeros(27),zeros(8),zeros(8)
    for cell in axes(system.cells,2)
        factors = ntuple(d->begin
            x = system.origins[d,cell]
            (x*(1-x),system.h*(1-2x),-system.h^2)
        end,3)
        for (q,power) in enumerate(moments.powers)
            coefficients[q] = 2sum((power[d]==0 ? 1.0 : 0.0)*
                prod(factors[j][power[j]+1] for j in 1:3 if j!=d) for d in 1:3)
        end
        mul!(localP,moments.projected,coefficients)
        mul!(localT,moments.reconstructed,coefficients)
        for a in 1:8
            node = system.cells[a,cell]
            bP[node] += system.h^3*localP[a]
            bT[node] += system.h^3*localT[a]
        end
    end
    return (;projected=bP,reconstructed=bT)
end

function pde_case_source(system,reference,problem,moments)
    problem.kind == :polynomial && return pde_polynomial_source(system,moments)
    bP,bT = zeros(size(system.points,2)),zeros(size(system.points,2))
    for (coefficient,mode) in problem.parts
        source = sine_uniform_source(system,reference,mode)
        bP .+= coefficient.*source.projected
        bT .+= coefficient.*source.reconstructed
    end
    return (;projected=bP,reconstructed=bT)
end

function validate_uniform_pde(;output=joinpath(@__DIR__,"..","results","sdwr_improvement","uniform_pde"))
    BLAS.set_num_threads(1)
    reference = sine_uniform_reference(;order=8)
    moments = pde_polynomial_moments(reference)
    system = sine_uniform_system(8,reference)
    records = NamedTuple[]
    @testset "Uniform PDE completion inputs and old estimator" begin
        for problem in pde_cases()
            source = pde_case_source(system,reference,problem,moments)
            u = sine_uniform_solve(system,source.projected).u
            q4 = pde_source_square(system,problem.rhs;order=4)
            q6 = pde_source_square(system,problem.rhs;order=6)
            @test abs(q4-q6)/q6 < 1e-8
            for goal in (:integral_u,:integral_u2,:energy)
                baseline = pde_original_evaluate(system,reference,source,u,goal)
                a = pde_completed_estimate(system,reference,source,problem.rhs,u,baseline.z,goal;source_square=q4)
                b = pde_completed_estimate(system,reference,source,problem.rhs,u,baseline.z,goal;source_square=q6)
                @test a.estimate ≈ b.estimate rtol=1e-8 atol=1e-14
                @test a.retained_estimate ≈ b.retained_estimate rtol=1e-8 atol=1e-14
                if goal != :energy
                    @test a.retained_estimate ≈ baseline.estimate-2baseline.stabilization/3+a.source_part rtol=1e-12
                end
                if problem.kind == :sine_mixture && length(problem.parts)==1
                    original = sine_uniform_evaluate(system,reference,sine_problem(problem.parts[1][2]),source,u,goal)
                    @test baseline.estimate ≈ original.estimate rtol=1e-13
                    @test baseline.value ≈ original.reconstructed rtol=1e-13
                end
                push!(records,(;problem=String(problem.name),goal=String(goal),
                    source_square_q4=q4,source_square_q6=q6,
                    candidate_q4=a.estimate,candidate_q6=b.estimate))
            end
        end
        # Independent original assembly checks the polynomial moment path.
        mesh = build_mesh(:cube,2)
        fan = build_fan(mesh)
        problem = manufactured_problem(:cube_smooth)
        full = assemble_primal(mesh,fan,problem;quadrature_order=8)
        small = sine_uniform_system(2,reference)
        source = pde_polynomial_source(small,moments)
        mapping = [begin
            i,j,k = round.(Int,2 .* mesh.points[:,a])
            sine_uniform_vertex(i,j,k,2)
        end for a in axes(mesh.points,2)]
        @test full.rhs ≈ source.projected[mapping] rtol=1e-12 atol=1e-14
    end
    mkpath(output)
    write_csv(joinpath(output,"validation.csv"),records)
    return records
end

function run_uniform_pde(;levels=(8,16,24,32),order=8,square_order=4,
        output=joinpath(@__DIR__,"..","results","sdwr_improvement","uniform_pde"))
    BLAS.set_num_threads(1)
    mkpath(output)
    reference = sine_uniform_reference(;order)
    moments = pde_polynomial_moments(reference)
    rows = NamedTuple[]
    for n in levels
        println("PDE_PREPARING ",(;n));flush(stdout)
        system = sine_uniform_system(n,reference)
        for problem in pde_cases()
            source = pde_case_source(system,reference,problem,moments)
            primal = sine_uniform_solve(system,source.projected)
            source_square = pde_source_square(system,problem.rhs;order=square_order)
            for goal in (:integral_u,:integral_u2,:energy)
                baseline = pde_original_evaluate(system,reference,source,primal.u,goal)
                candidate = pde_completed_estimate(system,reference,source,problem.rhs,
                    primal.u,baseline.z,goal;source_square)
                exact = problem.references[goal]
                error = exact-baseline.value
                row = (;problem=String(problem.name),goal=String(goal),n,cells=n^3,
                    free_dofs=length(system.free),exact_goal=exact,reconstructed_goal=baseline.value,
                    signed_goal_error=error,relative_goal_error=abs(error/exact),
                    original_estimate=baseline.estimate,original_effectivity=baseline.estimate/error,
                    candidate_estimate=candidate.estimate,candidate_effectivity=candidate.estimate/error,
                    retained_estimate=candidate.retained_estimate,retained_effectivity=candidate.retained_estimate/error,
                    candidate_kernel=candidate.kernel_part,candidate_source=candidate.source_part,
                    retained_correction=candidate.retained_correction,
                    source_square,primal_residual=primal.algebraic_residual,dual_residual=baseline.dual_residual)
                push!(rows,row)
                write_csv(joinpath(output,"accuracy.csv"),rows)
                println("PDE_COMPLETION ",(;problem=problem.name,goal,n,
                    old=row.original_effectivity,new=row.candidate_effectivity));flush(stdout)
            end
        end
        system = nothing
        GC.gc()
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    validate_uniform_pde()
    run_uniform_pde()
end
