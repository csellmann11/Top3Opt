isdefined(@__MODULE__,:sdwr_solve_rhs) || include("../sdwr_adaptive/run.jl")
isdefined(@__MODULE__,:sine_uniform_reference) || include("../sdwr_sine/uniform.jl")
isdefined(@__MODULE__,:hierarchical_geometry) || include("hierarchical_diagonal.jl")

"""Coupled isotropic elasticity: u=(sin(m*pi*x)sin(pi*y)sin(pi*z),0,0)."""
function elasticity_problem(m;lambda=1.0,mu=1.0)
    m > 0 || throw(ArgumentError("positive mode required"))
    function exact(x)
        return (sinpi(m*x[1])*sinpi(x[2])*sinpi(x[3]),0.0,0.0)
    end
    function rhs(x)
        sx,sy,sz = sinpi(m*x[1]),sinpi(x[2]),sinpi(x[3])
        cx,cy,cz = cospi(m*x[1]),cospi(x[2]),cospi(x[3])
        return (pi^2*((lambda+2mu)*m^2+2mu)*sx*sy*sz,
            -(lambda+mu)*m*pi^2*cx*cy*sz,
            -(lambda+mu)*m*pi^2*cx*sy*cz)
    end
    return (;m,lambda,mu,exact,rhs)
end

"""Element elasticity matrix for scalar basis gradients; component-fast DOFs."""
function elasticity_matrix(G,lambda,mu)
    nb = size(G,2)
    K = zeros(3nb,3nb)
    for b in 1:nb,a in 1:nb
        product = dot(view(G,:,a),view(G,:,b))
        for j in 1:3,i in 1:3
            K[3(a-1)+i,3(b-1)+j] = mu*(i==j ? product : 0.0)+
                mu*G[j,a]*G[i,b]+lambda*G[i,a]*G[j,b]
        end
    end
    return K
end

elasticity_dofs(nodes) = [3(node-1)+component for node in nodes for component in 1:3]

function elasticity_shapes(rule)
    return vcat(rule.barycentric,reduce(vcat,[transpose(4 .* rule.barycentric[a,:].*
        rule.barycentric[b,:]) for (a,b) in HIERARCHICAL_TET_EDGES]))
end

"""Assemble coupled vector stiffness and scalar mass in the fixed P2 fan space."""
function elasticity_enriched_operator(fan,geometry,lambda,mu)
    nt,ns = length(fan.volumes),size(geometry.K,1)
    rows,columns,values = Vector{Int}(undef,900nt),Vector{Int}(undef,900nt),Vector{Float64}(undef,900nt)
    mr,mc,mv = Vector{Int}(undef,100nt),Vector{Int}(undef,100nt),Vector{Float64}(undef,100nt)
    rule = tetra_rule(4)
    shapes = elasticity_shapes(rule)
    mass = (shapes.*transpose(rule.weights))*transpose(shapes)
    moment = shapes*rule.weights
    load = zeros(ns)
    cursor,mindex = 0,0
    for tet in 1:nt
        G,volume = fan.gradients[tet],fan.volumes[tet]
        gradients = [hcat(G,[4 .* ((vertex==a ? G[:,b] : zeros(3))+
            (vertex==b ? G[:,a] : zeros(3))) for (a,b) in HIERARCHICAL_TET_EDGES]...)
            for vertex in 1:4]
        K = volume/20*(elasticity_matrix(sum(gradients),lambda,mu)+
            sum(elasticity_matrix(g,lambda,mu) for g in gradients))
        nodes = geometry.dofs[:,tet]
        dofs = elasticity_dofs(nodes)
        for j in 1:30,i in 1:30
            cursor += 1
            rows[cursor],columns[cursor],values[cursor] = dofs[i],dofs[j],K[i,j]
        end
        for b in 1:10,a in 1:10
            mindex += 1
            mr[mindex],mc[mindex],mv[mindex] = nodes[a],nodes[b],volume*mass[a,b]
        end
        load[nodes] .+= volume.*moment
    end
    K = sparse(rows,columns,values,3ns,3ns)
    M = sparse(mr,mc,mv,ns,ns)
    dropzeros!(K); dropzeros!(M)
    return (;K,M,load,free=elasticity_dofs(geometry.free),diagonal=diag(K))
end

"""Original vertex VEM-style vector operator and reused coarse factor."""
function elasticity_coarse_operator(mesh,lambda,mu)
    nc,nv = length(mesh.cells),size(mesh.points,2)
    rows,columns,values,stabilization = Vector{Int}(undef,576nc),Vector{Int}(undef,576nc),
        Vector{Float64}(undef,576nc),Vector{Float64}(undef,576nc)
    cells = [poisson_cell_operator(mesh,cell) for cell in 1:nc]
    cursor = 0
    for cell in 1:nc
        data = cells[cell]
        D = Matrix{Float64}(I,8,8)-data.P
        S = kron(mu*data.h*transpose(D)*D,Matrix{Float64}(I,3,3))
        K = data.volume*elasticity_matrix(data.G,lambda,mu)+S
        dofs = elasticity_dofs(mesh.cells[cell])
        for j in 1:24,i in 1:24
            cursor += 1
            rows[cursor],columns[cursor],values[cursor],stabilization[cursor] =
                dofs[i],dofs[j],K[i,j],S[i,j]
        end
    end
    K,S = sparse(rows,columns,values,3nv,3nv),sparse(rows,columns,stabilization,3nv,3nv)
    free = elasticity_dofs(findall(!,mesh.boundary_vertices))
    factor = cholesky(Symmetric(K[free,free]))
    return (;K,S,free,factor,cells)
end

"""Same source quadrature for projected coarse load and genuine enriched load."""
function elasticity_loads(mesh,fan,geometry,coarse,problem;order=6)
    projected = zeros(size(coarse.K,1))
    enriched = zeros(3size(geometry.K,1))
    rule = tetra_rule(order)
    shapes = elasticity_shapes(rule)
    for tet in eachindex(fan.volumes)
        cell = fan.tet_cells[tet]
        data = coarse.cells[cell]
        xyz = fan.points[:,fan.tet_nodes[:,tet]]*rule.barycentric
        P = fill(1/8,8,length(rule.weights))+transpose(data.G)*(xyz.-data.center)
        local_coarse,local_enriched = zeros(3,8),zeros(3,10)
        for q in eachindex(rule.weights)
            f = problem.rhs(view(xyz,:,q))
            weight = fan.volumes[tet]*rule.weights[q]
            for component in 1:3
                for a in 1:8
                    local_coarse[component,a] += weight*f[component]*P[a,q]
                end
                for a in 1:10
                    local_enriched[component,a] += weight*f[component]*shapes[a,q]
                end
            end
        end
        projected[elasticity_dofs(mesh.cells[cell])] .+= vec(local_coarse)
        enriched[elasticity_dofs(geometry.dofs[:,tet])] .+= vec(local_enriched)
    end
    return (;projected,enriched)
end

function elasticity_solve(coarse,rhs)
    u = zeros(length(rhs))
    u[coarse.free] = coarse.factor\rhs[coarse.free]
    return u
end

function elasticity_goal(enriched,U,problem,name)
    ns = size(enriched.M,1)
    u1 = U[1:3:end]
    rhs = zeros(length(U))
    if name == :integral_u
        rhs[1:3:end] = enriched.load
        value = dot(enriched.load,u1)
        exact = 8/(problem.m*pi^3)
        remainder = delta->0.0
    elseif name == :integral_u2
        rhs[1:3:end] = 2(enriched.M*u1)
        value = dot(u1,enriched.M*u1)
        exact = 1/8
        remainder = delta->dot(delta[1:3:end],enriched.M*delta[1:3:end])
    elseif name == :energy
        rhs = enriched.K*U
        value = dot(U,rhs)/2
        exact = pi^2*((problem.lambda+2problem.mu)*problem.m^2+2problem.mu)/16
        remainder = delta->dot(delta,enriched.K*delta)/2
    else
        throw(ArgumentError("unknown elasticity goal"))
    end
    return (;rhs,value,exact,remainder)
end

function elasticity_sdwr(mesh,coarse,enriched,source,uh,zh,U,Z,name,scalar_reference)
    S = dot(uh,coarse.S*zh)
    load = (dot(source.enriched,Z)-dot(source.projected,zh))/2
    correction = 0.0
    if name == :integral_u2
        for cell in eachindex(mesh.cells)
            local_u = uh[3 .* (mesh.cells[cell].-1) .+ 1]
            correction += mesh.volumes[cell]*dot(local_u,(scalar_reference.B-scalar_reference.M)*local_u)
        end
    elseif name == :energy
        correction = (dot(uh,(coarse.K-coarse.S)*uh)-dot(U,enriched.K*U))/2
    end
    return (;estimate=S+load+correction,stabilization=S,load_correction=load,goal_correction=correction)
end

function elasticity_estimates(enriched,P,coarse,bu,bz,U,Z,F,goal,name;
        checkpoints=(4,8,16,32),direct=false)
    free = enriched.free
    A = enriched.K[free,free]
    prolong = P[free,coarse.free]
    diagonal = enriched.diagonal[free]
    precondition(r) = r./diagonal+prolong*(coarse.factor\(transpose(prolong)*r))
    consistency = dot(F,Z)-dot(U,enriched.K*Z)
    primal_work = dot(F,U)-dot(U,enriched.K*U)
    us = hierarchical_pcg_checkpoints(A,bu[free],precondition;checkpoints)
    zs = hierarchical_pcg_checkpoints(A,bz[free],precondition;checkpoints)
    results = NamedTuple[]
    function record(method,steps,small_u,small_z,primal_residual,dual_residual)
        du,dz = zeros(length(U)),zeros(length(Z))
        du[free],dz[free] = small_u,small_z
        estimate = if name == :energy
            primal_work+dot(bu,du)-dot(small_u,A*small_u)/2
        else
            consistency+dot(bu,dz)+dot(bz,du)-dot(small_u,A*small_z)+goal.remainder(du)
        end
        push!(results,(;method,steps,estimate,primal_residual,dual_residual))
    end
    du,dz = bu[free]./diagonal,bz[free]./diagonal
    record(:diagonal,0,du,dz,norm(bu[free]-A*du)/norm(bu[free]),norm(bz[free]-A*dz)/norm(bz[free]))
    for (u,z) in zip(us,zs)
        record(:pcg,u.step,u.x,z.x,u.relative_residual,z.relative_residual)
    end
    if direct
        factor = cholesky(Symmetric(A))
        du,dz = factor\bu[free],factor\bz[free]
        record(:direct,0,du,dz,norm(bu[free]-A*du)/norm(bu[free]),norm(bz[free]-A*dz)/norm(bz[free]))
    end
    return results
end

"""Independent coupled-operator transfer test; no Poisson completion weights."""
function run_elasticity_control(;levels=(4,8),modes=(1,3),order=6,
        goals=(:integral_u,:integral_u2,:energy),checkpoints=(4,8,16,32),direct_max_n=4,
        output=joinpath(@__DIR__,"..","results","sdwr_elasticity_control"))
    BLAS.set_num_threads(1)
    mkpath(output)
    rows = NamedTuple[]
    reference = sine_uniform_reference(;order=4)
    for n in levels
        println("ELASTICITY_PREPARE ",(;n)); flush(stdout)
        mesh = build_mesh(:cube,n)
        fan = build_fan(mesh)
        geometry = hierarchical_geometry(fan)
        coarse = elasticity_coarse_operator(mesh,1.0,1.0)
        enriched = elasticity_enriched_operator(fan,geometry,1.0,1.0)
        P = kron(hierarchical_prolongation(mesh,fan,geometry),sparse(Matrix{Float64}(I,3,3)))
        for mode in modes
            problem = elasticity_problem(mode)
            source = elasticity_loads(mesh,fan,geometry,coarse,problem;order)
            uh = elasticity_solve(coarse,source.projected)
            U = P*uh
            bu = source.enriched-enriched.K*U
            coarse_residual = norm((coarse.K*uh-source.projected)[coarse.free])/norm(source.projected[coarse.free])
            for name in goals
                goal = elasticity_goal(enriched,U,problem,name)
                zh = elasticity_solve(coarse,transpose(P)*goal.rhs)
                Z = P*zh
                bz = goal.rhs-enriched.K*Z
                error = goal.exact-goal.value
                baseline = elasticity_sdwr(mesh,coarse,enriched,source,uh,zh,U,Z,name,reference)
                results = vcat([(;method=:sdwr,steps=0,estimate=baseline.estimate,
                    primal_residual=coarse_residual,dual_residual=NaN)],
                    elasticity_estimates(enriched,P,coarse,bu,bz,U,Z,source.enriched,goal,name;
                        checkpoints,direct=n<=direct_max_n))
                for result in results
                    row = (;n,mode,goal=String(name),method=String(result.method),steps=result.steps,
                        lambda=problem.lambda,mu=problem.mu,cells=length(mesh.cells),
                        coarse_unknowns=length(coarse.free),enriched_unknowns=length(enriched.free),
                        exact_goal=goal.exact,reconstructed_goal=goal.value,signed_goal_error=error,
                        relative_goal_error=abs(error/goal.exact),signed_estimate=result.estimate,
                        signed_effectivity=result.estimate/error,primal_residual=result.primal_residual,
                        dual_residual=result.dual_residual,quadrature_order=order)
                    push!(rows,row)
                    write_csv(joinpath(output,"accuracy.csv"),rows)
                    println("ELASTICITY ",(;n,mode,goal=name,method=result.method,steps=result.steps,
                        effectivity=row.signed_effectivity)); flush(stdout)
                end
            end
        end
        GC.gc()
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_elasticity_control()
end
