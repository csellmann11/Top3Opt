"""Reference matrices for the existing cube operator and continuous P1 fan.

Only translation and uniform scaling are reused. The operator, reconstruction,
projected body load and polynomial proxy are unchanged from the accuracy lab.
No modal solution formula or fitted estimator coefficient is used.
"""
function sine_uniform_reference(;order=6)
    mesh = build_mesh(:cube,1)
    fan = build_fan(mesh)
    data = poisson_cell_operator(mesh,1)
    reconstruction = hcat((reconstruct_fan(mesh,fan,Float64.((1:8) .== j)) for j in 1:8)...)
    C = data.volume*transpose(data.G)*data.G
    D = Matrix{Float64}(I,8,8)-data.P
    S = data.alpha*transpose(D)*D
    M,B,H = zeros(8,8),zeros(8,8),zeros(8,8)
    lT,lP = zeros(8),zeros(8)
    rule = tetra_rule(order)
    nq = length(fan.volumes)*length(rule.weights)
    points = Matrix{Float64}(undef,3,nq)
    weights = Vector{Float64}(undef,nq)
    shapeT,shapeP = Matrix{Float64}(undef,8,nq),Matrix{Float64}(undef,8,nq)
    cursor = 0
    for tet in eachindex(fan.volumes)
        ids = fan.tet_nodes[:,tet]
        xyz = fan.points[:,ids]
        T = reconstruction[ids,:]
        P = fill(1/8,4,8)+(transpose(xyz).-transpose(data.center))*data.G
        volume = fan.volumes[tet]
        mass = (volume/20)*(ones(4,4)+Matrix{Float64}(I,4,4))
        M .+= transpose(T)*mass*T
        B .+= transpose(T)*mass*P
        gradients = fan.gradients[tet]*T
        H .+= volume*transpose(gradients)*gradients
        lT .+= transpose(T)*fill(volume/4,4)
        lP .+= transpose(P)*fill(volume/4,4)
        count = length(rule.weights)
        slots = cursor+1:cursor+count
        points[:,slots] .= xyz*rule.barycentric
        weights[slots] .= volume.*rule.weights
        shapeT[:,slots] .= transpose(T)*rule.barycentric
        shapeP[:,slots] .= transpose(P)*rule.barycentric
        cursor += count
    end
    isapprox(sum(weights),1.0;atol=1e-13) || error("Reference volume mismatch")
    isapprox(M*ones(8),lT;atol=1e-13) || error("Reference mass mismatch")
    isapprox(C+S,data.K;atol=1e-13) || error("Reference stiffness mismatch")
    return (;K=data.K,C,S,M,B,H,lT,lP,points,weights,shapeT,shapeP,
        order,mesh,fan,reconstruction)
end

sine_uniform_vertex(i,j,k,n) = 1+i+(n+1)*j+(n+1)^2*k

"""Assemble and factor the actual sparse uniform-cube stiffness matrix."""
function sine_uniform_system(n::Integer,reference)
    n >= 2 || throw(ArgumentError("At least two cells per axis are required"))
    n = Int(n)
    h,nv,nc = 1/n,(n+1)^3,n^3
    points = Matrix{Float64}(undef,3,nv)
    boundary = falses(nv)
    for k in 0:n,j in 0:n,i in 0:n
        node = sine_uniform_vertex(i,j,k,n)
        points[:,node] .= (i*h,j*h,k*h)
        boundary[node] = i == 0 || i == n || j == 0 || j == n || k == 0 || k == n
    end
    cells = Matrix{Int}(undef,8,nc)
    origins = Matrix{Float64}(undef,3,nc)
    rows,columns = Vector{Int}(undef,64nc),Vector{Int}(undef,64nc)
    values = Vector{Float64}(undef,64nc)
    cursor = 0
    for k in 0:n-1,j in 0:n-1,i in 0:n-1
        cell = 1+i+n*j+n^2*k
        origins[:,cell] .= (i*h,j*h,k*h)
        for (a,offset) in enumerate(POISSON_HEX_OFFSETS)
            cells[a,cell] = sine_uniform_vertex(i+offset[1],j+offset[2],k+offset[3],n)
        end
        for b in 1:8,a in 1:8
            cursor += 1
            rows[cursor],columns[cursor] = cells[a,cell],cells[b,cell]
            values[cursor] = h*reference.K[a,b]
        end
    end
    K = sparse(rows,columns,values,nv,nv)
    free = findall(!,boundary)
    factor = cholesky(Symmetric(K[free,free]))
    return (;n,h,points,cells,origins,boundary,free,K,factor)
end

"""Apply an element matrix through ordinary local gathering and scattering."""
function sine_uniform_apply(system,local_matrix,scale,v)
    size(local_matrix) == (8,8) || throw(DimensionMismatch("Expected 8-by-8 element matrix"))
    length(v) == size(system.points,2) || throw(DimensionMismatch("Wrong vertex vector length"))
    result = zeros(length(v))
    local_v,local_result = zeros(8),zeros(8)
    for cell in axes(system.cells,2)
        for a in 1:8
            local_v[a] = v[system.cells[a,cell]]
        end
        mul!(local_result,local_matrix,local_v)
        for a in 1:8
            result[system.cells[a,cell]] += scale*local_result[a]
        end
    end
    return result
end

function sine_uniform_load(system,local_vector,scale)
    length(local_vector) == 8 || throw(DimensionMismatch("Expected eight element loads"))
    result = zeros(size(system.points,2))
    for cell in axes(system.cells,2),a in 1:8
        result[system.cells[a,cell]] += scale*local_vector[a]
    end
    return result
end

"""Solve with homogeneous Dirichlet data and retain the measured residual."""
function sine_uniform_solve(system,rhs)
    u = zeros(length(rhs))
    u[system.free] = system.factor\rhs[system.free]
    residual = system.K*u-rhs
    algebraic_residual = norm(residual[system.free])/max(norm(rhs[system.free]),eps(Float64))
    return (;u,rhs,algebraic_residual)
end

"""Integrate eight separated sine/cosine products against both local fields.

For x=x0+h*xi, sin(k*pi*x) is sin(k*pi*x0)*cos(k*pi*h*xi)
+ cos(k*pi*x0)*sin(k*pi*h*xi). The eight products allow the same reference
quadrature to be used for every translated cell without approximating f.
"""
function sine_uniform_source_moments(system,reference,modes)
    length(modes) == 3 && all(k->k isa Integer && k > 0,modes) ||
        throw(ArgumentError("Use three positive integer frequencies"))
    nq = length(reference.weights)
    products = Matrix{Float64}(undef,8,nq)
    for q in 1:nq
        sine = ntuple(d->sinpi(modes[d]*system.h*reference.points[d,q]),3)
        cosine = ntuple(d->cospi(modes[d]*system.h*reference.points[d,q]),3)
        for mask in 0:7
            products[mask+1,q] = reference.weights[q]*
                prod(((mask >> (d-1)) & 1) == 1 ? sine[d] : cosine[d] for d in 1:3)
        end
    end
    return (;projected=reference.shapeP*transpose(products),
        reconstructed=reference.shapeT*transpose(products))
end

"""Assemble F(Pi phi) and F(T phi) for the manufactured sine source."""
function sine_uniform_source(system,reference,modes)
    moments = sine_uniform_source_moments(system,reference,modes)
    lambda = pi^2*sum(abs2,modes)
    scale = system.h^3*lambda
    bP,bT = zeros(size(system.points,2)),zeros(size(system.points,2))
    coefficients,localP,localT = zeros(8),zeros(8),zeros(8)
    for cell in axes(system.cells,2)
        sine = ntuple(d->sinpi(modes[d]*system.origins[d,cell]),3)
        cosine = ntuple(d->cospi(modes[d]*system.origins[d,cell]),3)
        for mask in 0:7
            coefficients[mask+1] = prod(((mask >> (d-1)) & 1) == 1 ?
                cosine[d] : sine[d] for d in 1:3)
        end
        mul!(localP,moments.projected,coefficients)
        mul!(localT,moments.reconstructed,coefficients)
        for a in 1:8
            node = system.cells[a,cell]
            bP[node] += scale*localP[a]
            bT[node] += scale*localT[a]
        end
    end
    return (;projected=bP,reconstructed=bT,bP,bT,lambda,quadrature_order=reference.order)
end
