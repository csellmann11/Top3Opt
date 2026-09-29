"""
    regularization_faces(cv, states)

Collect active leaf faces and their incident states. Iterating leaf faces is
essential: each subface of a coarse/fine interface has its own flux.
"""
function regularization_faces(cv, states)
    owners = Dict{Int,Vector{Int}}()
    for (el_id, sid) in states.el_id_to_state_id
        Ju3VEM.VEMGeo.iterate_volume_areas(cv.facedata_col, cv.mesh.topo, el_id) do face, _, _
            push!(get!(owners, face.id, Int[]), sid)
        end
    end
    all(s -> 1 <= length(s) <= 2, values(owners)) ||
        error("Regularization requires manifold faces with one or two incident cells")
    return owners
end

"""Cell-to-vertex linear reconstruction, with equal-value mirrored boundary samples."""
function build_diamond_node_weights(cv, states, owners)
    nodes = cv.mesh.topo.nodes
    samples = [Dict{Int,SVector{3,Float64}}() for _ in eachindex(nodes)]
    ghosts = [Tuple{Int,SVector{3,Float64}}[] for _ in eachindex(nodes)]
    for (fid, sids) in owners
        fd = cv.facedata_col[fid]
        ids = fd.face_node_ids.v.args[1]
        for sid in sids, nid in ids
            samples[nid][sid] = states.x_vec[sid]
        end
        if length(sids) == 1
            sid = only(sids)
            x = states.x_vec[sid]
            n = Ju3VEM.VEMGeo.get_outward_normal(x, fd)
            ghost = x + 2dot(nodes[first(ids)] - x, n)*n
            for nid in ids
                push!(ghosts[nid], (sid, ghost))
            end
        end
    end
    weights = [Tuple{Int,Float64}[] for _ in eachindex(nodes)]
    for nid in eachindex(nodes)
        isempty(samples[nid]) && continue
        s = vcat(collect(pairs(samples[nid])), [sid => x for (sid,x) in ghosts[nid]])
        h = maximum(norm(x - nodes[nid]) for (_,x) in s)
        A = [j == 1 ? 1.0 : (x[j-1] - nodes[nid][j-1])/h for (_,x) in s, j in 1:4]
        # SVD handles rank-deficient boundary stencils without a ridge bias.
        w = pinv(A)[1,:]
        w ./= sum(w)
        weights[nid] = [(s[i].first, w[i]) for i in eachindex(s)]
    end
    return weights
end

"""
    diamond_face_coefficients(points, xL, xR, betaL, betaR; diamond=true)

Return transmissibility T and nodal correction coefficients c for the integrated
outward flux `T*(chiR-chiL) + sum(c .* chi_nodes)`. The polygon boundary integral
reconstructs both tangential gradient components in 3D. Harmonic beta uses the
two normal centroid-to-face distances. Vertices must follow the face boundary.
"""
function diamond_face_coefficients(points, xL, xR, betaL, betaR; diamond=true)
    origin = first(points)
    av = zero(xL)
    for i in eachindex(points)
        av += cross(points[i]-origin, points[mod1(i+1,length(points))]-origin)/2
    end
    area = norm(av)
    area > 0 || throw(ArgumentError("Degenerate regularization face"))
    polygon_normal = av/area
    d = xR-xL
    n = dot(d,polygon_normal) > 0 ? polygon_normal : -polygon_normal
    deltaL = dot(origin-xL,n)
    deltaR = dot(xR-origin,n)
    deltaL > 0 && deltaR > 0 ||
        throw(ArgumentError("Cell centers must lie on opposite sides of each face"))
    betaL >= 0 && betaR >= 0 || throw(ArgumentError("beta must be nonnegative"))
    T = betaL == 0 || betaR == 0 ? 0.0 : area/(deltaL/betaL + deltaR/betaR)
    correction = zeros(length(points))
    if diamond
        for i in eachindex(points)
            j = mod1(i+1,length(points))
            # Integral of tangential grad chi = integral_boundary chi * conormal.
            c = -T * dot(d, cross(points[j]-points[i],polygon_normal))/(2area)
            correction[i] += c
            correction[j] += c
        end
    end
    return T, correction
end

"""
    compute_flux_operator_mat(cv, states, sim_pars; scheme=:diamond)

Conservative approximation of div(beta_hat grad chi), beta_hat = 2*beta0*h^2.
The global p_avg factor is applied by state_update!, not assembled here.
Boundary faces have zero flux. Each interior flux is assembled once with opposite
signs and divided by the adjacent cell volumes. :tpfa omits the nonorthogonal
correction; :diamond uses reconstructed face vertex values as in the 2D scheme.
Rebuild after mesh changes. The corrected operator need not be symmetric or monotone.
"""
function compute_flux_operator_mat(cv, states, sim_pars; scheme::Symbol=:diamond)
    scheme in (:diamond, :tpfa) || throw(ArgumentError("Unknown flux scheme: $scheme"))
    sim_pars.β0 >= 0 || throw(ArgumentError("beta0 must be nonnegative"))
    owners = regularization_faces(cv, states)
    nw = scheme == :diamond ? build_diamond_node_weights(cv,states,owners) : nothing
    rows = Int[]; cols = Int[]; vals = Float64[]
    for (fid, sids) in owners
        length(sids) == 2 || continue
        l,r = sids
        ids = cv.facedata_col[fid].face_node_ids.v.args[1]
        points = [cv.mesh.topo.nodes[nid] for nid in ids]
        T,c = diamond_face_coefficients(points, states.x_vec[l], states.x_vec[r],
            2sim_pars.β0*states.h_vec[l]^2, 2sim_pars.β0*states.h_vec[r]^2;
            diamond=scheme == :diamond)
        flux = Dict(l => -T, r => T)
        if scheme == :diamond
            for (i,nid) in enumerate(ids), (sid,w) in nw[nid]
                flux[sid] = get(flux,sid,0.0) + c[i]*w
            end
        end
        for (sid,v) in flux
            push!(rows,l); push!(cols,sid); push!(vals,v/states.area_vec[l])
            push!(rows,r); push!(cols,sid); push!(vals,-v/states.area_vec[r])
        end
    end
    N = length(states.χ_vec)
    return sparse(rows,cols,vals,N,N)
end
