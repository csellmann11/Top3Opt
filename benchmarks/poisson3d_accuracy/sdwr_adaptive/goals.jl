"""Goals for the reconstructed continuous field U, with positive adjoint sign.

The local boxes are aligned with the initial n=2 grid and every octree child.
No point evaluation, empirical scaling, or discrete J_h enters these goals.
"""
function sdwr_goal(name::Symbol, domain::Symbol)
    box = domain == :cube ? ((0.5,1.0),(0.5,1.0),(0.5,1.0)) :
        ((-1.0,-0.5),(0.5,1.0),(0.5,1.0))
    inside(x) = all(box[d][1] <= x[d] <= box[d][2] for d in 1:3)
    weight = name == :localized_u ? (x -> inside(x) ? 1.0 : 0.0) : (x -> 1.0)
    name in (:integral_u,:integral_u2,:localized_u) || error("Unknown goal $name")
    power = name == :integral_u2 ? 2 : 1
    value = (x,u) -> weight(x)*u^power
    derivative_value = (x,u) -> power*weight(x)*(power == 1 ? 1.0 : u)
    derivative_gradient = (x,gradient) -> (0.0,0.0,0.0)
    return (;name,domain,box,power,weight,value,derivative_value,derivative_gradient)
end

function sdwr_box_integral(f,box;order=24)
    rule = gauss_legendre01(order)
    result = 0.0
    for i in eachindex(rule.nodes),j in eachindex(rule.nodes),k in eachindex(rule.nodes)
        indices = (i,j,k)
        x = ntuple(d -> box[d][1]+(box[d][2]-box[d][1])*rule.nodes[indices[d]],3)
        weight = prod((box[d][2]-box[d][1])*rule.weights[indices[d]] for d in 1:3)
        result += weight*f(x)
    end
    return result
end

"""Independent polar goal integration; r=R*t^3 removes fractional powers."""
function sdwr_singular_goal_reference(power;order=48)
    rule = gauss_legendre01(order)
    result = 0.0
    z_integral = power == 1 ? 1/6 : 1/30
    for segment in 0:5,i in eachindex(rule.nodes),j in eachindex(rule.nodes)
        theta = (segment+rule.nodes[i])*pi/4
        cosine,sine = cos(theta),sin(theta)
        radius = 1/max(abs(cosine),abs(sine))
        t = rule.nodes[j]
        r = radius*t^3
        x,y = r*cosine,r*sine
        h = (1-x*x)*(1-y*y)*r^(2/3)*sin(2theta/3)
        result += rule.weights[i]*rule.weights[j]*(pi/4)*3radius^2*t^5*h^power*z_integral
    end
    return result
end

function sdwr_goal_reference(problem,goal;order=48)
    if goal.name == :localized_u
        fine = sdwr_box_integral(x->problem.exact(x),goal.box;order)
        coarse = sdwr_box_integral(x->problem.exact(x),goal.box;order=max(8,order÷2))
        return (;value=fine,uncertainty=abs(fine-coarse))
    elseif problem.name == :cube_smooth
        return (;value=goal.power == 1 ? 1/216 : 1/27000,uncertainty=0.0)
    elseif problem.name == :lshape_smooth
        return (;value=goal.power == 1 ? 1/96 : 3*(8/105)^2/30,uncertainty=0.0)
    end
    fine = sdwr_singular_goal_reference(goal.power;order)
    coarse = sdwr_singular_goal_reference(goal.power;order=max(8,order÷2))
    return (;value=fine,uncertainty=abs(fine-coarse))
end

"""Exact tetrahedral moments of the P1 reconstruction, including U squared.

Localized box interfaces must be mesh-aligned; check each tetrahedron instead
of integrating an unresolved discontinuous characteristic function.
"""
function sdwr_reconstructed_goal(mesh,fan,U,goal)
    cell_values = zeros(length(mesh.cells))
    for tet in eachindex(fan.volumes)
        ids = view(fan.tet_nodes,:,tet)
        values = U[ids]
        points = fan.points[:,ids]
        center = vec(sum(points;dims=2))/4
        weight = goal.weight(center)
        if goal.name == :localized_u
            for d in 1:3,cut in goal.box[d]
                lo,hi = extrema(view(points,d,:))
                lo < cut-1e-12 && hi > cut+1e-12 &&
                    error("Localized goal boundary cuts a tetrahedron")
            end
        end
        moment = goal.power == 1 ? sum(values)/4 :
            (sum(values)^2+sum(abs2,values))/20
        cell_values[fan.tet_cells[tet]] += weight*fan.volumes[tet]*moment
    end
    return (;value=sum(cell_values),cell_values)
end

"""Apply the transpose of the continuous fan reconstruction, without a matrix."""
function sdwr_reconstruction_transpose(mesh,fan,fan_load)
    nv,nf = fan.vertex_count,fan.face_count
    result = copy(fan_load[1:nv])
    for (face,ids) in enumerate(mesh.faces)
        value = fan_load[nv+face]/length(ids)
        for node in ids
            result[node] += value
        end
    end
    for (cell,ids) in enumerate(mesh.cells)
        value = fan_load[nv+nf+cell]/length(ids)
        for node in ids
            result[node] += value
        end
    end
    return result
end

"""Exact derivative d[J(Tu_h)]/du_h for the three benchmark goals.

The primal body load remains projected as in the original VEM benchmark.
The dual uses the derivative of the continuous reconstructed goal being
measured. This also avoids quadrature error from the U-dependent dual load.
"""
function sdwr_goal_load(mesh,fan,U,goal)
    nodal = zeros(length(U))
    for tet in eachindex(fan.volumes)
        ids = view(fan.tet_nodes,:,tet)
        center = vec(sum(fan.points[:,ids];dims=2))/4
        weight = goal.weight(center)
        total_u = sum(U[ids])
        for node in ids
            nodal[node] += weight*fan.volumes[tet]*(goal.power == 1 ?
                1/4 : (total_u+U[node])/10)
        end
    end
    return sdwr_reconstruction_transpose(mesh,fan,nodal)
end

"""Deterministic bulk marking using nonnegative estimator indicators only."""
function sdwr_mark(indicators;theta=0.5)
    0 < theta <= 1 || throw(ArgumentError("theta must lie in (0,1]"))
    all(x->isfinite(x)&&x>=0,indicators) || error("Invalid indicator")
    order = sortperm(indicators;rev=true,alg=MergeSort)
    target = theta*sum(indicators)
    marked = Int[]
    total = 0.0
    for cell in order
        push!(marked,cell)
        total += indicators[cell]
        total >= target && break
    end
    return marked
end
