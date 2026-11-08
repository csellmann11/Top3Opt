"""Goal integration on exact tetrahedron/box intersections for unaligned meshes.

Rules use original-tetrahedron barycentric coordinates and physical weights.
All three consumers (goal, dual load, and estimator) share the same cut geometry.
"""
function sdwr_geometric_goal_rules(fan,goal)
    goal.name == :localized_u || return nothing
    return [sdwr_goal_tetra_rule(fan.points[:,fan.tet_nodes[:,tet]],goal)
        for tet in eachindex(fan.volumes)]
end

function sdwr_geometric_goal(mesh,fan,U,goal,rules)
    rules === nothing && return sdwr_reconstructed_goal(mesh,fan,U,goal)
    cell_values = zeros(length(mesh.cells))
    for tet in eachindex(fan.volumes)
        ids = view(fan.tet_nodes,:,tet)
        rule = rules[tet]
        xyz = fan.points[:,ids]*rule.barycentric
        for q in eachindex(rule.weights)
            u = dot(U[ids],view(rule.barycentric,:,q))
            cell_values[fan.tet_cells[tet]] += rule.weights[q]*goal.value(view(xyz,:,q),u)
        end
    end
    return (;value=sum(cell_values),cell_values)
end

function sdwr_geometric_goal_load(mesh,fan,U,goal,rules)
    rules === nothing && return sdwr_goal_load(mesh,fan,U,goal)
    nodal = zeros(length(U))
    for tet in eachindex(fan.volumes)
        ids = view(fan.tet_nodes,:,tet)
        rule = rules[tet]
        xyz = fan.points[:,ids]*rule.barycentric
        gradient = fan.gradients[tet]*U[ids]
        for q in eachindex(rule.weights)
            shape = view(rule.barycentric,:,q)
            x = view(xyz,:,q)
            u = dot(U[ids],shape)
            derivative = goal.derivative_value(x,u)
            derivative_gradient = goal.derivative_gradient(x,gradient)
            for hat in 1:4
                value = derivative*shape[hat]+dot(derivative_gradient,view(fan.gradients[tet],:,hat))
                nodal[ids[hat]] += rule.weights[q]*value
            end
        end
    end
    return sdwr_reconstruction_transpose(mesh,fan,nodal)
end
