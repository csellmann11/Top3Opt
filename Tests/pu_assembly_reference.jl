# Frozen scalar assembly used to validate the optimized block kernel.
function reference_pu3d_add_tetra!(w,c,index,hat,map)
    t = c.geometry.tets[index]
    stress = c.stress[index]
    measure = c.scale[t.cell]*t.volume
    @inbounds for i in 1:10
        rows = w.dofs[map[i]]
        iszero(rows) && continue
        gradient,weighted_gradient = pu3d_internal_gradients(t,i,hat)
        internal = t.volume*(stress*gradient)
        weighted_internal = t.volume*(stress*weighted_gradient)
        for component in 1:3
            row = rows[component]
            row == 0 && continue
            localrow = 3i-3+component
            force,weighted_force = pu3d_load_entry(c.loads,index,localrow,hat)
            w.rhs[row] += force-internal[component]
            w.weighted[row] += weighted_force-weighted_internal[component]
            for j in 1:10
                columns = w.dofs[map[j]]
                for other in 1:3
                    column = columns[other]
                    0 < column <= row || continue
                    w.matrix[row,column] += measure*pu3d_stiffness_entry(t,i,j,
                        component,other,c.lambda,c.mu)
                end
            end
        end
    end
    return t.volume*(c.energy[index]-c.projected_energy[t.cell])/8
end

