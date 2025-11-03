using Chairmarks
const to = TimerOutput()

num_points_per_dir = 10

function get_velocity_at_cell_faces_in_x(u_matrix::MArray, i::Int)
    # find the velocity at the cell faces
    # use integer quotient (0-based)

    # @timeit to "get velocity at cell faces in x" begin
    begin

        node_row = cld(i,2)
        u_at_fe = zeros(Float64, size(u_matrix, 2))
        if node_row == 1
            u_at_fe .= @views (3/4 .* u_matrix[node_row, :, 1] .+ 1/4 .* u_matrix[node_row+1, :, 1])
        else
            u_at_fe .= @views(1/4 .* u_matrix[node_row, :, 1] .+ 3/4 .* u_matrix[node_row+1, :, 1])
        end

        # create u with midpoints between each entry of u_at_fe corresponding to the cell faces for each quadrature point
        n = length(u_at_fe)
        u = zeros(Float64, 2*n - 1)
        u[1:2:end] .= u_at_fe
        if n > 1
            u[2:2:end] .= 0.5 .* @views (u_at_fe[1:end-1] .+ u_at_fe[2:end])
        end
    end

    return u
end

ux = @SVector zeros(Float64, 10+1)
u_matrix = @MMatrix zeros(Float64, 10+1, 3)
@timeit to "get velocity" ux = get_velocity_at_cell_faces_in_x(u_matrix, 1)


show(to)  

@b get_velocity_at_cell_faces_in_x($u_matrix, 1)