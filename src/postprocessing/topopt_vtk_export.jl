using Ju3VEM.VEMUtils: trail_zeros3,vtk_volume_helper
using Ju3VEM.VEMGeo: get_iterative_area_vertex_ids, iterate_volume_areas, get_unique_values
using Ju3VEM.VEMUtils: WriteVTK
using Ju3VEM.VEMUtils.WriteVTK: VTKPolyhedron, VTKPointData, VTKCellData, vtk_grid

# Build the XML with appended raw data. We compute offsets based on payload sizes.
# Each appended block is: [UInt64 payload_bytes][payload_bytes]
struct _VTUBlock
    name::String
    bytes::UInt64
    writer::Function
end

function vtk_volume_helper_new(cv::CellValues{D,U}, 
    volume::Volume{D},
    eldata_col::Dict{Int,<:ElData},
    node_map::Vector{Int}) where {D,U}

    fdc = cv.facedata_col
    topo = cv.mesh.topo
    area_ids  = Int[]
    iterate_volume_areas(fdc,topo,volume.id) do area, _ ,_
        push!(area_ids, area.id)
    end
    unique_vol_node_ids = node_map[eldata_col[volume.id].node_ids]
    vol_area_node_ids = [
        node_map[fdc[area_id].face_node_ids |> get_first]
        for area_id in area_ids
    ]
    return unique_vol_node_ids,vol_area_node_ids
end

@inline function write_vtk_polyhedron(
    node_ids,faces_node_ids::AbstractVector{<:AbstractVector{Int}}
)
    VTKPolyhedron(node_ids,faces_node_ids...)
end

function create_volume_cells(cv::CellValues{D,U},
    eldata_col::Dict{Int,<:ElData},
    node_map::Vector{Int}) where {D,U}

    topo = cv.mesh.topo
    vol_data = Dict(
        volume.id => vtk_volume_helper_new(cv,volume,eldata_col,node_map)
        for volume in RootIterator{4}(topo)
    )


    volume_cells = if D ≥ 3 
        [
            write_vtk_polyhedron(vol_data[volume.id][1],vol_data[volume.id][2])
            for volume in RootIterator{D,4}(topo)
        ]
    else
        nothing
    end

    return volume_cells
end

function write_vtu_file(cv::CellValues{D,U},
    eldata_col::Dict{Int,<:ElData},
    filename::String = "vtk/sol_to_vtk",
    u::Union{AbstractVector,Nothing} = nothing;
    cell_data_col::P = ()
    ) where {D,U,P<:Tuple}

    dh = cv.dh
    topo = cv.mesh.topo

    raw_points = filter(is_active,get_nodes(topo))



    points = reduce(hcat,get_coords.(raw_points))

    if U == 2 
        points = vcat(points,zeros(1,size(points,2)))
    end



    node_map = zeros(Int,get_nodes(topo) |> length)
    for (i,node) in enumerate(raw_points)
        node_map[get_id(node)] = i
    end
    


    volume_cells = create_volume_cells(cv,eldata_col,node_map)


    if dh !== nothing && u !== nothing
        u_processed = Vector{SVector{3,Float64}}(undef,length(dh.dof_mapping))
        for (n_id,n_dofs) in dh.dof_mapping
            idx = node_map[n_id]

            u_processed[idx] = trail_zeros3(u[n_dofs])
        end
    else
        u_processed = nothing
    end

    vtk_grid(filename,points,volume_cells ; append = false) do vtk
        if u_processed !== nothing
            vtk["u", VTKPointData()] = u_processed
        end
        
        if !isempty(cell_data_col)
            for i in eachindex(cell_data_col)
                vtk["cell_data_$i", VTKCellData()] = cell_data_col[i]
            end
        end

        if D == 3 
            vtk["volume_id", VTKCellData()] = [volume.id for volume in RootIterator{D,4}(topo)]
        elseif D == 2
            vtk["area_id", VTKCellData()] = [area.id for area in RootIterator{D,3}(topo)]
        else
            throw(ErrorException("Unsupported dimension: $D"))
        end
    end
    
    nothing
end


