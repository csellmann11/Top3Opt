using JLD2

# --- Configuration ---
data_dir = joinpath(@__DIR__, "jld2_files") 
output_csv = joinpath(@__DIR__, "all_times.csv")

# --- Regex to parse filenames ---
# Expected format: MBB...r3...a1...csv
r_pattern = r"r(\d+)"
a_pattern = r"a(\d+)"

# Store all rows here
# Tuple format: (RefLevel, MeshType, IsAdaptive, Assembly, Solve, Update, Adaptivity)
all_rows = []

println("Processing files in $data_dir...")

if isdir(data_dir)
    for filename in filter(f -> endswith(f, ".jld2"), readdir(data_dir))
        # 1. Parse Filename
        r_match = match(r_pattern, filename)
        a_match = match(a_pattern, filename)
        
        if r_match === nothing || a_match === nothing
            continue
        end
        
        r_val = parse(Int, r_match.captures[1])
        a_val = parse(Int, a_match.captures[1]) # 0 or 1
        
        # 2. Determine Mesh Info
        is_adaptive = a_val # 1 if adaptive, 0 if constant
        mesh_type_str = is_adaptive == 1 ? "Adaptive" : "Constant"

        # 3. Load Data
        try
            full_path = joinpath(data_dir, filename)
            # Load only the specific struct we need
            loaded_data = load(full_path, "sim_results")
            times = loaded_data.simulation_times
            
            push!(all_rows, (
                r_val,
                mesh_type_str,
                is_adaptive,
                times.assembly_time,
                times.solve_time,
                times.state_update_time,
                times.adaptivity_time
            ))
        catch e
            println("  Warning: Failed to read $filename")
        end
    end
else
    println("Error: Directory not found.")
end

# 4. Sort (by RefLevel, then by IsAdaptive)
sort!(all_rows, by = x -> (x[1], x[3]))

# 5. Write to Single CSV
open(output_csv, "w") do io
    println(io, "RefLevel,MeshType,IsAdaptive,Assembly,Solve,Update,Adaptivity")
    for row in all_rows
        println(io, join(row, ","))
    end
end

println("Successfully created $output_csv")