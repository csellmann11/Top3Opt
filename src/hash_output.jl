include("general_utils.jl")



function show_setting_hash()

    args = parse_commandline()
    
    MAX_OPT_STEPS = args["max_opt_steps"]
    MAX_REF_LEVEL = args["max_ref_level"]
    MeshType = args["mesh_type"]
    do_adaptivity = args["do_adaptivity"]
    do_adaptivity_at_the_start = args["do_adaptivity_at_the_start"]
    b_case = args["b_case"]
    
    setting_hash = hash((MAX_OPT_STEPS,
        MAX_REF_LEVEL,MeshType,do_adaptivity,
        do_adaptivity_at_the_start,b_case))

    display(setting_hash)
end
show_setting_hash()