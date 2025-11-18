include("general_utils.jl")



function generate_setting_hash(
    args = parse_commandline())


    MAX_OPT_STEPS = args["max_opt_steps"]
    MAX_REF_LEVEL = args["max_ref_level"]
    MeshType = args["mesh_type"]
    do_adaptivity = args["do_adaptivity"]
    do_adaptivity_at_the_start = args["do_adaptivity_at_the_start"]
    b_case = args["b_case"]
    density_marking = args["density_marking"]
    laplace_rescale = args["laplace_rescale"]
    # setting_hash = hash((MAX_OPT_STEPS,
    #     MAX_REF_LEVEL,MeshType,do_adaptivity,
    #     do_adaptivity_at_the_start,b_case))
    hash = """
    s$(MAX_OPT_STEPS)
    r$(MAX_REF_LEVEL)
    m$(string(MeshType)[1])
    a$(Int(do_adaptivity))
    b$(Int(do_adaptivity_at_the_start))
    c$(string(b_case)[1])
    d$(Int(density_marking))
    l$(Int(laplace_rescale))
    """
    return filter(!isspace, hash)  # This line removes all whitespace
end

function show_setting_hash()

    args = parse_commandline()
    
    display(generate_setting_hash(args))
end
show_setting_hash()