isdefined(@__MODULE__, :parse_commandline) || include("utils/general_utils.jl")
using SHA: sha256

"""Stable, readable run identity including density and estimator controls.
The deprecated Laplace-rescaling switch has no effect and is deliberately omitted.
"""
function generate_setting_hash(args=parse_commandline())
    settings = (
        "s" => args["max_opt_steps"], "r" => args["max_ref_level"],
        "m" => args["mesh_type"], "a" => Int(args["do_adaptivity"]),
        "b" => Int(args["do_adaptivity_at_the_start"]), "c" => args["b_case"],
        "d" => Int(args["density_marking"]), "f" => get(args,"flux_scheme",:tpfa),
        "e" => get(args,"error_estimator",:sdwr), "q" => get(args,"pu_marking_exponent",1.0),
        "g" => get(args,"regularization",:mean), "u" => get(args,"update_mode",:explicit),
        "p" => Int(get(args,"projection",false)), "R" => get(args,"filter_radius",0.0),
        "n" => get(args,"density_substeps",0), "t" => get(args,"filter_tolerance",1e-12),
        "v" => get(args,"density_activity_tolerance",1e-3), "S" => get(args,"solver",:petsc),
    )
    encoded = join((string(key,'=',value) for (key,value) in settings),';')
    # Appended only when changed, so identities of default-parameter runs are stable.
    beta0, eta0 = get(args,"beta0",1.0), get(args,"eta0",15.0)
    beta0 == 1.0 || (encoded *= string(";B=",beta0))
    eta0 == 15.0 || (encoded *= string(";E=",eta0))
    fingerprint = bytes2hex(sha256(encoded))[1:16]
    return string("s",args["max_opt_steps"],"r",args["max_ref_level"],
        "m",first(string(args["mesh_type"])),"a",Int(args["do_adaptivity"]),
        "g",get(args,"regularization",:mean),"p",Int(get(args,"projection",false)),
        '_',fingerprint)
end

show_setting_hash() = display(generate_setting_hash())

abspath(PROGRAM_FILE) == (@__FILE__) && show_setting_hash()
