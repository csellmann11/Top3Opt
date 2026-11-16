isdefined(@__MODULE__, :parse_commandline) || include("utils/general_utils.jl")
using SHA: sha256

"""Regularization label of a run: the scaling of a gradient penalty or `helmholtz`.
It equals the former `--regularization` value, so earlier identities are kept."""
regularization_label(args) = get(args,"regularization",:gradient) == :helmholtz ? :helmholtz :
    get(args,"scaling",:mean)

"""Stable, readable run identity including density and estimator controls.
The deprecated Laplace-rescaling switch has no effect and is deliberately omitted.
Scaling and integrator are encoded with the former `regularization`/`update_mode`
values; damping and a finite move limit are appended only when used, so the
identities of runs without them are unchanged. A finite move limit is encoded as
`;M=<value>box`: the per-cell box replaced the former eta-increase retry, so these
runs do not reuse directories of the old rule. Projected runs carry the
continuation rule (`;C=stall`), since `advance_on_stall` changed their beta schedule.
Explicit gradient runs with zero requested substeps carry `;N=cfl`: their count now
comes from the measured CFL limit alone instead of the former fixed formula.
"""
function generate_setting_hash(args=parse_commandline())
    integrator = get(args,"integrator",:explicit) == :semi_implicit ? :implicit : :explicit
    settings = (
        "s" => args["max_opt_steps"], "r" => args["max_ref_level"],
        "m" => args["mesh_type"], "a" => Int(args["do_adaptivity"]),
        "b" => Int(args["do_adaptivity_at_the_start"]), "c" => args["b_case"],
        "d" => Int(args["density_marking"]), "f" => get(args,"flux_scheme",:tpfa),
        "e" => get(args,"error_estimator",:sdwr), "q" => get(args,"pu_marking_exponent",1.0),
        "g" => regularization_label(args), "u" => integrator,
        "p" => Int(get(args,"projection",false)), "R" => get(args,"filter_radius",0.0),
        "n" => get(args,"density_substeps",0), "t" => get(args,"filter_tolerance",1e-12),
        "v" => get(args,"density_activity_tolerance",1e-3), "S" => get(args,"solver",:petsc),
    )
    encoded = join((string(key,'=',value) for (key,value) in settings),';')
    # Appended only when changed, so identities of default-parameter runs are stable.
    beta0, eta0 = get(args,"beta0",3.0), get(args,"eta0",15.0)
    beta0 == 3.0 || (encoded *= string(";B=",beta0))
    eta0 == 15.0 || (encoded *= string(";E=",eta0))
    get(args,"diag",false) && (encoded *= ";D=1")
    move_limit = get(args,"move_limit",Inf)
    isfinite(move_limit) && (encoded *= string(";M=",move_limit,"box"))
    # Projected runs continue beta on stalled progress (advance_on_stall); they
    # differ from earlier projected runs and must not share their directory.
    get(args,"projection",false) && (encoded *= ";C=stall")
    # Explicit gradient substeps from the measured CFL limit only (former fixed count removed).
    get(args,"regularization",:gradient) != :helmholtz && integrator == :explicit &&
        get(args,"density_substeps",0) == 0 && (encoded *= ";N=cfl")
    fingerprint = bytes2hex(sha256(encoded))[1:16]
    return string("s",args["max_opt_steps"],"r",args["max_ref_level"],
        "m",first(string(args["mesh_type"])),"a",Int(args["do_adaptivity"]),
        "g",regularization_label(args),"p",Int(get(args,"projection",false)),
        '_',fingerprint)
end

show_setting_hash() = display(generate_setting_hash())

abspath(PROGRAM_FILE) == (@__FILE__) && show_setting_hash()
