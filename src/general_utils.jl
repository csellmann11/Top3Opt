using ArgParse 



function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--max_opt_steps","-s"
            help = "Maximum number of optimization steps"
            arg_type = Int
            default  = 2
        "--max_ref_level","-r"
            help = "Maximum refinement level"
            arg_type = Int
            default  = 3
        "--mesh_type","-m"
            help = "Mesh type"
            arg_type = Symbol
            default  = :Hexahedra
        "--do_adaptivity","-a"
            help = "Boolean to determine if adaptivity is enabled"
            arg_type = Bool
            default  = true
        "--do_adaptivity_at_the_start","-b"
            help = "if true, the mesh is refined to the finest level at the start"
            arg_type = Bool
            default  = true
    end
    return parse_args(s)


end