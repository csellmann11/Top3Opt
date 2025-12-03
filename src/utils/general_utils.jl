using ArgParse 


function def_rhs_fun(x)
    SA[0.0, 0.0, 0.0]
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--max_opt_steps","-s"
            help = "Maximum number of optimization steps"
            arg_type = Int
            default  = 1
        "--max_ref_level","-r"
            help = "Maximum refinement level"
            arg_type = Int
            default  = 5
        "--mesh_type","-m"
            help = "Mesh type"
            arg_type = Symbol
            default  = :Hexahedra # :Tetrahedra # :Voronoi
        "--do_adaptivity","-a"
            help = "Boolean to determine if adaptivity is enabled"
            arg_type = Bool
            default  = false
        "--do_adaptivity_at_the_start","-b"
            help = "if true, the mesh is refined to the finest level at the start"
            arg_type = Bool
            default  = true
        "--b_case","-c"
            help = "Case name"
            arg_type = Symbol
            default  = :L_cantilever #[:simple_lever,:MBB_sym,:Cantilever_sym,:Bending_Beam_sym,:pressure_plate,:L_cantilever]
        "--density_marking","-d"
            help = "Toogle if elements are marked for refinement if density is growing"
            arg_type = Bool
            default  = true
        "--laplace_rescale","-l"
            help = "Toogle if the distance between two nodes is rescaled for the computation of the laplace operator"
            arg_type = Bool
            default  = true
        "--rhs_fun","-f"
            help = "Right hand side function"
            arg_type = Function
            default  = def_rhs_fun 
    end
    return parse_args(s)


end