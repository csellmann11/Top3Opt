# julia --project=. --startup-file=no Tests/regularization_comparison.jl [saved-mesh.bin]
# The optional serialized mesh is produced by an oversized-stencil failure in
# regularization_benchmark.jl; without it, compare a deterministic graded mesh.
include("regularization_benchmark.jl")

function compare_operators(cv,states,pars)
    for scheme in (:diamond,:tpfa,:taylor)
        build() = scheme == :taylor ? compute_reference_taylor_mat(cv,states,pars) :
            compute_flux_operator_mat(cv,states,pars;scheme)
        build() # compilation and warm-up
        timings = [@timed build() for _ in 1:5]
        best = timings[argmin(getproperty.(timings,:time))]
        R = best.value
        println((scheme=scheme,cells=length(states.χ_vec),seconds=best.time,
            allocated_MiB=best.bytes/2.0^20,stored_MiB=Base.summarysize(R)/2.0^20,
            nonzeros=nnz(R),row_bound=maximum(vec(sum(abs,R;dims=2))),
            constant_error=norm(R*ones(length(states.χ_vec)),Inf),
            conservation_error=norm(states.area_vec'*R,Inf)))
    end
end

function comparison_main(args)
    if isempty(args)
        mesh = create_rectangular_mesh(8,4,4,2.,1.,1.,StandardEl{1})
        for _ in 1:3
            cv = CellValues{3}(mesh)
            states = DesignVarInfo{3}(cv,0.3)
            for el in collect(RootIterator{4}(mesh.topo))
                x = states.x_vec[get_state_id(states,el.id)]
                norm(x-SA[1.,0.5,0.5]) < 0.3 && _refine!(el,mesh.topo)
            end
            mesh = Mesh(mesh.topo,StandardEl{1}())
        end
        cv = CellValues{3}(mesh)
        compare_operators(cv,DesignVarInfo{3}(cv,0.3),(β0=1.,))
    else
        cv,states,pars = deserialize(args[1])
        compare_operators(cv,states,pars)
    end
end

comparison_main(ARGS)
