include("../Tests/regularization_benchmark.jl")

function report(label, f)
    f()
    samples = [@timed(f()) for _ in 1:3]
    println((label=label, bytes=minimum(x.bytes for x in samples), seconds=minimum(x.time for x in samples), retained=Base.summarysize(last(samples).value)))
    flush(stdout)
end

function reinit_all(cv)
    for el in RootIterator{4}(cv.mesh.topo)
        reinit!(el.id,cv)
    end
    nothing
end

function run_audit()
    mesh = create_rectangular_mesh(12,2,4,3.,0.5,1.,StandardEl{1})
    for el in RootIterator{4}(mesh.topo)
        el.id % 3 == 0 && _refine!(el,mesh.topo)
    end
    mesh = Mesh(mesh.topo,StandardEl{1}())
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.3)
    println((cells=length(states.χ_vec), faces=length(cv.facedata_col), nodes=length(cv.dh.dof_mapping)))
    report("DofHandler", () -> DofHandler{3}(mesh))
    report("all face data", () -> Ju3VEM.VEMUtils._create_facedata_col(mesh))
    report("CellValues", () -> CellValues{3}(mesh))
    report("reinit all cells", () -> reinit_all(cv))
    report("fill_states", () -> fill_states!(states,cv,0.3))
    marker = zeros(Bool,length(get_volumes(mesh.topo)))
    report("no-op adapt_mesh", () -> adapt_mesh(cv,marker,marker))
    report("no-op transfer", () -> update_states_after_mesh_adaption!(states,cv,marker,marker))
    report("mesh clearing", () -> clear_up_topo!(mesh.topo))
end
run_audit()
