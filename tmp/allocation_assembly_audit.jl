using Ju3VEM, Ju3VEM.FixedSizeArrays, StaticArrays, LinearAlgebra, SparseArrays, Statistics
const U=3
include("../src/get_sparsity_pattern.jl")
BLAS.set_num_threads(1)

function nodeincidence(cv)
    node_id_map = create_dense_node_id_map(cv.mesh)
    rows=Int32[]; cols=Int32[]; ids=Set{Int32}(); count=Ref(1)
    for el in RootIterator{4}(cv.mesh.topo)
        push!(cols,count[]); empty!(ids)
        Ju3VEM.VEMGeo.iterate_volume_areas(cv.facedata_col,cv.mesh.topo,el.id) do _,fd,_
            for id in fd.face_node_ids
                id in ids && continue
                push!(ids,id); push!(rows,node_id_map[id]); count[]+=1
            end
        end
        for id in cv.mesh.int_coords_connect[4][el.id]
            push!(rows,node_id_map[id]); count[]+=1
        end
    end
    push!(cols,count[])
    SparseMatrixCSC(length(keys(cv.dh.dof_mapping)),length(cols)-1,cols,rows,ones(Bool,length(rows)))
end

function expand_direct(P::SparseMatrixCSC{T,Ti},u::Int,::Type{To}=Int64) where {T,Ti,To}
    n=size(P,2)*u
    cp=Vector{To}(undef,n+1); rv=Vector{To}(undef,nnz(P)*u*u)
    values=zeros(Float64,length(rv)); cursor=1
    @inbounds for j in axes(P,2), b in 1:u
        cp[(j-1)*u+b]=cursor
        for p in nzrange(P,j), a in 1:u
            rv[cursor]=(P.rowval[p]-1)*u+a; cursor+=1
        end
    end
    cp[end]=cursor
    SparseMatrixCSC(size(P,1)*u,n,cp,rv,values)
end

function oldexpand(P,u)
    B=sparse(ones(Bool,u,u)); S=kron(P,B)
    SparseMatrixCSC(S.m,S.n,S.colptr,S.rowval,zeros(Float64,length(S.nzval)))
end

function oldpattern(cv)
    incidence=nodeincidence(cv)
    oldexpand(incidence*incidence',3)
end

function report(name,f)
    f()
    samples=[@timed f() for _ in 1:5]
    println(name," median_seconds=",median(x.time for x in samples)," alloc_MiB=",median(x.bytes for x in samples)/2.0^20)
    return samples[end].value
end

function audit()
    mesh=create_rectangular_mesh(24,8,8,3.,1.,1.,StandardEl{1})
    cv=CellValues{3}(mesh)
    println("cells=",length(collect(RootIterator{4}(mesh.topo))))
    incidence=report("incidence",()->nodeincidence(cv))
    P=report("incidence product",()->incidence*incidence')
    println("product type=",typeof(P)," nnz=",nnz(P))
    old=report("old kron plus zero values",()->oldexpand(P,3))
    direct=report("production CSC expansion (same Int64 index type)",()->_expand_sparsity_blocks(P,3))
    narrow=report("direct CSC expansion (Int32 index type)",()->expand_direct(P,3,Int32))
    println("old_type=",typeof(old)," direct_type=",typeof(direct)," narrow_type=",typeof(narrow))
    @assert old.colptr == direct.colptr && old.rowval == direct.rowval && old.nzval == direct.nzval
    report("complete original",()->oldpattern(cv))
    report("complete production",()->get_sparsity_pattern(cv))
    println("final bytes=",Base.summarysize(old)," nnz=",nnz(old)," direct exact=true")
end
audit()
