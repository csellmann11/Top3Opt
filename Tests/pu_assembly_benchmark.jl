# Standalone exact comparison of scalar and block assembly (no mechanics JIT).
using LinearAlgebra, StaticArrays, Test, Random, Statistics
include("../src/refinement_utils/pu_geometry.jl")
include("../src/refinement_utils/pu_tetra.jl")
include("../src/refinement_utils/pu_estimator.jl")
include("pu_assembly_reference.jl")

@inline function pu3d_load_entry(loads,index,row,hat)
    iszero(loads.slots[index]) && return 0.,0.
    return loads.force[row],loads.weighted[row,hat]
end

function assembly_fixture(;hat=1,full=false,loaded=false,randomize=false)
    points = [SA[0.1,-0.3,0.2],SA[1.1,0.1,0.4],SA[0.3,0.7,-0.1],SA[-0.1,0.2,1.3]]
    t = pu3d_tetra(SA[1,2,3,4],points,1,1)
    isfree(i) = full || i == hat || (i > 4 && hat in PU3D_P2_EDGES[i-4])
    count = 0
    dofs = SVector{3,Int}[]
    for i in 1:10
        row = MVector(0,0,0)
        for component in 1:3
            if isfree(i) && (!randomize || rand() > 0.3)
                count += 1
                row[component] = count
            end
        end
        push!(dofs,SVector(row))
    end
    if randomize && count > 0
        permutation = randperm(count)
        dofs = [SVector{3,Int}(ntuple(k -> row[k] == 0 ? 0 : permutation[row[k]],3)) for row in dofs]
    end
    map = SVector{10}(1:10)
    w = (;matrix=zeros(count,count),rhs=zeros(count),weighted=zeros(count),dofs)
    stress = SA[3.1 0.3 -0.2;0.3 -1.1 0.7;-0.2 0.7 0.5]
    loads = (;slots=[Int(loaded)],force=SVector{30}(randn(30)),weighted=SMatrix{30,4}(randn(30,4)))
    c = (;geometry=(;tets=[t]),stress=[stress],scale=[0.17^3],lambda=2.3,mu=5.7,
        energy=[0.83],projected_energy=[0.61],loads)
    return w,c,map
end

function clear_assembly!(w)
    fill!(w.matrix,0.);fill!(w.rhs,0.);fill!(w.weighted,0.)
end

@testset "Exact block assembly with constraints and arbitrary DOF order" begin
    Random.seed!(8431)
    for hat in 1:4,full in (false,true),loaded in (false,true),randomize in (false,true)
        w,c,map = assembly_fixture(;hat,full,loaded,randomize)
        reference = deepcopy(w)
        expected = reference_pu3d_add_tetra!(reference,c,1,hat,map)
        actual = pu3d_add_tetra!(w,c,1,hat,map)
        @test actual == expected
        @test w.matrix ≈ reference.matrix atol=1e-15 rtol=2e-14
        @test w.rhs ≈ reference.rhs atol=1e-14
        @test w.weighted ≈ reference.weighted atol=1e-14
    end
end

function assembly_trial(kernel,w,c,map,repetitions)
    clear_assembly!(w)
    total = 0.
    for _ in 1:repetitions
        total += kernel(w,c,1,1,map)
    end
    return total+w.matrix[end,end]
end

function benchmark_assembly(;repetitions=20000)
    for full in (false,true)
        w,c,map = assembly_fixture(;full)
        for kernel in (reference_pu3d_add_tetra!,pu3d_add_tetra!)
            assembly_trial(kernel,w,c,map,10)
        end
        old = Float64[]; new = Float64[]
        for _ in 1:5
            push!(old,@elapsed assembly_trial(reference_pu3d_add_tetra!,w,c,map,repetitions))
            push!(new,@elapsed assembly_trial(pu3d_add_tetra!,w,c,map,repetitions))
        end
        allocation = @allocated assembly_trial(pu3d_add_tetra!,w,c,map,1)
        println((;full,scalar_ns=1e9median(old)/repetitions,
            block_ns=1e9median(new)/repetitions,speedup=median(old)/median(new),allocation))
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && benchmark_assembly()
