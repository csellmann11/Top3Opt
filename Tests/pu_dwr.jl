# julia --project=. --startup-file=no Tests/pu_dwr.jl
isdefined(@__MODULE__, :run_optimization) || include("regularization_benchmark.jl")
using Random

function pu_fixture(n=2;hanging=false,skew=false)
    mesh = create_rectangular_mesh(n,n,n,1.,1.,1.,StandardEl{1})
    if hanging
        Ju3VEM.VEMGeo._refine!(first(RootIterator{4}(mesh.topo)),mesh.topo)
        mesh = Mesh(mesh.topo,StandardEl{1}())
    end
    if skew
        A = SA[1.0 0.2 0.1;0.0 0.8 0.15;0.0 0.0 0.7]
        for i in eachindex(mesh.topo.nodes)
            node = mesh.topo.nodes[i]
            mesh.topo.nodes[i] = Node(node.id,A*node.coords)
        end
        mesh = Mesh(mesh.topo,StandardEl{1}())
    end
    cv = CellValues{3}(mesh)
    states = DesignVarInfo{3}(cv,0.4)
    lambda,mu = E_ν_to_lame(210.e3,0.3)
    pars = SimPars(Helmholtz{3,3}(Ψlin_totopt,(lambda,mu,1.)),lambda,mu,1e-3,15.,1.,0.4)
    return cv,states,pars
end

function pu_interpolate(cv,field)
    u = zeros(3length(cv.dh.dof_mapping))
    for (id,dofs) in cv.dh.dof_mapping
        u[dofs] = field(cv.mesh.nodes[id].coords)
    end
    return u
end

function pu_clamped_problem(n=2;hanging=false)
    cv,states,pars = pu_fixture(n;hanging)
    ch = ConstraintHandler{3}(cv.mesh)
    loads = PU3DBoundaryData()
    add_face_set!(cv.mesh,"clamp",x -> x[1] ≈ 0)
    add_face_set!(cv.mesh,"traction",x -> x[1] ≈ 1)
    pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"clamp",x -> SA[0.,0.,0.])
    pu3d_add_neumann!(loads,ch,cv.dh,cv.facedata_col,"traction",x -> SA[0.,0.,-1.])
    u,K,ed = compute_displacement(cv,ch,states,x -> SA[0.,0.,0.],pars)
    cache = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads)
    return cv,states,pars,ch,loads,u,K,ed,cache
end

function pu_reconstruction_tests()
    @testset "Continuous mean-preserving affine reconstruction" begin
        for hanging in (false,true), skew in (false,true)
            cv,states,pars = pu_fixture(2;hanging,skew)
            _,_,ed = assembly(cv,states,x -> SA[0.,0.,0.],pars)
            ch = ConstraintHandler{3}(cv.mesh)
            c = build_pu_dwr_cache(cv,states,ed,ch,pars)
            G = SA[0.2 -0.3 0.1;0.4 0.2 -0.2;-0.1 0.3 0.2]
            field(x) = G*x+SA[0.1,-0.2,0.3]
            u = pu_interpolate(cv,field)
            pu3d_reconstruct!(c,u,ed)
            @test maximum(norm(c.values[i]-field(x)) for (i,x) in enumerate(c.geometry.node_points)) < 2e-11
            @test all(isapprox(pu3d_projected_gradient(c,e,ed[id]),G;atol=2e-11)
                for (e,id) in enumerate(c.geometry.element_ids))
            # Non-affine data: independent integrals of the reconstructed field
            # must match every volume projection's constant moment.
            u = pu_interpolate(cv,x -> SA[x[1]^2+x[2]*x[3],sin(x[2]),x[3]^2])
            pu3d_reconstruct!(c,u,ed)
            nv = length(c.geometry.vertex_ids)
            for (face,ids) in enumerate(c.geometry.face_nodes)
                fd = cv.facedata_col[c.geometry.face_ids[face]]
                center = c.geometry.node_points[nv+face]
                integral = zero(SVector{3,Float64})
                for i in eachindex(ids)
                    j = mod1(i+1,length(ids))
                    a,b = c.geometry.node_points[ids[i]],c.geometry.node_points[ids[j]]
                    area = norm(cross(a-center,b-center))/2
                    integral += area*(c.values[ids[i]]+c.values[ids[j]]+c.values[nv+face])/3
                end
                expected = Ju3VEM.VEMGeo.get_area(fd)*sum(fd.ΠsL2[1,j]*c.values[node]
                    for (j,node) in enumerate(ids))
                @test integral ≈ expected atol=2e-12
            end
            for (e,id) in enumerate(c.geometry.element_ids)
                integral = zero(SVector{3,Float64})
                for ti in c.geometry.cell_offsets[e]:c.geometry.cell_offsets[e+1]-1
                    t = c.geometry.tets[ti]
                    integral += t.volume*sum(c.values[node] for node in t.nodes)/4
                end
                expected = ed[id].volume*sum(ed[id].proj_s[1,j]*c.values[node]
                    for (j,node) in enumerate(c.geometry.cell_nodes[e]))
                @test integral ≈ expected atol=2e-12
            end
        end
    end
end

function pu_equilibrium_tests()
    @testset "Affine traction patch has zero PU-DWR error" begin
        for hanging in (false,true)
            cv,states,pars = pu_fixture(2;hanging)
            ch,loads = ConstraintHandler{3}(cv.mesh),PU3DBoundaryData()
            G = SA[0.001 0.0 0.0;0.0002 0.0 0.0;-0.0003 0.0 0.0]
            strain = (G+G')/2
            stress = states.χ_vec[1]^3*(pars.λ*tr(strain)*one(strain)+2pars.μ*strain)
            add_face_set!(cv.mesh,"clamp",x -> x[1] ≈ 0)
            pu3d_add_dirichlet!(loads,ch,cv.dh,cv.facedata_col,"clamp",x -> SA[0.,0.,0.])
            for (axis,side) in ((1,1.),(2,0.),(2,1.),(3,0.),(3,1.))
                name = "load_$(axis)_$(side)"
                add_face_set!(cv.mesh,name,x -> x[axis] ≈ side)
                normal = SVector{3,Float64}(ntuple(k -> k == axis ? 2side-1 : 0.,3))
                traction = stress*normal
                pu3d_add_neumann!(loads,ch,cv.dh,cv.facedata_col,name,x -> traction)
            end
            u,K,ed = compute_displacement(cv,ch,states,x -> SA[0.,0.,0.],pars)
            exact = pu_interpolate(cv,x -> G*x)
            @test u ≈ exact rtol=2e-9 atol=2e-12
            c = build_pu_dwr_cache(cv,states,ed,ch,pars;boundary_data=loads)
            result = estimate_pu_dwr!(c,u,states,ed,pars)
            @test result.magnitude < 2e-9*dot(u,K*u)
            @test abs(result.signed) <= result.magnitude+eps()
        end
    end
end

function pu_localization_tests()
    @testset "PU stars, localization and reusable evaluation" begin
        for hanging in (false,true)
            cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(2;hanging)
            firstresult = estimate_pu_dwr!(c,u,states,ed,pars)
            reference = copy(firstresult.indicators)
            signed,magnitude = firstresult.signed,firstresult.magnitude
            @test all(x -> isfinite(x) && x >= 0,reference)
            @test magnitude > 0
            @test sum(reference) ≈ magnitude rtol=5e-14
            @test abs(signed) <= magnitude*(1+1e-14)
            @test estimate_pu_dwr!(c,u,states,ed,pars).indicators ≈ reference rtol=1e-14
            # Changing the redistribution exponent cannot change the signed
            # estimator. It must preserve total magnitude for unequal densities.
            density = [0.2+0.5i/length(states.χ_vec) for i in eachindex(states.χ_vec)]
            result0 = estimate_pu_dwr!(c,u,states,ed,pars;density,q=0.)
            zero_signed,zero_magnitude = result0.signed,result0.magnitude
            zero_marks = copy(result0.indicators)
            result3 = estimate_pu_dwr!(c,u,states,ed,pars;density,q=3.)
            @test result3.signed ≈ zero_signed rtol=1e-13
            @test result3.magnitude ≈ zero_magnitude rtol=1e-13
            @test sum(result3.indicators) ≈ zero_magnitude rtol=2e-14
            @test !isapprox(result3.indicators,zero_marks;rtol=1e-5)
            @test_throws ArgumentError estimate_pu_dwr!(c,u,states,ed,pars;q=-1.)
            # Check each artificial-boundary trace and all shared P2 values
            # using the optional observer, independent of the numbering arrays.
            samples = Dict{Tuple{Int,Tuple{Int,Int}},SVector{3,Float64}}()
            calls = Ref(0)
            observer = function(node,ti,hat,delta)
                t = c.geometry.tets[ti]
                physical = c.geometry.face_cells[t.face][2] == 0
                for i in 1:10
                    val = SVector{3}(delta[3i-2],delta[3i-1],delta[3i])
                    key = (node,pu3d_local_key(t,i))
                    haskey(samples,key) && @test val ≈ samples[key] atol=1e-13
                    samples[key] = val
                    if !(hat == 4 && physical)
                        on_outer = i <= 4 ? i != hat : all(!=(hat),PU3D_P2_EDGES[i-4])
                        on_outer && @test iszero(val)
                    end
                end
                calls[] += 1
            end
            observed = estimate_pu_dwr!(c,u,states,ed,pars;observer)
            @test observed.indicators ≈ reference rtol=1e-13
            @test calls[] == 4length(c.geometry.tets)
            # Artificially isolate node contributions to verify the exact
            # two-cell face-center and single-cell center redistribution.
            fill!(c.nodal,0.); fill!(c.element_correction,0.)
            nv,nf = length(c.geometry.vertex_ids),length(c.geometry.face_ids)
            face = findfirst(x -> x[2] != 0,c.geometry.face_cells)
            left,right = c.geometry.face_cells[face]
            c.nodal[nv+face] = -2.5
            c.marking_weight .= states.area_vec .* density
            marks = pu3d_collect!(c,states).indicators
            @test count(!iszero,marks) == 2
            @test marks[left]/marks[right] ≈ c.marking_weight[left]/c.marking_weight[right]
            @test sum(marks) ≈ 2.5
            fill!(c.marking_weight,0.)
            marks = pu3d_collect!(c,states).indicators
            @test marks[left]/marks[right] ≈ states.area_vec[left]/states.area_vec[right]
            fill!(c.nodal,0.)
            c.nodal[nv+nf+1] = -1.25
            marks = pu3d_collect!(c,states).indicators
            @test marks[1] == 1.25
            @test count(!iszero,marks) == 1
        end
    end
end

function pu_density_contrast_tests()
    @testset "Physical SIMP density and high contrast" begin
        cv,states,pars,ch,loads,u,K,ed,c = pu_clamped_problem(2)
        firstresult = estimate_pu_dwr!(c,u,states,ed,pars)
        original = firstresult.signed
        magnitude = firstresult.magnitude
        states.χ_vec .*= 0.5
        u,K,ed = compute_displacement(cv,ch,states,ZeroBodyForce(),pars)
        scaled = estimate_pu_dwr!(c,u,states,ed,pars)
        @test scaled.signed ≈ 8original rtol=2e-9
        @test scaled.magnitude ≈ 8magnitude rtol=2e-9
        states.χ_vec .= [isodd(i) ? 1.0 : pars.χmin for i in eachindex(states.χ_vec)]
        u,K,ed = compute_displacement(cv,ch,states,ZeroBodyForce(),pars)
        contrasted = estimate_pu_dwr!(c,u,states,ed,pars)
        @test all(isfinite,contrasted.indicators)
        @test contrasted.magnitude > 0
        @test sum(contrasted.indicators) ≈ contrasted.magnitude rtol=2e-14
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    pu_reconstruction_tests()
    pu_equilibrium_tests()
    pu_localization_tests()
    pu_density_contrast_tests()
end
