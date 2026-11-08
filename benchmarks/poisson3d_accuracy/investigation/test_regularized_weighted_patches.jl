using Test

isdefined(@__MODULE__, :poisson_estimator_data) ||
    include(joinpath(@__DIR__, "..", "load.jl"))
isdefined(@__MODULE__, :irregular_tetrahedral_mesh) || include("irregular_tetrahedra.jl")
isdefined(@__MODULE__, :locally_refined_tetrahedral_mesh) || include("local_tetra_refinement.jl")
isdefined(@__MODULE__, :selected_weighted_patch_system) || include("selected_weighted_patches.jl")
isdefined(@__MODULE__, :regularized_weighted_patch_system) || include("regularized_weighted_patches.jl")

function regularized_test_interior_node(fan)
    candidates = [node for node in fan.vertex_count+1:fan.vertex_count+fan.face_count
        if all(vertex -> !fan.boundary_nodes[vertex],
            unique(vec(fan.tet_nodes[:, fan.stars[node]])))]
    isempty(candidates) && error("No wholly interior patch")
    return candidates[argmin([sum(abs2, fan.points[:,node].-0.5) for node in candidates])]
end

"""Independent pointwise quadrature for the weighted-test H1 Gram matrix."""
function regularized_test_gram_quadrature(fan, system; order=6)
    H = zeros(length(system.edges), length(system.edges))
    rule = tetra_rule(order)
    for tet in system.tets
        ids, G = fan.tet_nodes[:,tet], fan.gradients[tet]
        hat = findfirst(==(system.node),ids)
        rows = [get(system.edge_rows,minmax(ids[a],ids[b]),0) for (a,b) in POISSON_P2_EDGES]
        for q in eachindex(rule.weights)
            N = rule.barycentric[:,q]
            weighted_gradients = [4 .* (N[hat] .* (N[a].*G[:,b] + N[b].*G[:,a]) +
                N[a]*N[b].*G[:,hat]) for (a,b) in POISSON_P2_EDGES]
            for test in 1:6, trial in 1:6
                row, column = rows[test], rows[trial]
                (row == 0 || column == 0) && continue
                H[row,column] += fan.volumes[tet]*rule.weights[q]*
                    dot(weighted_gradients[test],weighted_gradients[trial])
            end
        end
    end
    return H
end

"""Same physical patch under separate changes of trial and test coordinates."""
function regularized_test_basis_change(system, trial, test)
    return merge(system,(;A=test'*system.A*trial,b=test'*system.b,
        g=trial'*system.g,K=trial'*system.K*trial,H=test'*system.H*test,
        internal=trial'*system.internal))
end

function regularized_test_spatial_scaling(control, factor)
    mesh,fan=control.mesh,control.fan
    scaled_mesh=merge(mesh,(;points=factor.*mesh.points,centers=factor.*mesh.centers,
        volumes=factor^3 .* mesh.volumes))
    scaled_fan=merge(fan,(;points=factor.*fan.points,
        gradients=[G./factor for G in fan.gradients],volumes=factor^3 .* fan.volumes))
    return (;mesh=scaled_mesh,fan=scaled_fan)
end

"""An independent definition of the prescribed energy normalization."""
function regularized_test_normalization(system)
    Lh=Matrix(cholesky(Symmetric(system.H)).L)
    Lk=Matrix(cholesky(Symmetric(system.K)).L)
    B=Lh\system.A/Lk'
    c=Lh\system.b
    return (;Lh,Lk,B,c)
end

function regularized_test_spectral_system()
    A=Matrix(Diagonal([4.0,0.4,1e-4,0.0]))
    H=Matrix(Diagonal([4.0,1.0,4.0,1.0]))
    K=Matrix(Diagonal([1.0,4.0,1.0,1.0]))
    b=[2.0,-1.0,0.3,0.7]
    g=[-0.3,0.7,1.1,0.5]
    return (;A,H,K,b,g,internal=zeros(4))
end

regularized_test_spec(normalization,filter,parameter) =
    (;name=:test,normalization,filter,parameter=Float64(parameter))

function regularized_test_stationarity(prepared,state,normalization)
    transformed=normalization==:energy ? prepared.energy : prepared.raw
    y=normalization==:energy ? prepared.Lk'*state.coefficients : state.coefficients
    gradient=transformed.B'*(transformed.B*y-transformed.c)+state.chosen_alpha^2 .* y
    scale=max(norm(transformed.B)*norm(transformed.c),eps(Float64))
    @test norm(gradient) <= 2e-9*scale
end

function regularized_test_cap(prepared,G)
    state=regularized_weighted_solution(prepared,regularized_test_spec(:energy,:gain_cap,G))
    B,c=prepared.energy.B,prepared.energy.c
    y=prepared.Lk'*state.coefficients
    @test norm(y) <= G*norm(c)*(1+2e-10)+1e-14
    @test state.actual_gain ≈ norm(y)/norm(c)
    if state.gain_cap_active
        @test state.chosen_alpha > 0
        @test norm(y) ≈ G*norm(c) rtol=2e-9
        # A slightly smaller alpha must violate the cap: test minimality
        # independently, without reusing the implementation's root search.
        S=svd(B)
        alpha=state.chosen_alpha*(1-1e-5)
        smaller=S.V*((S.S./(S.S.^2 .+ alpha^2)).*(S.U'*c))
        @test norm(smaller) > G*norm(c)
        regularized_test_stationarity(prepared,state,:energy)
    else
        @test state.chosen_alpha == 0
    end
    @test state.readout ≈ dot(prepared.system.g,state.coefficients)
    @test abs(state.readout) <= state.readout_gain_bound*(1+2e-10)+1e-14
    return state
end

@testset "Regularized weighted patch equations" begin
    BLAS.set_num_threads(1)
    @testset "Exact spectral systems and unprojected readout" begin
        system=regularized_test_spectral_system()
        prepared=regularized_weighted_prepare(system)
        expected=regularized_test_normalization(system)
        @test prepared.Lh*prepared.Lh' ≈ system.H
        @test prepared.Lk*prepared.Lk' ≈ system.K
        @test prepared.energy.B ≈ expected.B
        @test prepared.energy.c ≈ expected.c
        @test prepared.raw.B == system.A
        @test prepared.raw.c == system.b
        for normalization in (:raw,:energy),filter in (:tsvd,:tikhonov)
            spec=regularized_test_spec(normalization,filter,0.01)
            state=regularized_weighted_solution(prepared,spec)
            B,c=normalization==:energy ? (expected.B,expected.c) : (system.A,system.b)
            sigma=diag(B)
            alpha=spec.parameter*maximum(sigma)
            weights=filter==:tsvd ? [s>alpha ? inv(s) : 0.0 for s in sigma] :
                sigma./(sigma.^2 .+ alpha^2)
            y=weights.*c
            correction=normalization==:energy ? expected.Lk'\y : y
            @test state.coefficients ≈ correction rtol=2e-13 atol=1e-14
            @test state.spectral_cutoff ≈ alpha
            @test state.chosen_alpha ≈ (filter==:tikhonov ? alpha : 0.0)
            @test state.readout ≈ dot(system.g,correction)
            @test state.residual ≈ system.A*correction-system.b
            @test state.correction_energy ≈ dot(correction,system.K*correction)/2
            filter==:tikhonov && regularized_test_stationarity(prepared,state,normalization)
        end

        # Left and right spaces differ. Projecting g onto the retained LEFT
        # space would incorrectly change the exact raw readout from 10 to 0.
        mixed=(;A=[0.0 1.0;0.0 0.0],H=Matrix{Float64}(I,2,2),
            K=Matrix{Float64}(I,2,2),b=[2.0,3.0],g=[4.0,5.0],internal=zeros(2))
        mixed_prepared=regularized_weighted_prepare(mixed)
        mixed_state=regularized_weighted_solution(mixed_prepared,regularized_test_spec(:energy,:tsvd,1e-3))
        @test mixed_state.coefficients ≈ [0.0,2.0]
        @test mixed_state.readout ≈ 10.0
        @test mixed_state.residual ≈ [0.0,-3.0]
        @test mixed_state.normalized_residual_norm ≈ 3.0

        cap_system=(;A=Matrix(Diagonal([1.0,1e-4,0.0])),H=Matrix{Float64}(I,3,3),
            K=Matrix{Float64}(I,3,3),b=[0.0,1.0,1.0],g=[0.3,0.4,0.5],internal=zeros(3))
        cap_prepared=regularized_weighted_prepare(cap_system)
        for G in (10.0,100.0)
            cap_state=regularized_test_cap(cap_prepared,G)
            @test cap_state.gain_cap_active
            for amplitude in (-3.7,1e-20,1e20)
                amplified=merge(cap_system,(;b=amplitude.*cap_system.b,g=amplitude.*cap_system.g))
                amplified_state=regularized_weighted_solution(regularized_weighted_prepare(amplified),
                    regularized_test_spec(:energy,:gain_cap,G))
                @test amplified_state.chosen_alpha ≈ cap_state.chosen_alpha rtol=2e-10
                @test amplified_state.coefficients ≈ amplitude.*cap_state.coefficients rtol=2e-10
                @test amplified_state.actual_gain ≈ cap_state.actual_gain rtol=2e-10
            end
        end
        inactive=merge(cap_system,(;b=[1.0,0.0,0.0]))
        @test !regularized_test_cap(regularized_weighted_prepare(inactive),10.0).gain_cap_active

        # Numerical null modes must remain null under a positive Tikhonov
        # parameter. Otherwise sigma≈eps/alpha² can turn roundoff into an
        # order-one correction. The genuinely retained weak mode stays active.
        floored=(;A=Matrix(Diagonal([1.0,1e-7,1e-16])),H=Matrix{Float64}(I,3,3),
            K=Matrix{Float64}(I,3,3),b=[0.0,1.0,1.0],g=[0.0,0.0,1.0],internal=zeros(3))
        floor_prepared=regularized_weighted_prepare(floored;numerical_rank_tolerance=1e-12)
        for normalization in (:raw,:energy)
            state=regularized_weighted_solution(floor_prepared,
                regularized_test_spec(normalization,:tikhonov,1e-8))
            @test state.coefficients[3] == 0.0
            @test state.coefficients[2] ≈ 1e-7/(1e-14+1e-16) rtol=2e-13
            @test state.readout == 0.0
            regularized_test_stationarity(floor_prepared,state,normalization)
        end
        floor_cap=regularized_test_cap(floor_prepared,1e6)
        @test floor_cap.gain_cap_active
        @test floor_cap.coefficients[3] == 0.0
    end

    @testset "Continuous filter around the hard threshold" begin
        tau=0.01
        states=Dict{Tuple{Symbol,Int},Any}()
        for (side,ratio) in ((-1,1-1e-5),(1,1+1e-5))
            system=(;A=Matrix(Diagonal([1.0,tau*ratio])),H=Matrix{Float64}(I,2,2),
                K=Matrix{Float64}(I,2,2),b=[0.0,tau],g=[0.0,1.0],internal=zeros(2))
            prepared=regularized_weighted_prepare(system)
            for filter in (:tsvd,:tikhonov)
                states[(filter,side)]=regularized_weighted_solution(prepared,
                    regularized_test_spec(:energy,filter,tau))
            end
        end
        tsvd_jump=abs(states[(:tsvd,1)].readout-states[(:tsvd,-1)].readout)
        tikh_change=abs(states[(:tikhonov,1)].readout-states[(:tikhonov,-1)].readout)
        @test tsvd_jump > 0.99
        @test tikh_change < 1e-8
        println("REGULARIZED_THRESHOLD ",(;tsvd_jump,tikh_change,
            tsvd_gains=[states[(:tsvd,side)].actual_gain for side in (-1,1)],
            tikhonov_gains=[states[(:tikhonov,side)].actual_gain for side in (-1,1)]))
    end

    @testset "Actual tetrahedral patch Grams and basis invariance" begin
        problem=manufactured_problem(:cube_smooth)
        regular=irregular_tetrahedral_mesh(:cube,3;amplitude=0.0)
        perturbed=irregular_tetrahedral_mesh(:cube,3;amplitude=.08)
        refined=locally_refined_tetrahedral_mesh(perturbed;stride=7)
        for (label,control) in ((:regular,regular),(:perturbed,perturbed),(:refined,refined))
            @testset "$label" begin
                mesh,fan=control.mesh,control.fan
                primal=solve_tetrahedral_p1(fan,problem)
                data=tetrahedral_p1_estimator_data(mesh,fan,primal,problem)
                interior=regularized_test_interior_node(fan)
                boundary=findfirst(fan.boundary_nodes)
                for node in (interior,boundary)
                    system=regularized_weighted_patch_system(fan,data,node)
                    directH=regularized_test_gram_quadrature(fan,system)
                    @test system.H ≈ directH rtol=2e-11 atol=1e-14
                    @test system.H ≈ system.H'
                    @test isposdef(Symmetric(system.H))
                    prepared=regularized_weighted_prepare(system)
                    expected=regularized_test_normalization(system)
                    @test prepared.energy.B ≈ expected.B rtol=2e-11 atol=1e-13
                    @test prepared.energy.c ≈ expected.c rtol=2e-11 atol=1e-13
                    @test opnorm(prepared.energy.B) <= 1+2e-10
                    previous=selected_weighted_representatives(system;rank_tolerance=1e-11)
                    raw=regularized_weighted_solution(prepared,regularized_test_spec(:raw,:tsvd,1e-11))
                    @test raw.coefficients ≈ previous.minimum_coefficient.coefficients rtol=2e-10 atol=2e-13
                    @test raw.readout ≈ previous.minimum_coefficient.readout rtol=2e-10 atol=2e-15

                    m=length(system.b)
                    trial=Matrix(Diagonal(range(.6,1.7;length=m)))+
                        [.02sin(1.3i+.7j) for i in 1:m,j in 1:m]
                    test=Matrix(Diagonal(range(1.5,.7;length=m)))+
                        [.025cos(.8i+1.1j) for i in 1:m,j in 1:m]
                    changed=regularized_weighted_prepare(regularized_test_basis_change(system,trial,test))
                    @test svdvals(prepared.energy.B) ≈ svdvals(changed.energy.B) rtol=2e-8 atol=2e-13
                    for spec in (regularized_test_spec(:energy,:tsvd,1e-3),
                            regularized_test_spec(:energy,:tikhonov,1e-3),
                            regularized_test_spec(:energy,:gain_cap,10.0))
                        state=regularized_weighted_solution(prepared,spec)
                        other=regularized_weighted_solution(changed,spec)
                        @test state.coefficients ≈ trial*other.coefficients rtol=2e-8 atol=2e-11
                        @test state.readout ≈ other.readout rtol=2e-8 atol=2e-13
                        @test state.correction_energy ≈ other.correction_energy rtol=2e-8 atol=2e-13
                        @test state.chosen_alpha ≈ other.chosen_alpha rtol=2e-8 atol=2e-13
                        @test state.readout ≈ dot(system.g,state.coefficients)
                        @test state.normalized_residual_norm ≈ norm(prepared.Lh\state.residual)
                        @test state.correction_energy_norm ≈ norm(prepared.Lk'*state.coefficients)
                        spec.filter==:tikhonov && regularized_test_stationarity(prepared,state,:energy)
                    end
                    regularized_test_cap(prepared,10.0)
                end

                if label==:perturbed
                    original=regularized_weighted_patch_system(fan,data,interior)
                    prepared=regularized_weighted_prepare(original)
                    # This is a genuine coordinate dilation and a newly
                    # assembled primal solve, rather than scaling A by hand.
                    dilation=3.7
                    scaled=regularized_test_spatial_scaling(control,dilation)
                    scaled_problem=(;rhs=x->problem.rhs(x./dilation)/dilation^2)
                    scaled_primal=solve_tetrahedral_p1(scaled.fan,scaled_problem)
                    scaled_data=tetrahedral_p1_estimator_data(scaled.mesh,scaled.fan,scaled_primal,scaled_problem)
                    scaled_system=regularized_weighted_patch_system(scaled.fan,scaled_data,interior)
                    scaled_prepared=regularized_weighted_prepare(scaled_system)
                    @test scaled_primal.u ≈ primal.u rtol=2e-12 atol=2e-14
                    @test scaled_system.A ≈ dilation.*original.A rtol=2e-12 atol=2e-14
                    @test scaled_system.b ≈ dilation.*original.b rtol=2e-11 atol=2e-14
                    @test scaled_prepared.energy.B ≈ prepared.energy.B rtol=2e-11 atol=2e-13

                    amplitude=-2.5
                    amplitude_problem=(;rhs=x->amplitude*problem.rhs(x))
                    amplitude_primal=solve_tetrahedral_p1(fan,amplitude_problem)
                    amplitude_data=tetrahedral_p1_estimator_data(mesh,fan,amplitude_primal,amplitude_problem)
                    amplitude_prepared=regularized_weighted_prepare(
                        regularized_weighted_patch_system(fan,amplitude_data,interior))
                    for spec in (regularized_test_spec(:energy,:tsvd,1e-3),
                            regularized_test_spec(:energy,:tikhonov,1e-3),
                            regularized_test_spec(:energy,:gain_cap,10.0),
                            regularized_test_spec(:raw,:tsvd,1e-3),
                            regularized_test_spec(:raw,:tikhonov,1e-3))
                        state=regularized_weighted_solution(prepared,spec)
                        space_state=regularized_weighted_solution(scaled_prepared,spec)
                        source_state=regularized_weighted_solution(amplitude_prepared,spec)
                        @test space_state.coefficients ≈ state.coefficients rtol=3e-8 atol=2e-11
                        @test space_state.readout ≈ dilation*state.readout rtol=3e-8 atol=2e-13
                        @test source_state.coefficients ≈ amplitude.*state.coefficients rtol=3e-8 atol=2e-11
                        @test source_state.readout ≈ amplitude^2*state.readout rtol=3e-8 atol=2e-13
                        @test source_state.chosen_alpha ≈ state.chosen_alpha rtol=3e-8 atol=2e-13
                        if spec.normalization==:energy
                            @test space_state.chosen_alpha ≈ state.chosen_alpha rtol=3e-8 atol=2e-13
                            @test space_state.actual_gain ≈ state.actual_gain rtol=3e-8
                        end
                    end

                    dual_problem=(;rhs=x->1.0)
                    dual=solve_tetrahedral_p1(fan,dual_problem)
                    dual_data=tetrahedral_p1_estimator_data(mesh,fan,dual,dual_problem)
                    mixed=regularized_weighted_patch_system(fan,dual_data,interior;readout_data=data)
                    @test norm(mixed.b-mixed.g)>1e-10
                    @test mixed.g ≈ original.g
                    mixed_prepared=regularized_weighted_prepare(mixed)
                    mixed_state=regularized_weighted_solution(mixed_prepared,regularized_test_spec(:energy,:tikhonov,1e-3))
                    @test mixed_state.readout ≈ dot(original.g,mixed_state.coefficients)
                    regularized_test_stationarity(mixed_prepared,mixed_state,:energy)
                end
            end
        end
    end
end
