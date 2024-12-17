using Test, Flux, ChainRulesCore, LinearAlgebra, Distributions, ReinforcementLearningCore
using Flux: params, gradient, unsqueeze, InvDecay, gpu, cpu
import ReinforcementLearningBase: RLBase

@testset "Approximators" begin
    #= These may need to be updated due to recent changes
    @testset "TabularApproximator" begin
        A = TabularVApproximator(; n_state = 2)

        @test A(1) == 0.0
        @test A(2) == 0.0

        push!(A, 2 => A(2) - 3.0)
        @test A(2) == 1.5
        push!(A, 2 => A(2) - 6.0)
        @test A(2) == 3.0
    end

    @testset "NeuralNetworkApproximator" begin
        NN = NeuralNetworkApproximator(; model = Dense(2, 3), optimizer = Descent())

        q_values = NN(rand(Float32, 2))
        @test size(q_values) == (3,)

        gs = gradient(NN) do
            sum(NN(rand(Float32, 2, 5)))
        end

        old_params = deepcopy(collect(Flux.trainable(NN).params))
        push!(NN, gs)
        new_params = collect(Flux.trainable(NN).params)

        @test old_params != new_params
    end

    @testset "ActorCritic" begin
        ac_cpu = ActorCritic(
            actor = NeuralNetworkApproximator(model = Dense(3, 2)),
            critic = NeuralNetworkApproximator(model = Dense(3, 1)),
        )

        ac = ac_cpu |> gpu

        # make sure optimizer is not changed
        @test ac_cpu.optimizer === ac.optimizer

        D = ac.actor.model |> gpu |> device
        @test D === device(ac) === device(ac.actor) == device(ac.critic)

        A = send_to_device(D, rand(Float32, 3))
        ac.actor(A)
        ac.critic(A)
    end=#

    @testset "GaussianNetwork" begin
        @testset "On CPU" begin
            gn = GaussianNetwork(Dense(20,15), Dense(15,10), Dense(15,10, softplus))
            state = rand(Float32, 20, 3) #batch of 3 states
            @testset "Correctness of outputs" begin
                m, L = gn(state)
                @test size(m) == size(L) == (10,3)
                a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
                @test size(a) == (10,3)
                @test size(logp) == (1,3)
                @test logp ≈ diagnormlogpdf(m, L, a)
                @test logp ≈ gn(state, a)
                as, logps = gn(Flux.unsqueeze(state,dims = 2), 5) #sample 5 actions
                @test size(as) == (10,5,3)
                @test size(logps) == (1,5,3)
                logps2 = gn(Flux.unsqueeze(state,dims = 2), as)
                @test logps2 ≈ logps
            end
            @testset "Correctness of gradients" begin
                @testset "One action per state" begin
                    @test Flux.trainable(gn).pre == gn.pre
                    @test Flux.trainable(gn).μ == gn.μ
                    @test Flux.trainable(gn).σ == gn.σ
                    action_saver = Matrix[]
                    g = Flux.gradient(gn) do model
                        a, logp = model(state, is_sampling = true, is_return_log_prob = true)
                        ChainRulesCore.ignore_derivatives() do 
                            push!(action_saver, a)
                        end
                        sum(logp)
                    end
                    g2 = Flux.gradient(gn) do model
                        logp = model(state, only(action_saver))
                        sum(logp)
                    end
                    #Check that gradients are identical
                    @test g == g2
                end
                @testset "Multiple actions per state" begin
                    #Same with multiple actions sampled
                    action_saver = []
                    state = unsqueeze(state, dims = 2)
                    g1 = Flux.gradient(gn) do model
                        a, logp = model(state, 3)
                        ChainRulesCore.ignore_derivatives() do 
                            push!(action_saver, a)
                        end
                        sum(logp)
                    end
                    g2 = Flux.gradient(gn) do model
                        logp = model(state, only(action_saver))
                        sum(logp)
                    end
                    @test g1 == g2
                end
            end
        end
        @testset "CUDA" begin
            if (@isdefined CUDA) && CUDA.functional()
                CUDA.allowscalar(false)
                gn = GaussianNetwork(Dense(20,15), Dense(15,10), Dense(15,10, softplus)) |> gpu
                state = rand(Float32, 20,3)  |> gpu #batch of 3 states
                @testset "Forward pass compatibility" begin
                    m, L = gn(state)
                    @test size(m) == size(L) == (10,3)
                    a, logp = gn(CUDA.CURAND.RNG(), state, is_sampling = true, is_return_log_prob = true)
                    @test size(a) == (10,3)
                    @test size(logp) == (1,3)
                    @test logp ≈ diagnormlogpdf(m, L, a)
                    @test logp ≈ gn(state, a)
                    as, logps = gn(CUDA.CURAND.RNG(), Flux.unsqueeze(state,dims = 2), 5) #sample 5 actions
                    @test size(as) == (10,5,3)
                    @test size(logps) == (1,5,3)
                    logps2 = gn(Flux.unsqueeze(state,dims = 2), as)
                    @test logps2 ≈ logps
                end
                @testset "Backward pass compatibility" begin
                    @testset "One action sampling" begin
                        action_saver = CuMatrix[]
                        g = Flux.gradient(gn) do model
                            a, logp = model(CUDA.CURAND.RNG(), state, is_sampling = true, is_return_log_prob = true)
                            ChainRulesCore.ignore_derivatives() do 
                                push!(action_saver, a)
                            end
                            sum(logp)
                        end
                        g2 = Flux.gradient(gn) do model 
                            logp = model(state, only(action_saver))
                            sum(logp)
                        end
                        #Check that gradients are identical
                        for (grad1, grad2) in zip(g,g2)
                            @test grad1 ≈ grad2
                        end
                    end
                    @testset "Multiple actions sampling" begin
                        action_saver = []
                        state = unsqueeze(state, dims = 2)
                        g = Flux.gradient(gn) do 
                            a, logp = gn(CUDA.CURAND.RNG(), state, 3)
                            ChainRulesCore.ignore_derivatives() do 
                                push!(action_saver, a)
                            end
                            sum(logp)
                        end
                        g2 = Flux.gradient(gn) do model
                            logp = model(state, only(action_saver))
                            sum(logp)
                        end
                        for (grad1, grad2) in zip(g,g2)
                            @test grad1 ≈ grad2
                        end
                    end
                end
            end
        end
    end
    @testset "CovGaussianNetwork" begin
        @testset "utility functions" begin
            cholesky_vec = [1:6;]
            cholesky_mat = [RLCore.softplusbeta(1) 0 0; 2 RLCore.softplusbeta(4) 0; 3 5 RLCore.softplusbeta(6)]
            @test RLCore.vec_to_tril(cholesky_vec, 3) ≈ cholesky_mat
            for i in 1:3, j in 1:i
                inds_mat = [1 0 0; 2 4 0; 3 5 6]
                @test RLCore.cholesky_matrix_to_vector_index(i, j, 3) == inds_mat[i,j]
            end
            for x in -10:10
                @test RLCore.softplusbeta(x,1) ≈ softplus(x) ≈ log(exp(x) +1)
            end
            for x in -10:10
                @test RLCore.softplusbeta(x,2) ≈ log(exp(x/2) +1)*2 >= softplus(x)
            end
            for x in -10:10
                @test RLCore.softplusbeta(x,0.5) ≈ log(exp(x/0.5) +1)*0.5 <= softplus(x)
            end
            cholesky_mats = stack([cholesky_mat for _ in 1:5], dims = 3)
            cholesky_vecs = stack([reshape(cholesky_vec, :, 1) for _ in 1:5], dims = 3)
            @test RLCore.vec_to_tril(cholesky_vecs, 3) ≈ cholesky_mats
            for i in 1:3
                @test RLCore.cholesky_columns(cholesky_vecs, i, 5, 3) ≈ reshape(cholesky_mats[:, i, :], 3, 1, :)
            end
        end
        @testset "identity normalizer" begin
            pre = Dense(20,15)
            μ = Dense(15,10)
            Σ = Dense(15,10*11÷2)
            gn = CovGaussianNetwork(pre, μ, Σ)
            @test Flux.trainable(gn).pre == pre
            @test Flux.trainable(gn).μ == μ
            @test Flux.trainable(gn).Σ == Σ

            state = rand(Float32, 20,3) #batch of 3 states
            #Check that it works in 2D
            m, L = gn(state)
            @test size(m) == (10,3)
            @test size(L) == (10, 10,3)
            a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,3)
            @test size(logp) == (1,3)
            logp2d = gn(state,a)
            @test size(logp2d) == (1,3)
            #rest is 3D
            m, L = gn(Flux.unsqueeze(state,dims = 2))
            @test size(m) == (10,1,3)
            @test size(L) == (10, 10,3)
            a, logp = gn(Flux.unsqueeze(state,dims = 2), is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,1,3)
            @test size(logp) == (1,1,3)

            @test logp ≈ mvnormlogpdf(m, L, a)
            @test logp ≈ gn(Flux.unsqueeze(state,dims = 2), a)
            as, logps = gn(Flux.unsqueeze(state,dims = 2), 5) #sample 5 actions
            @test size(as) == (10,5,3)
            @test size(logps) == (1,5,3)
            logps2 = gn(Flux.unsqueeze(state,dims = 2), as)
            @test logps2 ≈ logps
            s = stack(map(l -> l*l', eachslice(L, dims=3)); dims=3)
            mvnormals = map(z -> MvNormal(Array(vec(z[1])), Array(z[2])), zip(eachslice(m, dims = 3), eachslice(s, dims = 3)))
            logp_truth = [logpdf(mvn, a) for (mvn, a) in zip(mvnormals, eachslice(as, dims = 3))]
            @test stack(logp_truth; dims=2) ≈ dropdims(logps,dims = 1) #test against ground truth
            action_saver = []
            g1 = Flux.gradient(gn) do model
                a, logp = model(Flux.unsqueeze(state,dims = 2), is_sampling = true, is_return_log_prob = true)
                ChainRulesCore.ignore_derivatives() do 
                    push!(action_saver, a)
                end
                mean(logp)
            end
            g2 = Flux.gradient(gn) do model
                logp = model(Flux.unsqueeze(state,dims = 2), only(action_saver))
                mean(logp)
            end
            @test g1 == g2

            empty!(action_saver)

            g3 = Flux.gradient(gn) do model
                a, logp = model(Flux.unsqueeze(state,dims = 2), is_sampling = true, is_return_log_prob = true)
                ChainRulesCore.ignore_derivatives() do 
                    push!(action_saver, a)
                end
                mean(logp)
            end
            g4 = Flux.gradient(gn) do model
                logp = model(Flux.unsqueeze(state, dims = 2), only(action_saver))
                mean(logp)
            end

            @test g4 == g3
        end
        @testset "CUDA" begin
            if (@isdefined CUDA) && CUDA.functional()
                CUDA.allowscalar(false) 
                rng = CUDA.CURAND.RNG()
                pre = Dense(20,15) |> gpu
                μ = Dense(15,10) |> gpu
                Σ = Dense(15,10*11÷2) |> gpu
                gn = CovGaussianNetwork(pre, μ, Σ)
                state = rand(Float32, 20,3)|> gpu #batch of 3 states
                m, L = gn(Flux.unsqueeze(state,dims = 2))
                @test size(m) == (10,1,3)
                @test size(L) == (10, 10,3)
                a, logp = gn(rng, Flux.unsqueeze(state,dims = 2), is_sampling = true, is_return_log_prob = true)
                @test size(a) == (10,1,3)
                @test size(logp) == (1,1,3)

                @test logp ≈ mvnormlogpdf(m, L, a)
                @test logp ≈ gn(Flux.unsqueeze(state,dims = 2), a)
                as, logps = gn(rng,Flux.unsqueeze(state,dims = 2), 5) #sample 5 actions
                @test size(as) == (10,5,3)
                @test size(logps) == (1,5,3)
                logps2 = gn(Flux.unsqueeze(state,dims = 2), as)
                @test logps2 ≈ logps
                s = stack(map(l -> l*l', eachslice(L, dims=3)); dims=3)
                mvnormals = map(z -> MvNormal(Array(vec(z[1])), Array(z[2])), zip(eachslice(m, dims = 3), eachslice(s, dims = 3)))
                logp_truth = [logpdf(mvn, cpu(a)) for (mvn, a) in zip(mvnormals, eachslice(as, dims = 3))]
                @test reduce(hcat, collect(logp_truth)) ≈ dropdims(cpu(logps); dims=1) #test against ground truth
                action_saver = []
                g = Flux.gradient(gn) do model
                    a, logp = model(rng, Flux.unsqueeze(state,dims = 2), is_sampling = true, is_return_log_prob = true)
                    ChainRulesCore.ignore_derivatives() do 
                        push!(action_saver, a)
                    end
                    mean(logp)
                end

                g2 = Flux.gradient(gn) do model
                    logp = model(Flux.unsqueeze(state,dims = 2), only(action_saver))
                    mean(logp)
                end
                for (grad1, grad2) in zip(g,g2)
                    @test grad1 ≈ grad2
                end
                empty!(action_saver)
                g3 = Flux.gradient(gn) do model
                    a, logp = model(rng, Flux.unsqueeze(state,dims = 2), 3)
                    ChainRulesCore.ignore_derivatives() do 
                        push!(action_saver, a)
                    end
                    mean(logp)
                end
                g4 = Flux.gradient(gn) do model
                    logp = model(Flux.unsqueeze(state,dims = 2), only(action_saver))
                    mean(logp)
                end
                for (grad1, grad2) in zip(g4,g3)
                    @test grad1 ≈ grad2
                end
            end
        end
    end
    @testset "CategoricalNetwork" begin
        d = CategoricalNetwork(Dense(5,3))
        s = rand(Float32, 5, 10)
        a, logits = d(s, is_sampling = true, is_return_log_prob = true)
        @test size(a) == (3,10) == size(logits)
        a, logits = d(s, 4)
        @test size(a) == (3,4,10) == size(logits)
        
        #3D input
        s = rand(Float32, 5,1,10)
        a, logits = d(s, is_sampling = true, is_return_log_prob = true)
        @test size(a) == (3,1,10) == size(logits)
        @test logits isa Array{Float32, 3}
        a, logits = d(s, 4)        
        @test size(a) == (3,4,10) == size(logits)

        #Masking
        ##2D
        s = rand(Float32, 5, 10)
        mask = trues(3, 10)
        mask[1,:] .= false
        a_masked, logits = d(s, mask, is_sampling = true, is_return_log_prob = true)
        @test size(a_masked) == (3, 10)
        @test all(a -> a == 0, a_masked[1,:])
        @test all(l -> l == -Inf32, logits[1, :]) && all(l -> l !== -Inf32, logits[2:3, :])
        a_masked, logits = d(s, mask, 4)
        @test size(a_masked) == (3,4,10) == size(logits)
        ##3D
        s = rand(Float32, 5,1,10)
        mask = trues(3, 1, 10)
        mask[1,:, :] .= false
        a_masked, logits = d(s, mask, is_sampling = true, is_return_log_prob = true)
        @test size(a_masked) == (3, 1, 10)
        @test all(a -> a == 0, a_masked[1,:, :])
        @test all(l -> l == -Inf32, logits[1, :, :]) && all(l -> l !== -Inf32, logits[2:3, :, :])
        a_masked, logits = d(s, mask, 4)
        @test size(a_masked) == (3,4,10) == size(logits)

        @testset "CUDA" begin
            if false #CUDA.functional() BROKEN due to scalar indexing. Solve in other PR.
                CUDA.allowscalar(false) 
                rng = CUDA.CURAND.RNG()
                d = CategoricalNetwork(Dense(5,3) |> gpu)
                s = cu(rand(Float32, 5, 10))
                a, logits = d(rng, s, is_sampling = true, is_return_log_prob = true);
                @test size(a) == (3,10) == size(logits)
                a, logits = d(rng, s, 4);
                @test size(a) == (3,4,10) == size(logits)
                
                #3D input
                s = cu(rand(Float32, 5,1,10))
                a, logits = d(rng, s, is_sampling = true, is_return_log_prob = true);
                @test size(a) == (3,1,10) == size(logits)
                a, logits = d(rng, s, 4);
                @test size(a) == (3,4,10) == size(logits)
                a_masked, logits = d(rng, s, 4)
                @test size(a_masked) == (3,4,10) == size(logits)
            end
        end
    end
end
