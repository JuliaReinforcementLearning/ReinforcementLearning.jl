@testset "Approximators" begin

    @testset "TabularApproximator" begin
        A = TabularVApproximator(; n_state = 2, opt = InvDecay(1.0))

        @test A(1) == 0.0
        @test A(2) == 0.0

        update!(A, 2 => A(2) - 3.0)
        @test A(2) == 1.5
        update!(A, 2 => A(2) - 6.0)
        @test A(2) == 3.0
    end

    @testset "NeuralNetworkApproximator" begin
        NN = NeuralNetworkApproximator(; model = Dense(2, 3), optimizer = Descent())

        q_values = NN(rand(2))
        @test size(q_values) == (3,)

        gs = gradient(params(NN)) do
            sum(NN(rand(2, 5)))
        end

        old_params = deepcopy(collect(params(NN).params))
        update!(NN, gs)
        new_params = collect(params(NN).params)

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

        A = send_to_device(D, rand(3))
        ac.actor(A)
        ac.critic(A)
    end

    @testset "GaussianNetwork" begin
        @testset "identity normalizer" begin
            pre = Dense(20,15)
            μ = Dense(15,10)
            logσ = Dense(15,10)
            gn = GaussianNetwork(pre, μ, logσ, identity)
            @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, logσ.W, logσ.b])
            state = rand(20,3) #batch of 3 states
            m, L = gn(state)
            @test size(m) == size(L) == (10,3)
            a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,3)
            @test size(logp) == (1,3)
            @test logp ≈ sum(normlogpdf(m, exp.(L), a) .- (2.0f0 .* (log(2.0f0) .- a .- softplus.(-2.0f0 .* a))), dims = 1)
            @test logp ≈ gn(state, a)
            as, logps = gn(Flux.unsqueeze(state,2), 5) #sample 5 actions
            @test size(as) == (10,5,3)
            @test size(logps) == (1,5,3)
            logps2 = gn(Flux.unsqueeze(state,2), as)
            @test logps2 ≈ logps
            action_saver = []
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                sum(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do 
                logp = gn(state, only(action_saver))
                sum(logp)
            end
            #Check that gradients are identical
            for (grad1, grad2) in zip(g,g2)
                @test grad1 ≈ grad2
            end
            #Same with multiple actions sampled
            empty!(action_saver)
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(state, 3)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                sum(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do 
                logp = gn(state, only(action_saver))
                sum(logp)
            end
            for (grad1, grad2) in zip(g,g2)
                @test grad1 ≈ grad2
            end
        end
        @testset "tanh normalizer" begin
            pre = Dense(20,15)
            μ = Dense(15,10)
            logσ = Dense(15,10)
            gn = GaussianNetwork(pre, μ, logσ)
            @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, logσ.W, logσ.b])
            state = rand(20,3) #batch of 3 states
            m, L = gn(state)
            @test size(m) == size(L) == (10,3)
            a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,3)
            @test size(logp) == (1,3)
            @test logp ≈ sum(normlogpdf(m, exp.(L), a) .- (2.0f0 .* (log(2.0f0) .- a .- softplus.(-2.0f0 .* a))), dims = 1)
            @test logp ≈ gn(state, a)
            as, logps = gn(Flux.unsqueeze(state,2), 5) #sample 5 actions
            @test size(as) == (10,5,3)
            @test size(logps) == (1,5,3)
            logps2 = gn(Flux.unsqueeze(state,2), as)
            @test logps2 ≈ logps
            action_saver = []
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                sum(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do 
                logp = gn(state, only(action_saver))
                sum(logp)
            end
            #Check that gradients are identical
            for (grad1, grad2) in zip(g,g2)
                @test grad1 ≈ grad2
            end
            #Same with multiple actions sampled
            empty!(action_saver)
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(state, 3)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                sum(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do 
                logp = gn(state, only(action_saver))
                sum(logp)
            end
            for (grad1, grad2) in zip(g,g2)
                @test grad1 ≈ grad2
            end
        end
        @testset "CUDA" begin
            if CUDA.functional()
                pre = Dense(20,15) |> gpu
                μ = Dense(15,10) |> gpu
                logσ = Dense(15,10) |> gpu
                gn = GaussianNetwork(pre, μ, logσ)
                @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, logσ.W, logσ.b])
                state = rand(20,3)  |> gpu #batch of 3 states
                m, L = gn(state)
                @test size(m) == size(L) == (10,3)
                a, logp = gn(CUDA.CURAND.RNG(), state, is_sampling = true, is_return_log_prob = true)
                @test size(a) == (10,3)
                @test size(logp) == (1,3)
                @test logp ≈ sum(normlogpdf(m, exp.(L), a) .- (2.0f0 .* (log(2.0f0) .- a .- softplus.(-2.0f0 .* a))), dims = 1)
                @test logp ≈ gn(state, a)
                as, logps = gn(CUDA.CURAND.RNG(), Flux.unsqueeze(state,2), 5) #sample 5 actions
                @test size(as) == (10,5,3)
                @test size(logps) == (1,5,3)
                logps2 = gn(Flux.unsqueeze(state,2), as)
                @test logps2 ≈ logps
                action_saver = []
                g = Flux.gradient(Flux.params(gn)) do 
                    a, logp = gn(CUDA.CURAND.RNG(), state, is_sampling = true, is_return_log_prob = true)
                    Flux.Zygote.ignore() do 
                        push!(action_saver, a)
                    end
                    sum(logp)
                end
                g2 = Flux.gradient(Flux.params(gn)) do 
                    logp = gn(state, only(action_saver))
                    sum(logp)
                end
                #Check that gradients are identical
                for (grad1, grad2) in zip(g,g2)
                    @test grad1 ≈ grad2
                end
                #Same with multiple actions sampled
                empty!(action_saver)
                g = Flux.gradient(Flux.params(gn)) do 
                    a, logp = gn(CUDA.CURAND.RNG(), state, 3)
                    Flux.Zygote.ignore() do 
                        push!(action_saver, a)
                    end
                    sum(logp)
                end
                g2 = Flux.gradient(Flux.params(gn)) do 
                    logp = gn(state, only(action_saver))
                    sum(logp)
                end
                for (grad1, grad2) in zip(g,g2)
                    @test grad1 ≈ grad2
                end
            end
        end
    end
    @testset "CovGaussianNetwork" begin
        @testset "identity normalizer" begin
            pre = Dense(20,15)
            μ = Dense(15,10)
            Σ = Dense(15,10*11÷2)
            gn = CovGaussianNetwork(pre, μ, Σ, identity)
            @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, Σ.W, Σ.b])
            state = rand(20,3) #batch of 3 states
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
            m, L = gn(Flux.unsqueeze(state,2))
            @test size(m) == (10,1,3)
            @test size(L) == (10, 10,3)
            a, logp = gn(Flux.unsqueeze(state,2), is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,1,3)
            @test size(logp) == (1,1,3)

            @test logp ≈ mvnormlogpdf(m, L, a)
            @test logp ≈ gn(Flux.unsqueeze(state,2), a)
            as, logps = gn(Flux.unsqueeze(state,2), 5) #sample 5 actions
            @test size(as) == (10,5,3)
            @test size(logps) == (1,5,3)
            logps2 = gn(Flux.unsqueeze(state,2), as)
            @test logps2 ≈ logps
            s = Flux.stack(map(l -> l*l', eachslice(L, dims=3)),3)
            mvnormals = map(z -> MvNormal(Array(vec(z[1])), Array(z[2])), zip(eachslice(m, dims = 3), eachslice(s, dims = 3)))
            logp_truth = [logpdf(mvn, a) for (mvn, a) in zip(mvnormals, eachslice(as, dims = 3))]
            @test Flux.stack(logp_truth,2) ≈ dropdims(logps,dims = 1) #test against ground truth
            action_saver = []
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(Flux.unsqueeze(state,2), is_sampling = true, is_return_log_prob = true)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                mean(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do
                logp = gn(Flux.unsqueeze(state,2), only(action_saver))
                mean(logp)
            end
            for (grad1, grad2) in zip(g,g2)
                @test grad1 ≈ grad2
            end
            empty!(action_saver)
            g3 = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(Flux.unsqueeze(state,2), 3)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                mean(logp)
            end
            g4 = Flux.gradient(Flux.params(gn)) do
                logp = gn(Flux.unsqueeze(state,2), only(action_saver))
                mean(logp)
            end
            for (grad1, grad2) in zip(g4,g3)
                @test grad1 ≈ grad2
            end
        end
        @testset "tanh normalizer" begin
            pre = Dense(20,15)
            μ = Dense(15,10)
            Σ = Dense(15,10*11÷2)
            gn = CovGaussianNetwork(pre, μ, Σ)
            @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, Σ.W, Σ.b])
            state = rand(20,3) #batch of 3 states
            m, L = gn(Flux.unsqueeze(state,2))
            @test size(m) == (10,1,3)
            @test size(L) == (10, 10,3)
            a, logp = gn(Flux.unsqueeze(state,2), is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,1,3)
            @test size(logp) == (1,1,3)

            @test logp ≈ mvnormlogpdf(m, L, a)
            @test logp ≈ gn(Flux.unsqueeze(state,2), a)
            as, logps = gn(Flux.unsqueeze(state,2), 5) #sample 5 actions
            @test size(as) == (10,5,3)
            @test size(logps) == (1,5,3)
            logps2 = gn(Flux.unsqueeze(state,2), as)
            @test logps2 ≈ logps
            s = Flux.stack(map(l -> l*l', eachslice(L, dims=3)),3)
            mvnormals = map(z -> MvNormal(Array(vec(z[1])), Array(z[2])), zip(eachslice(m, dims = 3), eachslice(s, dims = 3)))
            logp_truth = [logpdf(mvn, a) for (mvn, a) in zip(mvnormals, eachslice(as, dims = 3))]
            @test Flux.stack(logp_truth,2) ≈ dropdims(logps,dims = 1) #test against ground truth
            action_saver = []
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(Flux.unsqueeze(state,2), is_sampling = true, is_return_log_prob = true)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                mean(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do
                logp = gn(Flux.unsqueeze(state,2), only(action_saver))
                mean(logp)
            end
            for (grad1, grad2) in zip(g,g2)
                @test grad1 ≈ grad2
            end
            empty!(action_saver)
            g3 = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(Flux.unsqueeze(state,2), 3)
                Flux.Zygote.ignore() do 
                    push!(action_saver, a)
                end
                mean(logp)
            end
            g4 = Flux.gradient(Flux.params(gn)) do
                logp = gn(Flux.unsqueeze(state,2), only(action_saver))
                mean(logp)
            end
            for (grad1, grad2) in zip(g4,g3)
                @test grad1 ≈ grad2
            end
        end
        @testset "CUDA" begin
            if CUDA.functional()
                CUDA.allowscalar(false) 
                rng = CUDA.CURAND.RNG()
                pre = Dense(20,15) |> gpu
                μ = Dense(15,10) |> gpu
                Σ = Dense(15,10*11÷2) |> gpu
                gn = CovGaussianNetwork(pre, μ, Σ, identity)
                @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, Σ.W, Σ.b])
                state = rand(20,3)|> gpu #batch of 3 states
                m, L = gn(Flux.unsqueeze(state,2))
                @test size(m) == (10,1,3)
                @test size(L) == (10, 10,3)
                a, logp = gn(rng, Flux.unsqueeze(state,2), is_sampling = true, is_return_log_prob = true)
                @test size(a) == (10,1,3)
                @test size(logp) == (1,1,3)

                @test logp ≈ mvnormlogpdf(m, L, a)
                @test logp ≈ gn(Flux.unsqueeze(state,2), a)
                as, logps = gn(rng,Flux.unsqueeze(state,2), 5) #sample 5 actions
                @test size(as) == (10,5,3)
                @test size(logps) == (1,5,3)
                logps2 = gn(Flux.unsqueeze(state,2), as)
                @test logps2 ≈ logps
                s = Flux.stack(map(l -> l*l', eachslice(L, dims=3)),3)
                mvnormals = map(z -> MvNormal(Array(vec(z[1])), Array(z[2])), zip(eachslice(m, dims = 3), eachslice(s, dims = 3)))
                logp_truth = [logpdf(mvn, cpu(a)) for (mvn, a) in zip(mvnormals, eachslice(as, dims = 3))]
                @test Flux.stack(logp_truth,2) ≈ dropdims(cpu(logps),dims = 1) #test against ground truth
                action_saver = []
                g = Flux.gradient(Flux.params(gn)) do 
                    a, logp = gn(rng, Flux.unsqueeze(state,2), is_sampling = true, is_return_log_prob = true)
                    Flux.Zygote.ignore() do 
                        push!(action_saver, a)
                    end
                    mean(logp)
                end

                g2 = Flux.gradient(Flux.params(gn)) do
                    logp = gn(Flux.unsqueeze(state,2), only(action_saver))
                    mean(logp)
                end
                for (grad1, grad2) in zip(g,g2)
                    @test grad1 ≈ grad2
                end
                empty!(action_saver)
                g3 = Flux.gradient(Flux.params(gn)) do 
                    a, logp = gn(rng, Flux.unsqueeze(state,2), 3)
                    Flux.Zygote.ignore() do 
                        push!(action_saver, a)
                    end
                    mean(logp)
                end
                g4 = Flux.gradient(Flux.params(gn)) do
                    logp = gn(Flux.unsqueeze(state,2), only(action_saver))
                    mean(logp)
                end
                for (grad1, grad2) in zip(g4,g3)
                    @test grad1 ≈ grad2
                end
                CUDA.allowscalar(true) #to avoid breaking other tests 
            end
        end
    end
end
