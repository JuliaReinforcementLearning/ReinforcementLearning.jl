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
            m, s = gn(state)
            @test size(m) == size(s) == (10,3)
            a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,3)
            @test size(logp) == (1,3)
            @test logp ≈ sum(normlogpdf(m, exp.(s), a) .- (2.0f0 .* (log(2.0f0) .- a .- softplus.(-2.0f0 .* a))), dims = 1)
            @test logp ≈ gn(state, a)
            as, logps = gn(Flux.unsqueeze(state,2), 5) #sample 5 actions
            @test size(as) == (10,5,3)
            @test size(logps) == (1,5,3)
            logps2 = gn(Flux.unsqueeze(state,2), as)
            @test logps2 ≈ logps
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
                sum(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do 
                logp = gn(state, a)
                sum(logp)
            end
        end
        @testset "tanh normalizer" begin
            pre = Dense(20,15)
            μ = Dense(15,10)
            logσ = Dense(15,10)
            gn = GaussianNetwork(pre, μ, logσ)
            @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, logσ.W, logσ.b])
            state = rand(20,3) #batch of 3 states
            m, s = gn(state)
            @test size(m) == size(s) == (10,3)
            a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
            @test size(a) == (10,3)
            @test size(logp) == (1,3)
            @test logp ≈ sum(normlogpdf(m, exp.(s), a) .- (2.0f0 .* (log(2.0f0) .- a .- softplus.(-2.0f0 .* a))), dims = 1)
            @test logp ≈ gn(state, a) #this was broken
            as, logps = gn(Flux.unsqueeze(state,2), 5) #sample 5 actions
            @test size(as) == (10,5,3)
            @test size(logps) == (1,5,3)
            logps2 = gn(Flux.unsqueeze(state,2), as)
            @test logps2 ≈ logps
            g = Flux.gradient(Flux.params(gn)) do 
                a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
                sum(logp)
            end
            g2 = Flux.gradient(Flux.params(gn)) do 
                logp = gn(state, a)
                sum(logp)
            end
        end
        @testset "CUDA" begin
            if CUDA.is_functional()
                pre = Dense(20,15) |> gpu
                μ = Dense(15,10) |> gpu
                logσ = Dense(15,10) |> gpu
                gn = GaussianNetwork(pre, μ, logσ)
                @test Flux.params(gn) == Flux.Params([pre.W, pre.b, μ.W, μ.b, logσ.W, logσ.b])
                state = rand(20,3)  |> gpu #batch of 3 states
                m, s = gn(state)
                @test size(m) == size(s) == (10,3)
                a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
                @test size(a) == (10,3)
                @test size(logp) == (1,3)
                @test logp ≈ sum(normlogpdf(m, exp.(s), a) .- (2.0f0 .* (log(2.0f0) .- a .- softplus.(-2.0f0 .* a))), dims = 1)
                @test logp ≈ gn(state, a) #this was broken
                as, logps = gn(Flux.unsqueeze(state,2), 5) #sample 5 actions
                @test size(as) == (10,5,3)
                @test size(logps) == (1,5,3)
                logps2 = gn(Flux.unsqueeze(state,2), as)
                @test logps2 ≈ logps
                g = Flux.gradient(Flux.params(gn)) do 
                    a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
                    sum(logp)
                end
                g2 = Flux.gradient(Flux.params(gn)) do 
                    logp = gn(state, a)
                    sum(logp)
                end
            end
        end
    end
end
