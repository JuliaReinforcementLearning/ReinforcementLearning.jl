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

        @assert old_params != new_params
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
end
