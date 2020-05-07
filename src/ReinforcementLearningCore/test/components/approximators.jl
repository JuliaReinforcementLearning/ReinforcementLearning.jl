@testset "Approximators" begin

    @testset "TabularApproximator" begin
        A = TabularApproximator(ones(3))

        @test A(1) == 1.0
        @test A(2) == 1.0

        update!(A, 2 => 3.0)
        @test A(2) == 4.0
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
end
