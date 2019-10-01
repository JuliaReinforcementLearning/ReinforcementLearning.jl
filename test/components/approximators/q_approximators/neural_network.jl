@testset "NeuralNetworkQ" begin
    q = NeuralNetworkQ(;model=Dense(2, 2, initW = ones, initb = zeros), optimizer = Flux.Optimise.Descent(0.1))
    x = [1, 2]
    y = [1, 0]
    gs = Flux.gradient(q.params) do
        Flux.crossentropy(RL.batch_estimate(q, x), y)
    end
    update!(q, gs)

    @test q(x) == [3.2, 3.0]
end