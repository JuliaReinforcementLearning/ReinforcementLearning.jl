@testset "NeuralNetworkQ" begin
    q = NeuralNetworkQ(
        Dense(2,2,initW=ones, initb=zeros),
        Flux.Optimise.Descent(0.1)
    )
    x = [1, 2]
    ŷ = q(x)

    @test ŷ ≈ [3, 3]

    y = [1, 0]
    loss = Flux.crossentropy(ŷ, y)
    update!(q, loss)

    @test q(x) == [3.2, 3.0]
end