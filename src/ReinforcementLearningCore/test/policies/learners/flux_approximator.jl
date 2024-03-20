using Test
using Flux

@testset "FluxApproximator Tests" begin
    @testset "Creation, with use_gpu = true toggle" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        optimiser = Adam()
        approximator = FluxApproximator(model=model, optimiser=optimiser, use_gpu=true)

        @test approximator isa FluxApproximator
        @test typeof(approximator.model) == typeof(gpu(model))
        @test approximator.optimiser_state isa NamedTuple
    end

    @testset "Forward" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        optimiser = Adam()
        approximator = FluxApproximator(model=model, optimiser=optimiser, use_gpu=false)

        input = rand(Float32, 10)
        output = RLCore.forward(approximator, input)

        @test typeof(output) == Array{Float32,1}
        @test length(output) == 2
    end

    @testset "Forward to environment" begin
        model = Chain(Dense(4, 5, relu), Dense(5, 2))
        optimiser = Adam()
        approximator = FluxApproximator(model=model, optimiser=optimiser, use_gpu=false)

        env = CartPoleEnv(T=Float32)
        output = RLCore.forward(approximator, env)
        @test typeof(output) == Array{Float32,1}
        @test length(output) == 2
    end

    @testset "Optimise" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        optimiser = Adam()
        approximator = FluxApproximator(model=model, optimiser=optimiser)

        input = rand(Float32, 10)
        

        grad = Flux.Zygote.gradient(approximator) do model
            sum(RLCore.forward(model, input))
        end
    
        @test approximator.model.layers[2].bias == [0, 0]
        RLCore.optimise!(approximator, grad[1])

        @test approximator.model.layers[2].bias != [0, 0]
    end
end
