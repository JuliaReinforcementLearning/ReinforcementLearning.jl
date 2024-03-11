using Test
using Flux

@testset "Approximator Tests" begin
    @testset "Creation, with use_gpu = true toggle" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        optimiser = ADAM()
        approximator = Approximator(model=model, optimiser=optimiser, use_gpu=true)
    
        @test typeof(approximator) == Approximator
        @test approximator.model == model
        @test approximator.optimiser_state isa Flux.Optimise.OptimiserState    
    end

    @testset "Forward" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        optimiser = ADAM()
        approximator = Approximator(model=model, optimiser=optimiser, use_gpu=false)
    
        input = rand(10)
        output = forward(approximator, input)
    
        @test typeof(output) == Array{Float32,1}
        @test length(output) == 2    
    end

    @testset "Optimise" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        optimiser = ADAM()
        approximator = Approximator(model=model, optimiser=optimiser)
    
        grad = rand(2)
        optimise!(approximator, grad)
    
        @test approximator.optimiser_state isa Flux.Optimise.OptimiserState
    
    end
end
