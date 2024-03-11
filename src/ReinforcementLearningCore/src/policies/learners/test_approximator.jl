using Test
using Flux

include("/Users/jpslewis/git/ReinforcementLearning.jl/src/ReinforcementLearningCore/src/policies/learners/approximator.jl")

function test_approximator_creation()
    model = Chain(Dense(10, 5, relu), Dense(5, 2))
    optimiser = ADAM()
    approximator = Approximator(model=model, optimiser=optimiser)

    @test typeof(approximator) == Approximator
    @test approximator.model == model
    @test approximator.optimiser_state isa Flux.Optimise.OptimiserState
end

function test_approximator_forward()
    model = Chain(Dense(10, 5, relu), Dense(5, 2))
    optimiser = ADAM()
    approximator = Approximator(model=model, optimiser=optimiser)

    input = rand(10)
    output = approximator(input)

    @test typeof(output) == Array{Float32,1}
    @test length(output) == 2
end

function test_approximator_optimise()
    model = Chain(Dense(10, 5, relu), Dense(5, 2))
    optimiser = ADAM()
    approximator = Approximator(model=model, optimiser=optimiser)

    grad = rand(2)
    approximator.optimise!(grad)

    @test approximator.optimiser_state isa Flux.Optimise.OptimiserState
end

@testset "Approximator Tests" begin
    @testset "Creation" begin
        test_approximator_creation()
    end

    @testset "Forward" begin
        test_approximator_forward()
    end

    @testset "Optimise" begin
        test_approximator_optimise()
    end
end
