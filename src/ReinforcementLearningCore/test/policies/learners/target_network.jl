using Test
using Flux

@testset "TargetNetwork Tests" begin
    @testset "Creation" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        target_network = TargetNetwork(model=model, use_gpu=true)
    
        @test typeof(target_network) == TargetNetwork
        @test target_network.network == model
        @test target_network.target isa Flux.Chain
        @test target_network.sync_freq == 1
        @test target_network.ρ == 0.0
        @test target_network.n_optimise == 0    
    end

    @testset "Forward" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        target_network = TargetNetwork(model=model)
    
        input = rand(10)
        output = forward(target_network, input)
    
        @test typeof(output) == Array{Float32,1}
        @test length(output) == 2    
    end

    @testset "Optimise" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        target_network = TargetNetwork(model=model)
    
        grad = rand(2)
        optimise!(target_network, grad)
    
        @test target_network.n_optimise == 1    
    end

    @testset "Sync" begin
        model = Chain(Dense(10, 5, relu), Dense(5, 2))
        target_network = TargetNetwork(model=model, sync_freq=2, ρ=0.5)
    
        grad = rand(2)
        target_network.optimise!(grad)
        @test target_network.n_optimise == 1
    
        grad = rand(2)
        target_network.optimise!(grad)
        @test target_network.n_optimise == 0
    
        @test target_network.target[1].weight == 0.5 * target_network.target[1].weight + 0.5 * target_network.network[1].weight
        @test target_network.target[1].bias == 0.5 * target_network.target[1].bias + 0.5 * target_network.network[1].bias
        @test target_network.target[2].weight == 0.5 * target_network.target[2].weight + 0.5 * target_network.network[2].weight
        @test target_network.target[2].bias == 0.5 * target_network.target[2].bias + 0.5 * target_network.network[2].bias    
    end
end

@testset "TargetNetwork" begin 
    m = Chain(Dense(4,1))
    app = Approximator(model = m, optimiser = Flux.Adam(), use_gpu=true)
    tn = TargetNetwork(app, sync_freq = 3, use_gpu=true)
    @test typeof(model(tn)) == typeof(target(tn))
    p1 = Flux.destructure(model(tn))[1]
    pt1 = Flux.destructure(target(tn))[1]
    @test p1 == pt1
    input = gpu(ones(Float32, 4))
    grad = Flux.Zygote.gradient(tn.network) do model
        sum(RLCore.forward(model, input))
    end

    grad_model = grad[1].model
    
    RLCore.optimise!(tn, grad_model)
    @test p1 != Flux.destructure(model(tn))[1]
    @test p1 == Flux.destructure(target(tn))[1]
    RLCore.optimise!(tn, grad_model)
    @test p1 != Flux.destructure(model(tn))[1]
    @test p1 == Flux.destructure(target(tn))[1]
    RLCore.optimise!(tn, grad_model)
    @test Flux.destructure(target(tn))[1] == Flux.destructure(model(tn))[1]
    @test p1 != Flux.destructure(target(tn))[1]
    p2 = Flux.destructure(model(tn))[1]
    RLCore.optimise!(tn, grad_model)
    @test p2 != Flux.destructure(model(tn))[1]
    @test p2 == Flux.destructure(target(tn))[1]
end
