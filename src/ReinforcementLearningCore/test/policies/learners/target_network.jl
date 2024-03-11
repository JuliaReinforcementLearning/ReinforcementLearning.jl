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
