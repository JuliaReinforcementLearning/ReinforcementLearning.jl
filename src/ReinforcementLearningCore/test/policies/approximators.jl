
@testset "approximators.jl" begin
    @testset "TargetNetwork" begin 
        m = Chain(Dense(4,1))
        app = Approximator(model = m, optimiser = Flux.Adam())
        tn = TargetNetwork(app, sync_freq = 3)
        @test typeof(model(tn)) == typeof(target(tn))
        p1 = Flux.destructure(model(tn))[1]
        pt1 = Flux.destructure(target(tn))[1]
        @test p1 == pt1
        gs = Flux.Zygote.gradient(Flux.params(tn)) do 
            sum(RLCore.forward(tn, ones(Float32, 4)))
        end
        RLCore.optimise!(tn, gs)
        @test p1 != Flux.destructure(model(tn))[1]
        @test p1 == Flux.destructure(target(tn))[1]
        RLCore.optimise!(tn, gs)
        @test p1 != Flux.destructure(model(tn))[1]
        @test p1 == Flux.destructure(target(tn))[1]
        RLCore.optimise!(tn, gs)
        @test Flux.destructure(target(tn))[1] == Flux.destructure(model(tn))[1]
        @test p1 != Flux.destructure(target(tn))[1]
        p2 = Flux.destructure(model(tn))[1]
        RLCore.optimise!(tn, gs)
        @test p2 != Flux.destructure(model(tn))[1]
        @test p2 == Flux.destructure(target(tn))[1]
    end
end