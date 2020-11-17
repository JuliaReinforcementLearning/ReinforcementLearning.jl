using DistributedReinforcementLearning
using Test
using ReinforcementLearningBase
using Flux

@testset "DistributedReinforcementLearning.jl" begin

    @testset "Trainer" begin
        t = actor(Trainer(RandomPolicy()))
        tmp_mailbox = Channel(10)
        task = @async for _ in 1:100
            put!(t, BatchDataMsg(nothing))
            put!(t, FetchParamMsg(tmp_mailbox))
        end

        for _ in 1:100
            @test take!(tmp_mailbox).data == Flux.Params([])
        end
    end

end
