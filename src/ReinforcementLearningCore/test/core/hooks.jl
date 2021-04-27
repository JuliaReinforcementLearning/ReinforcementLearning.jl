let stages = (POST_EPISODE_STAGE, PRE_EPISODE_STAGE)
    @testset "DoEveryNEpisode stage=$(stage),s2=$(s2)" for stage in stages, s2 in stages
        hook = DoEveryNEpisode((x...) -> true; stage)
        if stage === s2
            @test hook(stage, nothing, nothing)
        else
            @test hook(s2, nothing, nothing) === nothing
        end
    end
end
