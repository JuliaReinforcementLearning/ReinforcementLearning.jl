@testset "TotalBatchRewardPerEpisode" begin
    env = RandomWalk1D()
    env.pos = 7
    policy = RandomPolicy(RLCore.legal_action_space(env))

    h_1 = TotalBatchRewardPerEpisode(10; is_display_on_exit = true)
    h_2 = TotalBatchRewardPerEpisode(10; is_display_on_exit = false)
    h_3 = TotalBatchRewardPerEpisode{Float32}(10)
    h_4 = TotalBatchRewardPerEpisode{Float32}(10; is_display_on_exit = false)
    h_5 = TotalBatchRewardPerEpisode(10)

    for h in (h_1, h_2, h_3, h_4, h_5)
        h_ = test_run!(h)
        @test length(h_.rewards) == 10

        push!(h, PostActStage(), policy, env)
        @test h.reward == fill(1, 10)
        push!(h, PostEpisodeStage(), policy, env)
        @test h.reward == fill(0.0, 10)
        @test h.rewards == fill([1.0], 10)
        push!(h, PostExperimentStage(), policy, env)

        test_noop!(h; stages=[PreActStage(), PreEpisodeStage(), PreExperimentStage()])
    end
end
