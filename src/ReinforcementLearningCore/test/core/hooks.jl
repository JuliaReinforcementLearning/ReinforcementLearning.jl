@testset "TotalRewardPerEpisode" begin
    h_1 = TotalRewardPerEpisode(; is_display_on_exit = true)
    h_2 = TotalRewardPerEpisode(; is_display_on_exit = false)
    h_3 = TotalRewardPerEpisode()

    env = RandomWalk1D()
    env.pos = 7
    policy = RandomPolicy(legal_action_space(env))

    for h in (h_1, h_2, h_3)
        h(PostActStage(), policy, env)
        @test h.reward == 1
        h(PostEpisodeStage(), policy, env)
        @test h.reward == 0
        @test h.rewards == [1]
        h(PostExperimentStage(), policy, env)
    end
end

@testset "StepsPerEpisode" begin
    env = RandomWalk1D()
    agent = RandomPolicy()
    h = StepsPerEpisode()
    [h(PostActStage()) for i in 1:100]
    @test h.count == 100
    h(PostEpisodeStage(), agent, env)
    @test h.count == 0
    @test h.steps == [100]
    h(PostExperimentStage(), agent, env)
    @test h.steps == [100, 0]
end
