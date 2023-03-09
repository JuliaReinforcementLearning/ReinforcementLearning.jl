function test_noop!(hook::AbstractHook; stages=[PreActStage(), PostActStage(), PreEpisodeStage(), PostEpisodeStage(), PreExperimentStage(), PostExperimentStage()])
    @testset "hook: $(typeof(hook))" begin
        env = RandomWalk1D()
        env.pos = 7
        policy = RandomPolicy(legal_action_space(env))

        hook_fieldnames = fieldnames(typeof(hook))
        for stage in stages
            hook_copy = deepcopy(hook)
            hook_copy(stage, policy, env)
            for field_ in hook_fieldnames
                @test getfield(hook, field_) == getfield(hook_copy, field_)
            end
        end
    end
end

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

        test_noop!(h; stages=[PreActStage(), PreEpisodeStage(), PreExperimentStage()])
    end
end

@testset "TotalBatchRewardPerEpisode" begin
    env = RandomWalk1D()
    env.pos = 7
    policy = RandomPolicy(legal_action_space(env))

    h_1 = TotalBatchRewardPerEpisode(10; is_display_on_exit = true)
    h_2 = TotalBatchRewardPerEpisode(10; is_display_on_exit = false)
    h_3 = TotalBatchRewardPerEpisode{Float32}(10)
    h_4 = TotalBatchRewardPerEpisode{Float32}(10; is_display_on_exit = false)
    h_5 = TotalBatchRewardPerEpisode(10)

    for h in (h_1, h_2, h_3, h_4, h_5)
        h(PostActStage(), policy, env)
        @test h.reward == fill(1, 10)
        h(PostEpisodeStage(), policy, env)
        @test h.reward == fill(0.0, 10)
        @test h.rewards == fill([1.0], 10)
        h(PostExperimentStage(), policy, env)

        test_noop!(h; stages=[PreActStage(), PreEpisodeStage(), PreExperimentStage()])
    end
end





# NOOP Check for hooks!
# E.g. nothing should change for certain states
# PostEpisodeStage()
