@testset "test StopAfterNoImprovement" begin
    env = StateTransformedEnv(CartPoleEnv{Float32}(); state_mapping = deepcopy)
    policy = RandomPolicy(action_space(env))

    total_reward_per_episode = TotalRewardPerEpisode()
    patience = 30
    stop_condition =
        StopAfterNoImprovement(() -> total_reward_per_episode.reward, patience, 0.0f0)

    # stop_condition is called between POST_ACT_STAGE & POST_EPISODE_STAGE.
    # total_reward_per_episode.rewards is updated at POST_EPISODE_STAGE.
    # total_reward_per_episode.reward is updated at POST_ACT_STAGE.
    # so the latter one should be used. or the value is from the previous episode.
    hook = ComposedHook(total_reward_per_episode)
    run(policy, env, stop_condition, hook)

    @test argmax(total_reward_per_episode.rewards) != patience
end

@testset "StopAfterNSeconds" begin
    s = StopAfterNSeconds(0.01)
    @test !s()
    sleep(0.02)
    @test s()
end
