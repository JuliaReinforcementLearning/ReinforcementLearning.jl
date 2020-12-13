@testset "test StopAfterNoImprovement" begin
    env = CartPoleEnv{Float32}() |> StateOverriddenEnv(deepcopy)
    agent = Agent(;
        policy = RandomPolicy(env),
        trajectory = CircularArraySARTTrajectory(;
            capacity = 10_000,
            state = Float32 => (4,),
        ),
    )
    total_reward_per_episode = TotalRewardPerEpisode()
    patience = 30
    stop_condition =
        StopAfterNoImprovement(() -> total_reward_per_episode.reward, patience, 0.0f0)
    # stop_condition is called between POST_ACT_STAGE & POST_EPISODE_STAGE.
    # total_reward_per_episode.rewards is updated at POST_EPISODE_STAGE.
    # total_reward_per_episode.reward is updated at POST_ACT_STAGE.
    # so the latter one should be used. or the value is from the previous episode.
    hook = ComposedHook(total_reward_per_episode)
    run(agent, env, stop_condition, hook)

    @test argmax(total_reward_per_episode.rewards) != patience
end
