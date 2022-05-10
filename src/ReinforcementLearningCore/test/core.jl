@testset "core" begin
    @testset "simple workflow" begin
        env = StateTransformedEnv(CartPoleEnv{Float32}(); state_mapping = deepcopy)
        policy = RandomPolicy(action_space(env))
        N_EPISODE = 10_000
        hook = TotalRewardPerEpisode()
        run(policy, env, StopAfterEpisode(N_EPISODE), hook)

        @test isapprox(sum(hook[]) / N_EPISODE, 21; atol = 2)
    end

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
        run(policy, env, stop_condition, total_reward_per_episode)

        @test argmax(total_reward_per_episode.rewards) + patience ==
              length(total_reward_per_episode.rewards)
    end

    @testset "StopAfterNSeconds" begin
        s = StopAfterNSeconds(0.01)
        @test !s()
        sleep(0.02)
        @test s()
    end
end