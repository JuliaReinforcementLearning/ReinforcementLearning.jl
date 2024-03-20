using ReinforcementLearningFarm: TotalRewardPerLastNEpisodes

@testset "TotalRewardPerLastNEpisodes" begin
    @testset "Single Agent" begin
        hook = TotalRewardPerLastNEpisodes(max_episodes = 10)
        env = TicTacToeEnv()
        agent = RandomPolicy()

        for i = 1:15
            push!(hook, PreEpisodeStage(), agent, env)
            push!(hook, PostActStage(), agent, env)
            @test length(hook.rewards) == min(i, 10)
            @test hook.rewards[min(i, 10)] == reward(env)
        end
    end

    @testset "MultiAgent" begin
        hook = TotalRewardPerLastNEpisodes(max_episodes = 10)
        env = TicTacToeEnv()
        agent = RandomPolicy()

        for i = 1:15
            push!(hook, PreEpisodeStage(), agent, env, :Cross)
            push!(hook, PostActStage(), agent, env, :Cross)
            @test length(hook.rewards) == min(i, 10)
            @test hook.rewards[min(i, 10)] == reward(env, :Cross)
        end
    end
end
