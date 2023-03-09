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

@testset "TimePerStep" begin
    h_1 = TimePerStep()
    h_2 = TimePerStep{Float32}()

    sleep_vect = [0.1, 0.2, 0.3]
    for h in (h_1, h_2)
        h(PostActStage(), 1, 1)
        [(sleep(i); h(PostActStage(), 1, 1)) for i in sleep_vect]
        @test all(round.(h.times[end-2:end]; digits=1) .≈ sleep_vect)
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

@testset "RewardsPerEpisode" begin
    env = RandomWalk1D()
    env.pos = 1
    agent = RandomPolicy()
    h_0 = RewardsPerEpisode()
    h_1 = RewardsPerEpisode{Float64}()
    h_2 = RewardsPerEpisode{Float16}()

    for h in [h_0, h_1, h_2]
        h(PreEpisodeStage(), agent, env)
        @test h.rewards == [[]]

        env.pos = 1
        h(PostActStage(), agent, env)
        @test h.rewards == [[-1.0]]
        env.pos = 7
        h(PostActStage(), agent, env)
        @test h.rewards == [[-1.0, 1.0]]
        env.pos = 3
        h(PostActStage(), agent, env)
        @test h.rewards == [[-1.0, 1.0, 0.0]]
    end
end
