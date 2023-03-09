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

