@testset "stop_conditions" begin
    episode_end_obs = Observation(; state = nothing, reward = 1.0, terminal = true)
    episode_not_end_obs = Observation(; state = nothing, reward = 1.0, terminal = false)
    agent, env = nothing, nothing

    s = StopAfterStep(2)

    @test s(agent, env, episode_not_end_obs) == false
    @test s(agent, env, episode_not_end_obs) == true
    @test s(agent, env, episode_not_end_obs) == true
    @test s(agent, env, episode_not_end_obs) == true

    s = StopAfterEpisode(2)

    @test s(agent, env, episode_end_obs) == false  # dummy first
    @test s(agent, env, episode_not_end_obs) == false
    @test s(agent, env, episode_end_obs) == true
    @test s(agent, env, episode_not_end_obs) == true
    @test s(agent, env, episode_end_obs) == true

    s = StopWhenDone()

    @test s(agent, env, episode_not_end_obs) == false
    @test s(agent, env, episode_end_obs) == true
    @test s(agent, env, episode_not_end_obs) == false
    @test s(agent, env, episode_end_obs) == true

    s = ComposedStopCondition([StopAfterEpisode(2), StopAfterStep(2)])

    @test s(agent, env, episode_not_end_obs) == false
    @test s(agent, env, episode_end_obs) == true
    @test s(agent, env, episode_not_end_obs) == true
    @test s(agent, env, episode_end_obs) == true
    @test s(agent, env, episode_not_end_obs) == true
    @test s(agent, env, episode_end_obs) == true
end