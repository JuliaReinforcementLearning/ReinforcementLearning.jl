@testset "StopAfterStep" begin
    stop_ = StopAfterStep(10)
    @test sum([stop_() for i in 1:20]) == 11

    stop_ = StopAfterStep(10; is_show_progress=false)
    @test sum([stop_() for i in 1:20]) == 11
end

@testset "StopAfterEpisode" begin
    stop_1 = StopAfterEpisode(2)
    stop_2 = StopAfterEpisode(2; is_show_progress=false)
    stop_3 = StopAfterEpisode(2; is_show_progress=true)
    for stop_ in (stop_1, stop_2)
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))
        @test stop_(policy, env) == false
        env.pos = 7
        @test stop_(policy, env) == false
        @test stop_(policy, env) == true
    end
end
