using ReinforcementLearningCore: stop

@testset "StopAfterStep" begin
    stop_condition = StopAfterStep(10)
    @test sum([stop(stop_condition) for i in 1:20]) == 11

    stop_condition = StopAfterStep(10; is_show_progress=false)
    @test sum([stop(stop_condition) for i in 1:20]) == 11
end

@testset "ComposedStopCondition" begin
    stop_10 = StopAfterStep(10)
    stop_3 = StopAfterStep(3)

    composed_stop = ComposedStopCondition(stop_10, stop_3)
    @test sum([stop(composed_stop) for i in 1:20]) == 18
end

@testset "StopAfterEpisode" begin
    stop_1 = StopAfterEpisode(2)
    stop_2 = StopAfterEpisode(2; is_show_progress=false)
    stop_3 = StopAfterEpisode(2; is_show_progress=true)

    for stop_condition in (stop_1, stop_2)
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))
        @test stop(stop_condition, policy, env) == false
        env.pos = 7
        @test stop(stop_condition, policy, env) == false
        @test stop(stop_condition, policy, env) == true
    end
end

@testset "StopAfterNoImprovement" begin
    stop_1 = StopAfterNoImprovement(() -> 1.0f0, 10)
    stop_2 = StopAfterNoImprovement(() -> 1.0, 10)
    stop_3 = StopAfterNoImprovement(() -> 1, 10)

    for stop_condition in [stop_1, stop_2, stop_3]
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))

        @test sum([stop(stop_condition, policy, env) for i in 1:11]) == 0
        env.pos = 7
        @test sum([stop(stop_condition, policy, env) for i in 1:11]) == 1
    end

    a_4 = 0.0f0
    a_5 = 0.0
    a_6 = 0

    plusone(a) = a += 1

    stop_condition_4 = StopAfterNoImprovement(() -> plusone(a_4), 10)
    stop_condition_5 = StopAfterNoImprovement(() -> plusone(a_5), 10)
    stop_condition_6 = StopAfterNoImprovement(() -> plusone(a_6), 10)

    for stop_condition in [stop_condition_4, stop_condition_5, stop_condition_6]
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))

        @test sum([stop(stop_condition, policy, env) for i in 1:11]) == 0
        env.pos = 7
        @test sum([stop(stop_condition, policy, env) for i in 1:11]) == 1
    end
end
