import ReinforcementLearningCore.check!

@testset "StopAfterNSteps" begin
    stop_condition = StopAfterNSteps(10)
    env = RandomWalk1D()
    policy = RandomPolicy(legal_action_space(env))

    @test sum([check!(stop_condition, policy, env) for i in 1:20]) == 11

    stop_condition = StopAfterNSteps(10; is_show_progress=false)
    @test sum([check!(stop_condition, policy, env) for i in 1:20]) == 11
end

@testset "StopIfAny" begin
    stop_10 = StopAfterNSteps(10)
    stop_3 = StopAfterNSteps(3)

    env = RandomWalk1D()
    policy = RandomPolicy(legal_action_space(env))

    composed_stop = StopIfAny(stop_10, stop_3)
    @test sum([RLCore.check!(composed_stop, policy, env) for i in 1:20]) == 18
end

@testset "StopIfAll" begin
    stop_10 = StopAfterNSteps(10)
    stop_3 = StopAfterNSteps(3)

    env = RandomWalk1D()
    policy = RandomPolicy(legal_action_space(env))

    composed_stop = StopIfAll(stop_10, stop_3)
    @test sum([RLCore.check!(composed_stop, policy, env) for i in 1:20]) == 11
end

@testset "StopAfterNEpisodes" begin
    stop_1 = StopAfterNEpisodes(2)
    stop_2 = StopAfterNEpisodes(2; is_show_progress=false)
    stop_3 = StopAfterNEpisodes(2; is_show_progress=true)

    for stop_condition in (stop_1, stop_2)
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))
        @test check!(stop_condition, policy, env) == false
        env.pos = 7
        @test check!(stop_condition, policy, env) == false
        @test check!(stop_condition, policy, env) == true
    end
end

@testset "StopAfterNoImprovement" begin
    stop_1 = StopAfterNoImprovement(() -> 1.0f0, 10)
    stop_2 = StopAfterNoImprovement(() -> 1.0, 10)
    stop_3 = StopAfterNoImprovement(() -> 1, 10)

    for stop_condition in [stop_1, stop_2, stop_3]
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))

        @test sum([check!(stop_condition, policy, env) for i in 1:11]) == 0
        env.pos = 7
        @test sum([check!(stop_condition, policy, env) for i in 1:11]) == 1
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

        @test sum([check!(stop_condition, policy, env) for i in 1:11]) == 0
        env.pos = 7
        @test sum([check!(stop_condition, policy, env) for i in 1:11]) == 1
    end
end
