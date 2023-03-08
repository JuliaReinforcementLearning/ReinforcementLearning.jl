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

@testset "StopAfterNoImprovement" begin
    using ReinforcementLearningBase
    using ReinforcementLearningCore
    using ReinforcementLearningEnvironments
    using BenchmarkTools
    using JET

@testset "StopAfterNoImprovement" begin
    stop_1 = StopAfterNoImprovement(() -> 1.0f0, 10)
    stop_2 = StopAfterNoImprovement(() -> 1.0, 10)
    stop_3 = StopAfterNoImprovement(() -> 1, 10)

    for stop_ in [stop_1, stop_2, stop_3]
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))
    
        @test sum([stop_(policy, env) for i in 1:11]) == 0
        env.pos = 7
        @test sum([stop_(policy, env) for i in 1:11]) == 1
    end

    a_4 = 0.0f0
    a_5 = 0.0
    a_6 = 0

    plusone(a) = a += 1

    stop_4 = StopAfterNoImprovement(() -> plusone(a_4), 10)
    stop_5 = StopAfterNoImprovement(() -> plusone(a_5), 10)
    stop_6 = StopAfterNoImprovement(() -> plusone(a_6), 10)

    for stop_ in [stop_4, stop_5, stop_6]
        env = RandomWalk1D()
        policy = RandomPolicy(legal_action_space(env))
    
        @test sum([stop_(policy, env) for i in 1:11]) == 0
        env.pos = 7
        @test sum([stop_(policy, env) for i in 1:11]) == 1
    end
end
