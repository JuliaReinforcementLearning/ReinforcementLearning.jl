"""
test_noop!(hook; stages=[PreActStage()])

Tests that the hook is not modified when called with the specified set of stages.
"""
function test_noop!(hook::AbstractHook; stages=[PreActStage(), PostActStage(), PreEpisodeStage(), PostEpisodeStage(), PreExperimentStage(), PostExperimentStage()])
    @testset "hook: $(typeof(hook))" begin
        env = RandomWalk1D()
        env.pos = 7
        policy = RandomPolicy(legal_action_space(env))

        hook_fieldnames = fieldnames(typeof(hook))
        for mode in [:MultiAgent, :SingleAgent]
            for stage in stages
                hook_copy = deepcopy(hook)
                if mode == :SingleAgent
                    push!(hook_copy, stage, policy, env)
                elseif mode == :MultiAgent
                    push!(hook_copy, stage, policy, env, :player_i)
                end
                for field_ in hook_fieldnames
                    if getfield(hook, field_) isa Ref
                        @test getfield(hook, field_)[] == getfield(hook_copy, field_)[]
                    else
                        @test getfield(hook, field_) == getfield(hook_copy, field_)
                    end
                end
            end
        end
    end
end

@testset "TotalRewardPerEpisode" begin
    h_1 = TotalRewardPerEpisode(; is_display_on_exit=true)
    h_2 = TotalRewardPerEpisode(; is_display_on_exit=false)
    h_3 = TotalRewardPerEpisode()
    h_4 = TotalRewardPerEpisode{Float32}()
    h_5 = TotalRewardPerEpisode{Float32}(; is_display_on_exit = false)

    env = RandomWalk1D()
    env.pos = 7
    policy = RandomPolicy(legal_action_space(env))

    for h in (h_1, h_2, h_3, h_4, h_5)
        push!(h, PostActStage(), policy, env)
        @test h.reward == 1
        push!(h, PostEpisodeStage(), policy, env)
        @test h.reward == 0
        @test h.rewards == [1]
        push!(h, PostExperimentStage(), policy, env)

        test_noop!(h; stages=[PreActStage(), PreEpisodeStage(), PreExperimentStage()])
    end
end

@testset "TotalBatchRewardPerEpisode" begin
    env = RandomWalk1D()
    env.pos = 7
    policy = RandomPolicy(legal_action_space(env))

    h_1 = TotalBatchRewardPerEpisode(10; is_display_on_exit = true)
    h_2 = TotalBatchRewardPerEpisode(10; is_display_on_exit = false)
    h_3 = TotalBatchRewardPerEpisode{Float32}(10)
    h_4 = TotalBatchRewardPerEpisode{Float32}(10; is_display_on_exit = false)
    h_5 = TotalBatchRewardPerEpisode(10)

    for h in (h_1, h_2, h_3, h_4, h_5)
        push!(h, PostActStage(), policy, env)
        @test h.reward == fill(1, 10)
        push!(h, PostEpisodeStage(), policy, env)
        @test h.reward == fill(0.0, 10)
        @test h.rewards == fill([1.0], 10)
        push!(h, PostExperimentStage(), policy, env)

        test_noop!(h; stages=[PreActStage(), PreEpisodeStage(), PreExperimentStage()])
    end
end

@testset "DoEveryNStep" begin
    h_1 = DoEveryNStep((hook, agent, env) -> (env.pos += 1); n=2)
    h_2 = DoEveryNStep((hook, agent, env) -> (env.pos += 1); n=1)

    for h in (h_1, h_2)
        env = RandomWalk1D()
        env.pos = 1
        policy = RandomPolicy(legal_action_space(env))
        for t = 1:4
            push!(h, PostActStage(), policy, env)
            @test env.pos == 1+ div(t,h.n)
        end
        test_noop!(h, stages=[PreActStage(), PreEpisodeStage(), PostEpisodeStage(), PreExperimentStage(), PostExperimentStage()])
    end
end

@testset "TimePerStep" begin
    h_1 = TimePerStep()
    h_2 = TimePerStep{Float32}()

    sleep_vect = [0.01, 0.02, 0.03]
    for h in (h_1, h_2)
        push!(h, PostActStage(), 1, 1)
        [(sleep(i); push!(h, PostActStage(), 1, 1)) for i in sleep_vect]
        @test all(0.1 .> h.times[2:end] .> 0)
        test_noop!(h, stages=[PreActStage(), PreEpisodeStage(), PostEpisodeStage(), PreExperimentStage(), PostExperimentStage()])
    end
end

@testset "StepsPerEpisode" begin
    env = RandomWalk1D()
    agent = RandomPolicy()
    h = StepsPerEpisode()

    [push!(h, PostActStage()) for i in 1:100]

    @test h.count == 100

    push!(h, PostEpisodeStage(), agent, env)
    @test h.count == 0
    @test h.steps == [100]

    push!(h, PostExperimentStage(), agent, env)
    @test h.steps == [100, 0]

    test_noop!(h, stages=[PreActStage(), PreEpisodeStage(), PreExperimentStage()])
end

@testset "RewardsPerEpisode" begin
    env = RandomWalk1D()
    env.pos = 1
    agent = RandomPolicy()
    h_1 = RewardsPerEpisode()
    h_2 = RewardsPerEpisode{Float64}()
    h_3 = RewardsPerEpisode{Float16}()

    for h in (h_1, h_2, h_3)
        push!(h, PreEpisodeStage(), agent, env)
        @test h.rewards == [[]]

        env.pos = 1
        push!(h, PostActStage(), agent, env)
        @test h.rewards == [[-1.0]]
        env.pos = 7
        push!(h, PostActStage(), agent, env)
        @test h.rewards == [[-1.0, 1.0]]
        env.pos = 3
        push!(h, PostActStage(), agent, env)
        @test h.rewards == [[-1.0, 1.0, 0.0]]
        test_noop!(h, stages=[PreActStage(), PostEpisodeStage(), PreExperimentStage(), PostExperimentStage()])
    end
end

@testset "DoOnExit" begin
    env = RandomWalk1D()
    env.pos = 1
    agent = RandomPolicy()

    h = DoOnExit((agent, env) -> (env.pos += 1))
    push!(h, PostExperimentStage(), agent, env)
    @test env.pos == 2
end

@testset "DoEveryNEpisode" begin
    h_1 = DoEveryNEpisode((hook, agent, env) -> (env.pos += 1); n=2, stage=PreEpisodeStage())
    h_2 = DoEveryNEpisode((hook, agent, env) -> (env.pos += 1); n=2, stage=PostEpisodeStage())
    h_3 = DoEveryNEpisode((hook, agent, env) -> (env.pos += 1); n=1)
    h_list = (h_1, h_2, h_3)
    stage_list = (PreEpisodeStage(), PostEpisodeStage(), PostEpisodeStage())

    for i in 1:3
        h = h_list[i]
        stage = stage_list[i]
        env = RandomWalk1D()
        env.pos = 1
        policy = RandomPolicy(legal_action_space(env))
        for t in 1:4
            push!(h, stage, policy, env)
            @test env.pos == 1 + div(t,h.n)
        end     
        test_noop!(h, stages=[PreActStage(), PostActStage(), PreExperimentStage(), PostExperimentStage()])
    end
end
