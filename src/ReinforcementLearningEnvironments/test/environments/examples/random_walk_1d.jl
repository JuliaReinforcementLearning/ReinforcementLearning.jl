@testset "RandomWalk1D" begin

    end_rewards = 3 => 5
    env = RandomWalk1D(; rewards = end_rewards)

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    rng = StableRNG(123)
    N = 50_000
    rewards = []
    for _ = 1:N
        while !is_terminated(env)
            RLBase.act!(env, rand(rng, legal_action_space(env)))
        end
        push!(rewards, reward(env))
        reset!(env)
    end

    @test isapprox(mean(rewards), mean(end_rewards); atol = 0.01)
end

@allocated RLEnvs.random_walk_reward(1, Pair(-1.0, 1.0), 10)

@testset "RandomPolicy / RandomWalk1D Performance Specs" begin
    env = RandomWalk1D()
    reward(env)
    RLBase.act!(env, 1)
    @test (@allocated reward(env)) == 0
    @test (@allocated RLBase.act!(env, 1)) == 0

    # Test zero allocations for RandomPolicy calls
    p = RandomPolicy(legal_action_space(env))
    RLBase.plan!(p, env)
    @test (@allocated RLBase.plan!(p, env)) == 0

    p_ = RandomPolicy()
    RLBase.plan!(p_, env)
    @test (@allocated RLBase.plan!(p_, env)) == 0
end

@testset "Reward Dispatch" begin
    @test RLEnvs.random_walk_reward(1, Pair(-1.0, 1.0), 10) == -1.0
    @test RLEnvs.random_walk_reward(10, Pair(-1.0, 1.0), 10) == 1.0
    @test RLEnvs.random_walk_reward(5, Pair(-1.0, 1.0), 10) == 0.0
end

@testset "RandomWalk1D Env Updating" begin
    # Reach positive outcome
    env = RandomWalk1D()
    for i = 1:(env.N+1)
        RLBase.act!(env, 1)
    end
    @test env.pos == 1
    @test reward(env) == -1
    @test (@allocated reward(env)) == 0

    # Reach negative outcome
    env = RandomWalk1D()
    for i = 1:(env.N+1)
        RLBase.act!(env, 2)
    end
    @test env.pos == env.N
    @test reward(env) == 1
    @test (@allocated reward(env)) == 0

    # Reach starting position
    env = RandomWalk1D()
    @test env.pos == 4
    RLBase.act!(env, 1)
    @test env.pos == 3
    RLBase.act!(env, 2)
    @test env.pos == 4
    @test reward(env) == 0
    @test (@allocated reward(env)) == 0
end

@testset "Full run" begin
    using BSON

    env = RandomWalk1D()
    ns, na = length(state_space(env)), length(action_space(env))

    policy = Agent(
        QBasedPolicy(;
            learner = TDLearner(
                TabularQApproximator(n_state = ns, n_action = na),
                :SARS;
            ),
            explorer = EpsilonGreedyExplorer(ϵ_stable=0.01),
        ),
        Trajectory(
            CircularArraySARTSTraces(;
                capacity = 1,
                state = Int64 => (),
                action = Int64 => (),
                reward = Float64 => (),
                terminal = Bool => (),
            ),
            DummySampler(),
            InsertSampleRatioController(),
        ),
    )

    parameters_dir = mktempdir()

    run(
        policy,
        env,
        StopAfterNSteps(10_000),
        DoEveryNSteps(n=1_000) do t, p, e
            ps = policy.policy.learner.approximator
            f = joinpath(parameters_dir, "parameters_at_step_$t.bson")
            BSON.@save f ps
            println("parameters at step $t saved to $f")
        end
    )
end
