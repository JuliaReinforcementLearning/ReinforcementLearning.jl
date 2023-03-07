@testset "RandomWalk1D" begin

    end_rewards = 3 => 5
    env = RandomWalk1D(; rewards = end_rewards)

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    rng = StableRNG(123)
    N = 50_000
    rewards = []
    for _ in 1:N
        while !is_terminated(env)
            env(rand(rng, legal_action_space(env)))
        end
        push!(rewards, reward(env))
        reset!(env)
    end

    @test isapprox(mean(rewards), mean(end_rewards); atol = 0.01)
end

@allocated _reward(1, Pair(-1.0, 1.0), 10)

@testset "RandomPolicy / RandomWalk1D Performance Specs" begin
    env = RandomWalk1D()
    reward(env)
    env(1)
    @test (@allocated reward(env)) == 0
    @test (@allocated env(1)) == 0
    
    # Test zero allocations for RandomPolicy calls
    p = RandomPolicy(legal_action_space(env))
    p(env)
    @test (@allocated p(env)) == 0

    p_ = RandomPolicy()
    p_(env)
    @test (@allocated p_(env)) == 0
end

@testset "Reward Dispatch" begin
    @test random_walk_reward(1, Pair(-1.0, 1.0), 10) == -1.0
    @test random_walk_reward(10, Pair(-1.0, 1.0), 10) == 1.0
    @test random_walk_reward(5, Pair(-1.0, 1.0), 10) == 0.0
end

@testset "RandomWalk1D Env Updating" begin
    # Reach positive outcome
    env = RandomWalk1D()
    for i in 1:(env.N+1)
        env(1)
    end
    @test env.pos == 1
    @test reward(env) == -1

    # Reach negative outcome
    env = RandomWalk1D()
    for i in 1:(env.N+1)
        env(2)
    end
    @test env.pos == env.N
    @test reward(env) == 1

    # Reach starting position
    env = RandomWalk1D()
    @test env.pos == 4
    env(1)
    @test env.pos == 3
    env(2)
    @test env.pos == 4
    @test reward(env) == 0
end
