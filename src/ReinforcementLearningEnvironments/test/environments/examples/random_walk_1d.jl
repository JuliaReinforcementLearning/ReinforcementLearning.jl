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

@testset "RandomPolicy / RandomWalk1D Performance Specs" begin
    env = RandomWalk1D()
    
    # Test zero allocations for RandomPolicy calls
    p = RandomPolicy(legal_action_space(env))
    @test (@allocated p(env)) == 0

    p_ = RandomPolicy()
    @test (@allocated p_(env)) == 0
    @test (@allocated reward(env)) == 0
    @test (@allocated env(1)) == 0
end

@testset "RandomWalk1D Env Updating" begin
    # Reach positive outcome
    env = RandomWalk1D()
    for i in 1:(env.N+1)
        env(1)
    end
    @test env.pos == 1
    @test RLCore.reward(env) == -1

    # Reach negative outcome
    env = RandomWalk1D()
    for i in 1:(env.N+1)
        env(2)
    end
    @test env.pos == env.N
    @test RLCore.reward(env) == 1

    # Reach starting position
    env = RandomWalk1D()
    @test env.pos == 4
    env(1)
    @test env.pos == 3
    env(2)
    @test env.pos == 4
    @test RLCore.reward(env) == 0
end
