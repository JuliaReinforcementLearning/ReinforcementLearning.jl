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
end
