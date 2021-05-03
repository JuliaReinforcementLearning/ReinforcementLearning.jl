@testset "MultiArmBanditsEnv" begin

    rng = StableRNG(123)
    env = MultiArmBanditsEnv(; rng = rng)
    rewards = []

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    N = 50_000
    for _ in 1:N
        while !is_terminated(env)
            env(rand(rng, legal_action_space(env)))
        end
        push!(rewards, reward(env))
        reset!(env)
    end

    @test isapprox(mean(rewards), mean(env.true_values); atol = 0.01)

end
