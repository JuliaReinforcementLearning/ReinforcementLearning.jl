@testset "TigerProblemEnv" begin

    rng = StableRNG(123)
    obs_prob = 0.85
    env = TigerProblemEnv(; rng = rng, obs_prob = obs_prob)

    RLBase.test_interfaces!(env)

    rewards = []
    for _ in 1:50_000
        @test state(env) in state_space(env)
        r = 0.0
        while !is_terminated(env)
            env(rand(rng, action_space(env)))
            r += reward(env)
        end
        push!(rewards, r)
        reset!(env)
    end

    # R̂ = 1/3 (-1 + R̂) + 2 * 1/3 (-100/2 + 10 / 2)
    expectation = -45.5
    @test isapprox(mean(rewards), expectation; atol = 0.5)
end
