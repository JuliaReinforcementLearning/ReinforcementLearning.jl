@testset "RockPaperScissorsEnv" begin

    rng = StableRNG(123)
    env = RockPaperScissorsEnv()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    rewards = [[], []]
    for _ in 1:50_000
        while !is_terminated(env)
            env(rand(rng, legal_action_space(env)))
        end
        push!(rewards[1], reward(env, 1))
        push!(rewards[2], reward(env, 2))
        reset!(env)
    end

    @test all(rewards[1] .+ rewards[2] .== 0)

end
