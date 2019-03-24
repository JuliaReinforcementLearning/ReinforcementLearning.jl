@testset "basic environment test" begin

    function basic_env_test(env, n=100)
        os = observation_space(env)
        as = action_space(env)
        @test os isa AbstractSpace
        @test as isa AbstractSpace
        @test reset!(env) == nothing
        for _ in 1:n
            a = rand(as)
            @test a in as
            obs, reward, isdone = interact!(env, a)
            @test obs in os
            if isdone
                reset!(env)
            end
        end
    end

    for env in [CartPoleEnv(),
        MountainCarEnv(),
        PendulumEnv(),
        MDPEnv(LegacyGridWorld()),
        POMDPEnv(TigerPOMDP()),
        SimpleMDPEnv(),
        deterministic_MDP(),
        absorbing_deterministic_tree_MDP(),
        stochastic_MDP(),
        stochastic_tree_MDP(),
        deterministic_tree_MDP_with_rand_reward(),
        deterministic_tree_MDP(),
        deterministic_MDP(),
        AtariEnv("pong")
        ]
    basic_env_test(env)
    end
end