@testset "classic control environments" begin
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
        deterministic_MDP()
        ]
    basic_env_test(env)
    end
end