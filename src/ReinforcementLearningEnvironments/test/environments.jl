@testset "basic environment test" begin

    function basic_env_test(env, n=100)
        reset!(env)
        os = observation_space(env)
        as = action_space(env)
        @test os isa AbstractSpace
        @test as isa AbstractSpace
        @test reset!(env) == nothing
        for _ in 1:n
            a = rand(as)
            @test a in as
            @test interact!(env, a) === nothing
            obs = observe(env)
            @test get_state(obs) in os
            if get_terminal(obs)
                reset!(env)
            end
        end
    end

    function basic_env_test(env::HanabiEnv, n=100)
        reset!(env)
        @test reset!(env) == nothing
        for _ in 1:n
            a = rand(legal_actions(env))
            interact!(env, a)
            obs = observe(env)
            if get_terminal(obs)
                reset!(env)
            end
        end
    end

    gym_env_names = ReinforcementLearningEnvironments.list_gym_env_names(modules=[
        "gym.envs.algorithmic",
        "gym.envs.classic_control",
        "gym.envs.toy_text",
        "gym.envs.unittest"])  # mujoco, box2d, robotics are not tested here

    gym_env_names = filter(x -> x != "KellyCoinflipGeneralized-v0", gym_env_names)  # not sure why this env has outliers

    atari_env_names = ReinforcementLearningEnvironments.list_atari_rom_names()
    atari_env_names = filter(x -> x != "defender", atari_env_names)

    for env_exp in [
        :(HanabiEnv()),
        # :(basic_ViZDoom_env()),  # comment out due to https://github.com/JuliaReinforcementLearning/ViZDoom.jl/issues/7
        :(CartPoleEnv()),
        :(MountainCarEnv()),
        :(ContinuousMountainCarEnv()),
        :(PendulumEnv()),
        :(MDPEnv(LegacyGridWorld())),
        :(POMDPEnv(TigerPOMDP())),
        :(SimpleMDPEnv()),
        :(DiscreteMazeEnv()),
        :(deterministic_MDP()),
        :(absorbing_deterministic_tree_MDP()),
        :(stochastic_MDP()),
        :(stochastic_tree_MDP()),
        :(deterministic_tree_MDP_with_rand_reward()),
        :(deterministic_tree_MDP()),
        :(deterministic_MDP()),
        (:(AtariEnv($x)) for x in atari_env_names)...,
        (:(GymEnv($x)) for x in gym_env_names)...
        ]

        @info "Testing $env_exp"
        env = eval(env_exp)
        basic_env_test(env)
    end
end
