@testset "basic environment test" begin

    function basic_env_test(env, n = 100)
        reset!(env)
        observation_space = get_observation_space(env)
        action_space = get_action_space(env)
        @test observation_space isa AbstractSpace
        @test action_space isa AbstractSpace
        @test reset!(env) == nothing
        for _ = 1:n
            a = rand(action_space)
            @test a in action_space
            @test env(a) === nothing
            obs = observe(env)
            @test get_state(obs) in observation_space
            if get_terminal(obs)
                reset!(env)
            end
        end
    end

    gym_env_names = ReinforcementLearningEnvironments.list_gym_env_names(modules = [
        "gym.envs.algorithmic",
        "gym.envs.classic_control",
        "gym.envs.toy_text",
        "gym.envs.unittest",
    ])  # mujoco, box2d, robotics are not tested here

    gym_env_names = filter(x -> x != "KellyCoinflipGeneralized-v0", gym_env_names)  # not sure why this env has outliers

    atari_env_names = ReinforcementLearningEnvironments.list_atari_rom_names()
    atari_env_names = filter(x -> x != "defender", atari_env_names)

    for env_exp in [
        # :(basic_ViZDoom_env()),  # comment out due to https://github.com/JuliaReinforcementLearning/ViZDoom.jl/issues/7
        :(POMDPEnv(TigerPOMDP())),
        :(MountainCarEnv()),
        :(ContinuousMountainCarEnv()),
        :(PendulumEnv()),
        (:(AtariEnv(;name=$x)) for x in atari_env_names)...,
        (:(GymEnv($x)) for x in gym_env_names)...,
    ]

        @info "Testing $env_exp"
        env = eval(env_exp)
        basic_env_test(env)
    end
end
