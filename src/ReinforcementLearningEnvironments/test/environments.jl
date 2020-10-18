@testset "basic environment test" begin

    function basic_env_test(env, n = 100)
        reset!(env)
        action_space = get_actions(env)
        for _ in 1:n
            a = rand(action_space)
            @test a in action_space
            env(a)
            if get_terminal(env)
                reset!(env)
            end
        end
    end

    gym_env_names = ReinforcementLearningEnvironments.list_gym_env_names(
        modules = [
            "gym.envs.algorithmic",
            "gym.envs.classic_control",
            "gym.envs.toy_text",
            "gym.envs.unittest",
        ],
    )  # mujoco, box2d, robotics are not tested here

    gym_env_names = filter(x -> x != "KellyCoinflipGeneralized-v0", gym_env_names)  # not sure why this env has outliers

    atari_env_names = ReinforcementLearningEnvironments.list_atari_rom_names()
    atari_env_names = filter(x -> x ∉ ["pacman", "surround"], atari_env_names)

    for env_exp in [
        # :(basic_ViZDoom_env()),  # comment out due to https://github.com/JuliaReinforcementLearning/ViZDoom.jl/issues/7
        # (:(SnakeGameEnv())),  # avoid breaking CI
        :(POMDPEnv(TigerPOMDP())),
        :(MDPEnv(MountainCar())),
        :(MountainCarEnv()),
        :(ContinuousMountainCarEnv()),
        :(AcrobotEnv()),
        :(PendulumEnv()),
        :(MountainCarEnv(T = Float32)),
        :(ContinuousMountainCarEnv(T = Float32)),
        :(AcrobotEnv(T = Float32)),
        :(PendulumEnv(T = Float32)),
        :(PendulumNonInteractiveEnv()),
        :(BlackjackEnv()),
        (:(AtariEnv(; name = $x)) for x in atari_env_names)...,
        (:(GymEnv($x)) for x in gym_env_names)...,
    ]
        @info "Testing $env_exp"
        env = eval(env_exp)
        basic_env_test(env)
    end
end
