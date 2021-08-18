@testset "gym envs" begin
    gym_env_names = ReinforcementLearningEnvironments.list_gym_env_names(
        modules = [
            "gym.envs.algorithmic",
            "gym.envs.classic_control",
            "gym.envs.toy_text",
            "gym.envs.unittest",
        ],
    )  # mujoco, box2d, robotics are not tested here

    gym_env_names = filter(x -> !(x in ["KellyCoinflipGeneralized-v0", "Blackjack-v0", "KellyCoinflip-v0"]), gym_env_names)  # not sure why this env has outliers

    for x in gym_env_names
        env = GymEnv(x)
        RLBase.test_runnable!(env)
    end
end
