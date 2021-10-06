@testset "gym envs" begin
    gym_env_names = ReinforcementLearningEnvironments.list_gym_env_names(
        modules = ["gym.envs.algorithmic", "gym.envs.classic_control", "gym.envs.unittest"],
    )  # mujoco, box2d, robotics are not tested here

    for x in gym_env_names
        env = GymEnv(x)
        RLBase.test_runnable!(env)
    end
end
