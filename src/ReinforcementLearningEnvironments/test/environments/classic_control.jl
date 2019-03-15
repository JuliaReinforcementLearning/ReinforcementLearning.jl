@testset "classic control environments" begin
    basic_env_test(CartPoleEnv())
    basic_env_test(MountainCarEnv())
    basic_env_test(PendulumEnv())
end