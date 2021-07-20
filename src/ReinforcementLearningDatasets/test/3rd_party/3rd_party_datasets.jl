@testset "get_dataset" begin
    d4rl_dataset = get_dataset("halfcheetah-expert-v0", "d4rl")
    d4rl_pybullet_dataset = get_dataset("hopper-bullet-mixed-v0", "d4rl_pybullet")
    #d4rl_atari_dataset = get_dataset("breakout-mixed-v0", "d4rl_atari")

    @test isa(d4rl_dataset, NamedTuple{SARTS})
    @test isa(d4rl_pybullet_dataset, NamedTuple{SART})
    #@test isa(d4rl_atari_dataset, NamedTuple{SART})

    @test length(d4rl_dataset[:reward]) > 0
    @test length(d4rl_pybullet_dataset[:reward]) > 0
    #@test length(d4rl_atari_dataset[:reward]) > 0
end

@testset "env_names" begin
    d4rl_env_names = env_names("d4rl")
    d4rl_pybullet_env_names = env_names("d4rl_pybullet")
    d4rl_atari_env_names = env_names("d4rl_atari")

    @test length(d4rl_env_names) > 0
    @test length(d4rl_atari_env_names) > 0
    @test length(d4rl_pybullet_env_names) > 0
end