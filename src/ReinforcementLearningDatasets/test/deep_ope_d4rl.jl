using ReinforcementLearningBase
using ReinforcementLearningEnvironments
using PyCall
using UnicodePlots

@testset "d4rl_policies" begin
    model = d4rl_policy("ant", "online", 10)

    @test typeof(model) <: D4RLGaussianNetwork 

    env = GymEnv("ant-medium-v0")

    a = state(env) |> model

    @test action_space(env) |> size == size(a[1])
end

@testset "d4rl_policy_evaluate" begin
    plt = deep_ope_d4rl_evaluate("halfcheetah", "online", 10; num_evaluations=100)
    @test typeof(plt) <: UnicodePlots.Plot
end