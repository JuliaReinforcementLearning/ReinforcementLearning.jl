@testset "nash_conv" begin
    env = OpenSpielEnv(
        "kuhn_poker";
        default_state_style = RLBase.Information{String}(),
        is_chance_agent_required = true,
    )
    p = TabularRandomPolicy()

    @test RLZoo.nash_conv(p, env) == 11 / 12

    p = get_optimal_kuhn_policy()
    @test RLZoo.nash_conv(p, env) == 0.0

    env = OpenSpielEnv(
        "leduc_poker";
        default_state_style = RLBase.Information{String}(),
        is_chance_agent_required = true,
    )
    p = TabularRandomPolicy()
    @test RLZoo.nash_conv(p, env) ≈ 4.747222222222222

    env = OpenSpielEnv(
        "kuhn_poker(players=3)";
        default_state_style = RLBase.Information{String}(),
        is_chance_agent_required = true,
    )
    p = TabularRandomPolicy()
    @test RLZoo.nash_conv(p, env) ≈ 2.0624999999999996

    env = OpenSpielEnv(
        "kuhn_poker(players=4)";
        default_state_style = RLBase.Information{String}(),
        is_chance_agent_required = true,
    )
    p = TabularRandomPolicy()
    @test RLZoo.nash_conv(p, env) ≈ 3.4760416666666663
end
