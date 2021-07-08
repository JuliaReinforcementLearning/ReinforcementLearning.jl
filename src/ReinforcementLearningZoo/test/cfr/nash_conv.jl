@testset "nash_conv" begin
    env = OpenSpielEnv("kuhn_poker")
    p = TabularRandomPolicy()

    @test RLZoo.nash_conv(p, env) == 11 / 12

    p = get_optimal_kuhn_policy()
    @test RLZoo.nash_conv(p, env) == 0.0

    env = OpenSpielEnv("leduc_poker")
    p = TabularRandomPolicy()
    @test isapprox(RLZoo.nash_conv(p, env), 4.747222222222222; atol = 0.0001)

    env = OpenSpielEnv("kuhn_poker(players=3)")
    p = TabularRandomPolicy()
    @test RLZoo.nash_conv(p, env) ≈ 2.0624999999999996

    env = OpenSpielEnv("kuhn_poker(players=4)")
    p = TabularRandomPolicy()
    @test RLZoo.nash_conv(p, env) ≈ 3.4760416666666663

    env = KuhnPokerEnv()
    p = TabularRandomPolicy()
    @test RLZoo.nash_conv(p, env) == 11 / 12

    p = get_optimal_kuhn_policy(env)
    @test RLZoo.nash_conv(p, env) == 0.0
end
