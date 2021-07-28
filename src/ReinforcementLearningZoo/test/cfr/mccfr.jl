@testset "ExternalSamplingMCCFR" begin
    env = OpenSpielEnv("kuhn_poker")
    p = ExternalSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(1000))
    @test_broken RLZoo.nash_conv(p, env) < 0.05

    env = OpenSpielEnv("leduc_poker")
    p = ExternalSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(1000))
    @test RLZoo.nash_conv(p, env) < 2.5

    env = OpenSpielEnv("liars_dice")
    p = ExternalSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(100))
    @test RLZoo.nash_conv(p, env) < 1.6

    env = OpenSpielEnv("kuhn_poker(players=3)")
    p = ExternalSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(100))
    @info "nash_conv for kuhn_poker(players=3)" RLZoo.nash_conv(p, env)
end

@testset "OutcomeSamplingMCCFR" begin
    env = OpenSpielEnv("kuhn_poker")
    p = OutcomeSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(10000))
    @test_broken RLZoo.nash_conv(p, env) < 0.04

    env = OpenSpielEnv("leduc_poker")
    p = OutcomeSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(10000))
    @test RLZoo.nash_conv(p, env) < 3

    env = OpenSpielEnv("liars_dice")
    p = OutcomeSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(1000))
    @test RLZoo.nash_conv(p, env) < 1.7

    env = OpenSpielEnv("kuhn_poker(players=3)")
    p = OutcomeSamplingMCCFRPolicy(rng = StableRNG(123))
    run(p, env, StopAfterStep(1000))
    @info "nash_conv for kuhn_poker(players=3)" RLZoo.nash_conv(p, env)
end
