@testset "StockTradingEnv" begin

    env = StockTradingEnvWithTurbulence()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)
end

@test "StockTradingEnv legal_action_space_mask" begin
    env = StockTradingEnv()
    @test legal_action_space_mask(env) == ones(30)
end

