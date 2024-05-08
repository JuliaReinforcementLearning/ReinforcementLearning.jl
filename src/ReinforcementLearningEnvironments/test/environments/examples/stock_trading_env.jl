@testset "StockTradingEnv" begin

    env = StockTradingEnvWithTurbulence()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)
end

@test "StockTradingEnv legal_action_space_mask" begin
    env = StockTradingEnv()
    # For MINIMAL_ACTION_SET, this is not and should not be defined
    @test_throws MethodError legal_action_space_mask(env) 
end
