@testset "StockTradingEnv" begin

    env = StockTradingEnvWithTurbulence()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)
end
