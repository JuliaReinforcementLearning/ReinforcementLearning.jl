using Flux

@testset "StockTradingEnv" begin

    env = StockTradingEnvWithTurbulence()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)
end

@testset "StockTradingEnv legal_action_space_mask" begin
    env = StockTradingEnv()
    # For MINIMAL_ACTION_SET, this is not and should not be defined
    @test_throws MethodError legal_action_space_mask(env) 
end

@testset "StockTradingEnv and Flux / QBasedPolicy" begin
    env = StockTradingEnv()

    ns, na = size(state_space(env))[1], size(action_space(env))[1]

    policy = Agent(
        QBasedPolicy(;
            learner = FluxApproximator(
                Chain(
                    Dense(ns, 64, relu),
                    Dense(64, na, relu),
                ),
                Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-5)),
            ),
            explorer = EpsilonGreedyExplorer(ϵ_stable=0.01),
        ),
        Trajectory(
            CircularArraySARTSTraces(;
                capacity = 10,
                state = Float64 => (ns,),
                action = Float64 => (na,),
                reward = Float64 => (),
                terminal = Bool => (),
            ),
            DummySampler(),
            InsertSampleRatioController(),
        ),
    )

    run(
        policy,
        env,
        StopAfterNSteps(10_000),
    )
end
