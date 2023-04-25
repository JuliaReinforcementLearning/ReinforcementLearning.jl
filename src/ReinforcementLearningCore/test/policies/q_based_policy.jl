using ReinforcementLearningCore

@testset "QBasedPolicy" begin
    policy = QBasedPolicy(;
        learner = TDLearner(;
            approximator = TabularApproximator(
                zeros(Float32, 10, 100),
                0.1,
            ),
            method = :SARS,
            γ = env.δ,
            n = 0,
        ),
        explorer = EpsilonGreedyExplorer(1e-5),
    )
    # QBasedPolicy()    
end
