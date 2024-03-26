@testset "QBasedPolicy" begin

    @testset "constructor" begin
        q_approx = TabularQApproximator(n_state = 5, n_action = 10)
        explorer = EpsilonGreedyExplorer(0.1)
        learner = TDLearner(q_approx, :SARS)
        p = QBasedPolicy(learner, explorer)
        @test p.learner == learner
        @test p.explorer == explorer
    end

    @testset "plan!" begin
        @testset "plan! without player argument" begin
            env = TicTacToeEnv()
            q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)))
            learner = TDLearner(q_approx, :SARS)
            explorer = EpsilonGreedyExplorer(0.1)
            policy = QBasedPolicy(learner, explorer)
            @test 1 <= RLBase.plan!(policy, env) <= 9
        end

        @testset "plan! with player argument" begin
            env = TicTacToeEnv()
            q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)))
            learner = TDLearner(q_approx, :SARS)
            explorer = EpsilonGreedyExplorer(0.1)
            policy = QBasedPolicy(learner, explorer)
            player = Player(:player1)
            @test 1 <= RLBase.plan!(policy, env) <= 9
        end
    end

    # Test prob function
    @testset "prob" begin
        env = TicTacToeEnv()
        q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)))
        learner = TDLearner(q_approx, :SARS)
        explorer = EpsilonGreedyExplorer(0.1)
        policy = QBasedPolicy(learner, explorer)
        trajectory = Trajectory(
            CircularArraySARTSTraces(;
                capacity = 1,
                state = Int64 => (),
                action = Int64 => (),
                reward = Float64 => (),
                terminal = Bool => (),
            ),
            DummySampler(),
            InsertSampleRatioController(),
        )
        t = (state=2, action=3)
        push!(trajectory, t)
        next_state = 4
        t = (action=3, state=next_state, reward=5.0, terminal=false)
        push!(trajectory, t)
        optimise!(policy, PostActStage(), trajectory)
        prob = RLBase.prob(policy, env)
        @test prob.p == [0.9111111111111111, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112]
    end

    # Test optimise! function
    @testset "optimise!" begin
        env = TicTacToeEnv()
        q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)))
        explorer = EpsilonGreedyExplorer(0.1)
        learner = TDLearner(q_approx, :SARS, γ=0.95, α=0.01, n=0)
        policy = QBasedPolicy(learner, explorer)
        trajectory = Trajectory(
            CircularArraySARTSTraces(;
                capacity = 1,
                state = Int64 => (),
                action = Int64 => (),
                reward = Float64 => (),
                terminal = Bool => (),
            ),
            DummySampler(),
            InsertSampleRatioController(),
        )
        t = (state=4, action=3)
        push!(trajectory, t)
        next_state = 4
        t = (action=3, state=next_state, reward=5.0, terminal=false)
        push!(trajectory, t)

        RLBase.optimise!(policy, PostActStage(), trajectory)
        @test policy.learner.approximator.model[t.action, t.state] ≈ 0.05
        RLBase.optimise!(policy, PostActStage(), trajectory)
        @test policy.learner.approximator.model[t.action, t.state] ≈ 0.09997500000000001
    
        for i in 1:100000
            RLBase.optimise!(policy, PostActStage(), trajectory)
        end
        @test policy.learner.approximator.model[t.action, t.state] ≈ t.reward / (1-policy.learner.γ) atol=0.01
    end
end
