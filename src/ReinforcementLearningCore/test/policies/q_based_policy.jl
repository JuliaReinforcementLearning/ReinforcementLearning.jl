@testset "QBasedPolicy" begin

    @testset "constructor" begin
        q_approx = TabularQApproximator(n_state = 5, n_action = 10, opt = InvDecay(0.5))
        explorer = EpsilonGreedyExplorer(0.1)
        learner = TDLearner(q_approx, :SARS)
        p = QBasedPolicy(learner, explorer)
        @test p.learner == learner
        @test p.explorer == explorer
    end

    @testset "plan!" begin
        @testset "plan! without player argument" begin
            env = TicTacToeEnv()
            q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)), opt = InvDecay(0.5))
            learner = TDLearner(q_approx, :SARS)
            explorer = EpsilonGreedyExplorer(0.1)
            policy = QBasedPolicy(learner, explorer)
            @test 1 <= RLBase.plan!(policy, env) <= 9
        end

        @testset "plan! with player argument" begin
            env = TicTacToeEnv()
            q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)), opt = InvDecay(0.5))
            learner = TDLearner(q_approx, :SARS)
            explorer = EpsilonGreedyExplorer(0.1)
            policy = QBasedPolicy(learner, explorer)
            player = :player1
            @test 1 <= RLBase.plan!(policy, env) <= 9
        end
    end

    # Test prob function
    @testset "prob" begin
        env = TicTacToeEnv()
        q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)), opt = InvDecay(0.5))
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
        t = (next_state=3, reward=5.0, terminal=false)
        push!(trajectory, t)
        optimise!(policy, PostActStage(), trajectory)
        prob = RLBase.prob(policy, env)
        # Add assertions here
    end

    # Test optimise! function
    @testset "optimise!" begin
        env = TicTacToeEnv()
        q_approx = TabularQApproximator(n_state = 5, n_action = length(action_space(env)), opt = InvDecay(0.5))
        explorer = EpsilonGreedyExplorer(0.1)
        learner = TDLearner(q_approx, :SARS)
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
        t = (next_state=3, reward=5.0, terminal=false)
        push!(trajectory, t)
        RLBase.optimise!(policy, PostActStage(), trajectory)
        # Add assertions here
    end
end
