@testset "wrappers" begin

    @testset "ActionTransformedEnv" begin
        env = TigerProblemEnv(; rng=StableRNG(123))
        env′ = ActionTransformedEnv(
            env;
            action_space_mapping=x -> Base.OneTo(3),
            action_mapping=i -> action_space(env)[i],
        )

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)
    end

    @testset "DefaultStateStyleEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng=rng)
        S = InternalState{Int}()
        env′ = DefaultStateStyleEnv{S}(env)
        @test DefaultStateStyle(env′) === S

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        # fix #129
        env = TicTacToeEnv()
        s_env = state(env)
        @test s_env ∈ state_space(env)

        E = DefaultStateStyleEnv{Observation{Int}()}(env)  # error probably somewhere here
        s = state(E)
        @test s isa Int
        @test s ∈ state_space(E)
    end

    @testset "MaxTimeoutEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng=rng)
        n = 100
        env′ = MaxTimeoutEnv(env, n)

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        while !is_terminated(env′)
            RLBase.act!(env′, :listen)
            n -= 1
            @test n >= 0
        end

        reset!(env′)
        @test env′.current_t == 1
        @test is_terminated(env′) == false
    end

    @testset "RewardTransformedEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng=rng)
        env′ = RewardTransformedEnv(env; reward_mapping = x -> sign(x))

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        while !is_terminated(env′)
            RLBase.act!(env′, rand(rng, legal_action_space(env′)))
            @test reward(env′) ∈ (-1, 0, 1)
        end
    end

    @testset "RewardOverriddenEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng=rng)
        env′ = RewardOverriddenEnv(env, e -> sign(reward(e)))

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        while !is_terminated(env′)
            RLBase.act!(env′, rand(rng, legal_action_space(env′)))
            @test reward(env′) ∈ (-1, 0, 1)
        end
    end

    @testset "StateCachedEnv" begin
        rng = StableRNG(123)
        env = CartPoleEnv(; rng=rng)
        env′ = StateCachedEnv(env)

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        while !is_terminated(env′)
            RLBase.act!(env′, rand(rng, legal_action_space(env′)))
            s1 = state(env)
            s2 = state(env)
            @test s1 === s2
        end
    end

    @testset "StateTransformedEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng=rng)
        # S = (:door1, :door2, :door3, :none)
        # env′ = StateTransformedEnv(env, state_mapping=s -> s+1)
        # RLBase.state_space(env::typeof(env′), ::RLBase.AbstractStateStyle, ::Any) = S

        # RLBase.test_interfaces!(env′)
        # RLBase.test_runnable!(env′)
    end

    @testset "StochasticEnv" begin
        env = KuhnPokerEnv()
        rng = StableRNG(123)
        env′ = StochasticEnv(env; rng=rng)

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)
    end

    @testset "RockPaperScissorsEnv" begin
        env =  RockPaperScissorsEnv()
        RLBase.test_interfaces!(env)
        RLBase.test_runnable!(env)

        @test RLBase.current_player(env) == SimultaneousPlayer()

        RLBase.act!(env, ['💎', '📃'])

        @test is_terminated(env) == true

        reset!(env)

        @test reward(env) == PlayerTuple(Player(1) => 0, Player(2) => 0)
        @test is_terminated(env) == false

        RLBase.act!(env, ['💎', '📃'])

        @test reward(env, Player(1)) == -1
        @test reward(env, Player(2)) == 1
        @test is_terminated(env) == true

        reset!(env)
        @test reward(env) == PlayerTuple(Player(1) => 0, Player(2) => 0)
        @test reward(env, Player(1)) == 0
        @test reward(env, Player(2)) == 0
        @test is_terminated(env) == false
    end
    
end

@testset "Wrapped Env run" begin
    Base.@kwdef mutable struct LotteryEnv <: AbstractEnv
        reward::Union{Nothing, Int} = nothing
    end
    
    struct LotteryAction{a}
        function LotteryAction(a)
            new{a}()
        end
    end

    RLBase.action_space(env::LotteryEnv) = LotteryAction.([:PowerRich, :MegaHaul, nothing])

    RLBase.reward(env::LotteryEnv) = env.reward
    RLBase.state(env::LotteryEnv, ::Observation, ::DefaultPlayer) = !isnothing(env.reward)
    RLBase.state_space(env::LotteryEnv) = [false, true]
    RLBase.is_terminated(env::LotteryEnv) = !isnothing(env.reward)
    RLBase.reset!(env::LotteryEnv) = env.reward = nothing

    function RLBase.act!(x::LotteryEnv, action)
        if action == LotteryAction(:PowerRich)
            x.reward = rand() < 0.01 ? 100_000_000 : -10
        elseif action == LotteryAction(:MegaHaul)
            x.reward = rand() < 0.05 ? 1_000_000 : -10
        elseif action == LotteryAction(nothing)
            x.reward = 0
        else
            @error "unknown action of $action"
        end
    end

    env = LotteryEnv()

    p = QBasedPolicy(
        learner = TDLearner(
            TabularQApproximator(
                n_state = length(state_space(env)),
                n_action = length(action_space(env)),
            ), :SARS
        ),
        explorer = EpsilonGreedyExplorer(0.1)
    )

    wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping=s -> s ? 1 : 2,
            state_space_mapping = _ -> Base.OneTo(2)
        );
        action_mapping = i -> action_space(env)[i],
        action_space_mapping = _ -> Base.OneTo(3),
    )
    @test plan!(p, wrapped_env) ∈ [1, 2, 3]

    h = TotalRewardPerEpisode()
    e = run(p, wrapped_env, StopAfterNEpisodes(1_000), h)
    @test h.reward ∈ [-10, 100_000_000, 1_000_000, 0]
end
