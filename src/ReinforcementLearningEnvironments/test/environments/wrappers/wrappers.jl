@testset "wrappers" begin

    @testset "ActionTransformedEnv" begin
        env = TigerProblemEnv(; rng = StableRNG(123))
        env′ = ActionTransformedEnv(
            env;
            action_space_mapping = x -> Base.OneTo(3),
            action_mapping = i -> action_space(env)[i],
        )

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)
    end

    @testset "DefaultStateStyleEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng = rng)
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
        env = TigerProblemEnv(; rng = rng)
        n = 100
        env′ = MaxTimeoutEnv(env, n)

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        while !is_terminated(env′)
            env′(:listen)
            n -= 1
            @test n >= 0
        end

        reset!(env′)
        @test env′.current_t == 1
        @test is_terminated(env′) == false
    end

    @testset "RewardOverriddenEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng = rng)
        env′ = RewardOverriddenEnv(env, x -> sign(x))

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        while !is_terminated(env′)
            env′(rand(rng, legal_action_space(env′)))
            @test reward(env′) ∈ (-1, 0, 1)
        end
    end

    @testset "StateCachedEnv" begin
        rng = StableRNG(123)
        env = CartPoleEnv(; rng = rng)
        env′ = StateCachedEnv(env)

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)

        while !is_terminated(env′)
            env′(rand(rng, legal_action_space(env′)))
            s1 = state(env)
            s2 = state(env)
            @test s1 === s2
        end
    end

    @testset "StateTransformedEnv" begin
        rng = StableRNG(123)
        env = TigerProblemEnv(; rng = rng)
        # S = (:door1, :door2, :door3, :none)
        # env′ = StateTransformedEnv(env, state_mapping=s -> s+1)
        # RLBase.state_space(env::typeof(env′), ::RLBase.AbstractStateStyle, ::Any) = S

        # RLBase.test_interfaces!(env′)
        # RLBase.test_runnable!(env′)
    end

    @testset "StochasticEnv" begin
        env = KuhnPokerEnv()
        rng = StableRNG(123)
        env′ = StochasticEnv(env; rng = rng)

        RLBase.test_interfaces!(env′)
        RLBase.test_runnable!(env′)
    end

end
