function test_independent_learner_env(e::AbstractEnv, π::AbstractPolicy, cap::Integer, nᵢ::Integer, state_shape)
    s = [state(e, player) for player ∈ players(e)]
    r = []
    t = []
    a = []

    n = 1
    loc_actions = []
    while n <= nᵢ*length(players(e))
        if is_terminated(e)
            reset!(e)
        end

        π(PreActStage(), e)

        action = π(e)

        e(action)
        push!(loc_actions, action)

        if n % length(players(e)) == 1 && n > 1
          r = isempty(r) ? [reward(e, player) for player ∈ players(e)] : hcat(r, [reward(e, player) for player ∈ players(e)])
          if n < (nᵢ-1)*length(players(e))
            s = hcat(s, [state(e, player) for player ∈ players(e)])
          end 
        end

        if n % length(players(e)) == 0 && n < nᵢ*length(players(e))
            a = isempty(a) ? loc_actions : hcat(a, loc_actions)
            t = isempty(t) ? [is_terminated(e) for _ ∈ players(e)] : hcat(t, [is_terminated(e) for _ ∈ players(e)])
            loc_actions = []
        end


        if n % length(players(e)) == 0 && n > length(players(e))
            for (i, p) ∈ enumerate(players(e))
                # states are images with dim 457 x 120 and 3 color channels
                for s ∈ s[i,:] @test size(s) == state_shape end
                @test min(cap, n ÷ length(players(e)) - 1) == length(π.agents[p].trajectory.container)
                @test s[i, max(1, n ÷ length(players(e)) - cap):(n == nᵢ * length(players(e)) ? end : end-1)] == π.agents[p].trajectory.container[:state]
                @test r[i, max(1, n ÷ length(players(e)) - cap): end] ≈ π.agents[p].trajectory.container[:reward]
                @test t[i, max(1, n ÷ length(players(e)) - cap):(n == nᵢ * length(players(e)) ? end : end-1)] ≈ π.agents[p].trajectory.container[:terminal]
                @test a[i, max(1, n ÷ length(players(e)) - cap):(n == nᵢ * length(players(e)) ? end : end-1)] == π.agents[p].trajectory.container[:action]

            end
        end


        # no need to call optimise!(m)

        π(PostActStage(), e)
        n += 1
    end
    for (i, p) ∈ enumerate(players(e)) @test r[i, max(1, n ÷ length(players(e)) - cap):end] ≈ π.agents[p].trajectory.container[:reward] end
    for (i, p) ∈ enumerate(players(e)) @test s[i, max(1, n ÷ length(players(e)) - cap):end] == π.agents[p].trajectory.container[:state] end
    for (i, p) ∈ enumerate(players(e)) @test t[i, max(1, n ÷ length(players(e)) - cap):end] == π.agents[p].trajectory.container[:terminal] end
    for (i, p) ∈ enumerate(players(e)) @test a[i, max(1, n ÷ length(players(e)) - cap):end] == π.agents[p].trajectory.container[:action] end
end

function test_independent_learner_mpe_env(cap::Integer, nᵢ::Integer, seed::Integer=3141)
    e = PettingZooEnv("mpe.simple_spread_v2"; seed=seed)
    m = MultiAgentManager(Dict(player =>
                    Agent(RandomPolicy(action_space(e, player)),
                        Trajectory(
                            container=CircularArraySARTTraces(
                              capacity=cap,
                              state=Float32 => (length(state_space(e, player).domains),),
                            ),
                            sampler=NStepBatchSampler{SS′ART}(
                                n=1,
                                γ=1,
                                batch_size=1,
                                rng=StableRNG(1)
                            ),
                            controller=InsertSampleRatioController(
                                threshold=1,
                                n_inserted=0
                            ))
                    )
                    for player in players(e)),
                    current_player(e));
    test_independent_learner_env(e, m, cap, nᵢ, (18,))
end

function test_independent_learner_pistonball_env(cap::Integer, nᵢ::Integer, seed::Integer=3141)
    e = PettingZooEnv("butterfly.pistonball_v6"; seed=seed);
    m = MultiAgentManager(Dict(player =>
                    Agent(RandomPolicy(action_space(e, player)),
                        Trajectory(
                            container=CircularArraySARTTraces(
                              capacity=cap,
                              state=UInt8 => (457, 120, 3,),
                              action=Vector{Float32} => ()
                            ),
                            sampler=NStepBatchSampler{SS′ART}(
                                n=1,
                                γ=1,
                                batch_size=1,
                                rng=StableRNG(1)
                            ),
                            controller=InsertSampleRatioController(
                                threshold=1,
                                n_inserted=0
                            ))
                    )
                    for player in players(e)),
                    current_player(e));
    test_independent_learner_env(e, m, cap, nᵢ, (457,120,3,))
end

@testset "IndependentLearningAgents" begin
    seeds=[3141, 2718]
    @testset "Pettingzoo_MPE_independent" begin
        for seed ∈ seeds
            test_independent_learner_mpe_env(2, 100, seed)
            test_independent_learner_mpe_env(40, 100, seed)
            test_independent_learner_mpe_env(40, 25, seed)
        end
    end
    @testset "Pettingzoo_Butterfly_pistonball_independent" begin
        for seed ∈ seeds
            test_independent_learner_pistonball_env(400, 20, seed)
            test_independent_learner_pistonball_env(40, 200, seed)
        end
        
    end
end