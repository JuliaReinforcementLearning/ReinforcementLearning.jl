using Random

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:Minimax}, ::Val{:OpenSpiel}, game;)
    env = OpenSpielEnv(string(game))
    agents = (
        Agent(policy = MinimaxPolicy(), role = 0),
        Agent(policy = MinimaxPolicy(), role = 1),
    )
    hooks = (TotalRewardPerEpisode(), TotalRewardPerEpisode())
    description = """
      # Play `$game` in OpenSpiel with Minimax
      """
    Experiment(agents, env, StopAfterEpisode(1), hooks, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:TabularCFR},
    ::Val{:OpenSpiel},
    game;
    n_iter = 300,
    seed = 123,
)
    env = OpenSpielEnv(
        game;
        default_state_style = RLBase.Information{String}(),
        is_chance_agent_required = true,
    )
    rng = StableRNG(seed)
    π = TabularCFRPolicy(; rng = rng)

    description = """
      # Play `$game` in OpenSpiel with TabularCFRPolicy
      """
    Experiment(π, env, StopAfterStep(300), EmptyHook(), description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DeepCFR},
    ::Val{:OpenSpiel},
    game;
    n_iter = 300,
    seed = 123,
)
    @assert game == "leduc_poker"
    env = OpenSpielEnv(
        "leduc_poker";
        default_state_style = RLBase.Information{Array}(),
        is_chance_agent_required = true,
    )

    #= using CxxWrap =#

    p = DeepCFR(
        Π = NeuralNetworkApproximator(
            model = Chain(Dense(30, 64, relu), Dense(64, 64, relu), Dense(64, 3)) |> gpu,
            optimizer = ADAM(0.001),
        ),
        MΠ = ReservoirTrajectory(
            3_000_000,
            :I => Any, #= CxxWrap.StdLib.StdVectorAllocated{Float64}, =#
            :t => Int,
            :σ => Vector{Float32},
            :m => Vector{Bool},
        ),
        V = Dict(
            p => NeuralNetworkApproximator(
                model = Chain(Dense(30, 64, relu), Dense(64, 64, relu), Dense(64, 3)) |> gpu,
                optimizer = ADAM(0.001),
            ) for p in get_players(env) if p != chance_player(env)
        ),
        MV = Dict(
            p => ReservoirTrajectory(
                3_000_000,
                :I => Any,#=CxxWrap.StdLib.StdVectorAllocated{Float64}=#
                :t => Int,
                :r̃ => Vector{Float32},
                :m => Vector{Bool},
            ) for p in get_players(env) if p != chance_player(env)
        ),
        K = 1500,
        n_training_steps_V = 750,
        n_training_steps_Π = 2000,
        batch_size_V = 2048,
        batch_size_Π = 2048,
        initializer = glorot_normal(CUDA.CURAND.default_rng()),
    )
    # nash_conv ≈ 0.23
    Experiment(p, env, StopAfterStep(500), EmptyHook(), "# run DeepcCFR on leduc_poker")
end
