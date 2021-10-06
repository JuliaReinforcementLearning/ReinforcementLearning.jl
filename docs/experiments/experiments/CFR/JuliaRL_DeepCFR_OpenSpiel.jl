# ---
# title: JuliaRL\_DeepCFR\_OpenSpiel(leduc_poker)
# cover: assets/logo.svg
# description: DeepCFR applied to OpenSpiel(leduc_poker)
# date: 2021-05-22
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=false
using ReinforcementLearning
using OpenSpiel

function RL.Experiment(
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
            ) for p in players(env) if p != chance_player(env)
        ),
        MV = Dict(
            p => ReservoirTrajectory(
                3_000_000,
                :I => Any,#=CxxWrap.StdLib.StdVectorAllocated{Float64}=#
                :t => Int,
                :r̃ => Vector{Float32},
                :m => Vector{Bool},
            ) for p in players(env) if p != chance_player(env)
        ),
        K = 1500,
        n_training_steps_V = 750,
        n_training_steps_Π = 2000,
        batch_size_V = 2048,
        batch_size_Π = 2048,
        initializer = glorot_normal(CUDA.CURAND.default_rng()),
    )
    Experiment(
        p,
        env,
        StopAfterStep(500, is_show_progress = !haskey(ENV, "CI")),
        EmptyHook(),
        "# run DeepcCFR on leduc_poker",
    )
end
