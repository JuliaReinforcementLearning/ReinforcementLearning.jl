# ---
# title: JuliaRL\_MADDPG\_MultiAgentUndirected
# cover: assets/JuliaRL_MADDPG_MultiAgentUndirected.png
# description: MADDPG applied to CollectGemsMultiAgentUndirected
# date: 2021-08-02
# author: "[Peter Chen](https://github.com/peterchen96)" 
# ---

#+ tangle=true
using ReinforcementLearning
using GridWorlds
using StableRNGs
using Flux
using IntervalSets

mutable struct StepRecorder <: AbstractHook
    step::Int
    recorder::Vector{Int}
end

function (hook::StepRecorder)(::PostActStage, policy, env)
    hook.step += 1
end

function (hook::StepRecorder)(::PostEpisodeStage, policy, env)
    push!(hook.recorder, hook.step)
    hook.step = 0
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MADDPG},
    ::Val{:MultiAgentUndirected},
    ::Nothing;
    seed=123,
)
    rng = StableRNG(seed)

    env = GridWorlds.CollectGemsMultiAgentUndirectedModule.CollectGemsMultiAgentUndirected(rng=rng)
    n_players = length(env.agent_positions)

    env = GridWorlds.RLBaseEnv(env)
    env = ActionTransformedEnv(
        StateTransformedEnv(env; state_mapping=x -> vec(Float32.(x))),
        action_mapping = x -> x + 2,
        )
    env = MaxTimeoutEnv(env, 500)

    ns, na = length(state(env)), 1
    init = glorot_uniform(rng)

    create_actor() = Chain(
            Dense(ns, 64, relu; init = init),
            Dense(64, 64, relu; init = init),
            Dense(64, na, tanh; init = init),
        )

    create_critic() = Chain(
        Dense(ns + n_players * na, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
        )

    agent = Agent(
        policy = MADDPGPolicy(
            Dict((player_idx, DDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            γ = 0.99f0,
            ρ = 0.995f0,
            na = 1,
            start_steps = 1000,
            start_policy = RandomPolicy(-1.9..1.9; rng = rng),
            update_after = 1000,
            update_every = 1,
            act_limit = 1.9,
            act_noise = 0.1,
            rng = rng,
        )) for player_idx in Base.OneTo(n_players)),
        64, # batch_size
        1000, # update_after
        1, # update_freq
        0, # step count
        rng),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10000,
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na * n_players, ),
        ),
    )

    stop_condition = StopAfterEpisode(100_000, is_show_progress=!haskey(ENV, "CI"))
    hook = StepRecorder(0, [])
    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
ENV["GKSwstype"]="nul"
using Plots
ex = E`JuliaRL_MADDPG_MultiAgentUndirected`
run(ex)
plot(ex.hook.recorder)
savefig("assets/JuliaRL_MADDPG_MultiAgentUndirected.png") #hide

# ![](assets/JuliaRL_BasicDQN_SingleRoomUndirected.png)
