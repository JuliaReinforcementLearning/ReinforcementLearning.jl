# ---
# title: JuliaRL\_MADDPG\_KuhnPoker
# cover: assets/JuliaRL_MADDPG_KuhnPoker.png
# description: MADDPG applied to KuhnPoker
# date: 2021-08-28
# author: "[Peter Chen](https://github.com/peterchen96)" 
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using IntervalSets

mutable struct KuhnMADDPGHook <: AbstractHook
    eval_freq::Int
    episode_counter::Int
    episode::Vector{Int}
    results::Vector{Float64}
end

function (hook::KuhnMADDPGHook)(::PostEpisodeStage, policy, env)
    hook.episode_counter += 1
    if hook.episode_counter % hook.eval_freq == 0
        push!(hook.episode, hook.episode_counter)
        push!(hook.results, reward(env, 1))
    end
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MADDPG},
    ::Val{:KuhnPoker},
    ::Nothing;
    seed=123,
)
    rng = StableRNG(seed)
    env = KuhnPokerEnv()
    wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping = s -> [findfirst(==(s), state_space(env))],
            state_space_mapping = ss -> [[findfirst(==(s), state_space(env))] for s in state_space(env)]
            ),
        ## drop the dummy action of the other agent.
        action_mapping = x -> length(x) == 1 ? x : Int(ceil(x[current_player(env)]) + 1),
    )
    ns, na = 1, 1 # dimension of the state and action.
    n_players = 2 # number of players

    init = glorot_uniform(rng)

    create_actor() = Chain(
            Dense(ns, 64, relu; init = init),
            Dense(64, 64, relu; init = init),
            Dense(64, na, tanh; init = init),
        )

    create_critic() = Chain(
        Dense(n_players * ns + n_players * na, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
        )

    
    policy = DDPGPolicy(
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
        γ = 0.95f0,
        ρ = 0.99f0,
        na = na,
        start_steps = 1000,
        start_policy = RandomPolicy(-0.99..0.99; rng = rng),
        update_after = 1000,
        act_limit = 0.99,
        act_noise = 0.,
        rng = rng,
    )
    trajectory = CircularArraySARTTrajectory(
        capacity = 100_000, # replay buffer capacity
        state = Vector{Int} => (ns, ),
        action = Float32 => (na, ),
    )

    agents = MADDPGManager(
        Dict((player, Agent(
            policy = NamedPolicy(player, deepcopy(policy)),
            trajectory = deepcopy(trajectory),
        )) for player in players(env) if player != chance_player(env)),
        SARTS, # trace's type
        512, # batch_size
        100, # update_freq
        0, # initial update_step
        rng
    )

    stop_condition = StopAfterEpisode(100_000, is_show_progress=!haskey(ENV, "CI"))
    hook = KuhnMADDPGHook(1000, 0, [], [])
    Experiment(agents, wrapped_env, stop_condition, hook, "# play KuhnPoker with MADDPG")
end

#+ tangle=false
using Plots
ex = E`JuliaRL_MADDPG_KuhnPoker`
run(ex)
scatter(ex.hook.episode, ex.hook.results, xaxis=:log, xlabel="episode", ylabel="reward of player 1")

savefig("assets/JuliaRL_MADDPG_KuhnPoker.png") #hide

# ![](assets/JuliaRL_MADDPG_KuhnPoker.png)
