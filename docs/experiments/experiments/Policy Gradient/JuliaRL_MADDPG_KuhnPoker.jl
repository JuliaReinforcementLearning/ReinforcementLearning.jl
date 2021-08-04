# ---
# title: JuliaRL\_MADDPG\_KuhnPoker
# cover: assets/JuliaRL_MADDPG_KuhnPoker.png
# description: MADDPG applied to KuhnPoker
# date: 2021-08-05
# author: "[Peter Chen](https://github.com/peterchen96)" 
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using IntervalSets

mutable struct ResultNEpisode <: AbstractHook
episode_counter::Int
eval_every::Int
episode::Vector{Int}
results::Vector{Float64}
end

function (hook::ResultNEpisode)(::PostEpisodeStage, policy, env)
hook.episode_counter += 1
if hook.episode_counter % hook.eval_every == 0
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
    ## Encode the KuhnPokerEnv's states for training.
    env = KuhnPokerEnv()
    wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping = s -> [findfirst(==(s), state_space(env)) / length(state_space(env))], # for normalization
            state_space_mapping = ss -> [[findfirst(==(s), state_space(env)) / length(state_space(env))] for s in state_space(env)]
            ),
        action_mapping = x -> current_player(env) == chance_player(env) ? x : Int(x[current_player(env)] + 1),
    )
    ns, na = 1, 1
    n_players = 2

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
            γ = 0.99f0,
            ρ = 0.995f0,
            na = na,
            start_steps = 1000,
            start_policy = RandomPolicy(-0.9..0.9; rng = rng),
            act_limit = 0.9,
            act_noise = 0.1,
            rng = rng,
        ),
        trajectory = CircularArrayTrajectory(;
            ## since the policy update process should use next_state, here the capacity need to plus one
            capacity = 10001, # replay buffer capacity + 1
            state = Vector{Float32} => (ns,),
        )
    )

    agents = Agent(
        policy = MADDPGManager(
            Dict((player_idx, deepcopy(agent)) for player_idx in Base.OneTo(n_players)),
            128, # batch_size
            1000, # update_after
            50, # update_freq
            0, # step_counter
            rng
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10000, # replay buffer capacity
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na * n_players, ),
            reward = Vector{Int} => (n_players, ),
        ),
    )

    stop_condition = StopAfterEpisode(1_000_000, is_show_progress=!haskey(ENV, "CI"))
    hook = ResultNEpisode(0, 1000, [], [])
    Experiment(agents, wrapped_env, stop_condition, hook, "# run MADDPG on KuhnPokerEnv")
end

#+ tangle=false
using Plots
ex = E`JuliaRL_MADDPG_KuhnPoker`
run(ex)
scatter(ex.hook.episode, ex.hook.results, xaxis=:log, xlabel="episode", ylabel="reward of player 1")

savefig("assets/JuliaRL_MADDPG_KuhnPoker.png") #hide

# ![](assets/JuliaRL_BasicDQN_SingleRoomUndirected.png)