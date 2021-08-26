# ---
# title: JuliaRL\_MADDPG\_SpeakerListener
# cover: assets/JuliaRL_MADDPG_SpeakerListenerEnv.png
# description: MADDPG applied to SpeakerListenerEnv
# date: 2021-08-24
# author: "[Peter Chen](https://github.com/peterchen96)" 
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Statistics
using Flux

mutable struct MeanRewardNEpisode <: AbstractHook
    step_reward
    eval_freq::Int
    record_episodes::Int
    episode_counter::Int
    episode::Vector{Int}
    rewards::Vector # sum of step reward
    reward_recorder::Vector
end

function (hook::MeanRewardNEpisode)(::PostActStage, policy, env)
    hook.step_reward += reward(env)
end

function (hook::MeanRewardNEpisode)(::PostEpisodeStage, policy, env)
    hook.episode_counter += 1
    push!(hook.reward_recorder, hook.step_reward)
    hook.step_reward = 0
    if length(hook.reward_recorder) > hook.record_episodes
        popfirst!(hook.reward_recorder)
    end

    if hook.episode_counter % hook.eval_freq == 0
        push!(hook.episode, hook.episode_counter)
        push!(hook.rewards, mean(hook.reward_recorder))
    end
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MADDPG},
    ::Val{:SpeakerListener},
    ::Nothing;
    seed=123,
)
    rng = StableRNG(seed)
    env = SpeakerListenerEnv(
        max_steps = 25,
    )

    init = glorot_uniform(rng)

    ns = Dict(
        player => length(state(env, player)) for player in (:Speaker, :Listener)
    )
    na = Dict(
        player => length(rand(action_space(env, player))) for player in (:Speaker, :Listener)
    )
    critic_dim = sum(ns[p] for p in (:Speaker, :Listener)) + sum(na[p] for p in (:Speaker, :Listener))
    create_actor(player) = Chain(
            Dense(ns[player], 64, relu; init = init),
            Dense(64, 64, relu; init = init),
            Dense(64, na[player]; init = init),
            softmax
            )
    create_critic(critic_dim) = Chain(
        Dense(critic_dim, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
        )
    create_policy(env, player) = DDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(player),
                optimizer = Flux.Optimise.Optimiser(ClipValue(0.5), ADAM(1e-2)),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(critic_dim),
                optimizer = Flux.Optimise.Optimiser(ClipValue(0.5), ADAM(1e-2)),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(player),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(critic_dim),
            ),
            γ = 0.95f0,
            ρ = 0.99f0,
            na = na[player],
            start_steps = 1000,
            start_policy = RandomPolicy(action_space(env, player); rng = rng),
            update_after = 1024 * env.max_steps, # batch_size * env.max_steps
            act_limit = 10.0,
            act_noise = 0.1,
            rng = rng,
        )
    create_trajectory(player) = CircularArraySARTTrajectory(
            capacity = 1_000_000, # replay buffer capacity
            state = Vector{Float64} => (ns[player], ),
            action = Vector{Float64} => (na[player], ),
        )

    agents = MADDPGManager(
        Dict(
            player => Agent(
                policy = NamedPolicy(player, create_policy(env, player)),
                trajectory = create_trajectory(player),
            ) for player in (:Speaker, :Listener)
        ),
        SARTS, # traces
        1024, # batch_size
        100, # update_freq
        0, # initial update_step
        rng
    )

    stop_condition = StopAfterEpisode(25_000, is_show_progress=!haskey(ENV, "CI"))
    hook = MeanRewardNEpisode(0, 1000, 100, 0, [], [], [])
    Experiment(agents, env, stop_condition, hook, "# run MADDPG on SpeakerListenerEnv")
end

#+ tangle=false
using Plots
ex = E`JuliaRL_MADDPG_SpeakerListener`
run(ex)
plot(ex.hook.episode, ex.hook.rewards, xlabel="episode", ylabel="mean reward")

savefig("assets/JuliaRL_MADDPG_SpeakerListenerEnv.png") #hide

# ![](assets/JuliaRL_MADDPG_SpeakerListenerEnv.png)
