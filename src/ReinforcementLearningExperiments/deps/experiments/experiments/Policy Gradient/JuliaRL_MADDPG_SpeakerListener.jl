# ---
# title: JuliaRL\_MADDPG\_SpeakerListener
# cover: assets/JuliaRL_MADDPG_SpeakerListenerEnv.png
# description: MADDPG applied to SpeakerListenerEnv
# date: 2021-08-28
# author: "[Peter Chen](https://github.com/peterchen96)" 
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Statistics
using Flux

mutable struct MeanRewardHook <: AbstractHook
    episode::Int
    eval_rate::Int
    eval_episode::Int
    episodes::Vector
    mean_rewards::Vector
end

function (hook::MeanRewardHook)(::PostEpisodeStage, policy, env)
    if hook.episode % hook.eval_rate == 0
        ## evaluate policy's performance
        rew = 0
        for _ in 1:hook.eval_episode
            reset!(env)
            while !is_terminated(env)
                env |> policy |> env
                rew += reward(env)
            end
        end

        push!(hook.episodes, hook.episode)
        push!(hook.mean_rewards, rew / hook.eval_episode)
    end
    hook.episode += 1
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MADDPG},
    ::Val{:SpeakerListener},
    ::Nothing;
    seed=123,
)
    rng = StableRNG(seed)
    env = SpeakerListenerEnv(max_steps = 25)

    init = glorot_uniform(rng)
    critic_dim = sum(length(state(env, p)) + length(action_space(env, p)) for p in (:Speaker, :Listener))

    create_actor(player) = Chain(
        Dense(length(state(env, player)), 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, length(action_space(env, player)); init = init)
        )
    create_critic(critic_dim) = Chain(
        Dense(critic_dim, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
        )
    create_policy(player) = DDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(player),
                optimizer = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-2)),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(critic_dim),
                optimizer = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-2)),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(player),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(critic_dim),
            ),
            γ = 0.95f0,
            ρ = 0.99f0,
            na = length(action_space(env, player)),
            start_steps = 0,
            start_policy = nothing,
            update_after = 512 * env.max_steps, # batch_size * env.max_steps
            act_limit = 1.0,
            act_noise = 0.,
        )
    create_trajectory(player) = CircularArraySARTTrajectory(
            capacity = 1_000_000, # replay buffer capacity
            state = Vector{Float64} => (length(state(env, player)), ),
            action = Vector{Float64} => (length(action_space(env, player)), ),
        )

    agents = MADDPGManager(
        Dict(
            player => Agent(
                policy = NamedPolicy(player, create_policy(player)),
                trajectory = create_trajectory(player),
            ) for player in (:Speaker, :Listener)
        ),
        SARTS, # trace's type
        512, # batch_size
        100, # update_freq
        0, # initial update_step
        rng
    )

    stop_condition = StopAfterEpisode(8_000, is_show_progress=!haskey(ENV, "CI"))
    hook = MeanRewardHook(0, 800, 100, [], [])
    Experiment(agents, env, stop_condition, hook, "# play SpeakerListener with MADDPG")
end

#+ tangle=false
using Plots
ex = E`JuliaRL_MADDPG_SpeakerListener`
run(ex)
plot(ex.hook.episodes, ex.hook.mean_rewards, xlabel="episode", ylabel="mean episode reward")

savefig("assets/JuliaRL_MADDPG_SpeakerListenerEnv.png") #hide

# ![](assets/JuliaRL_MADDPG_SpeakerListenerEnv.png)
