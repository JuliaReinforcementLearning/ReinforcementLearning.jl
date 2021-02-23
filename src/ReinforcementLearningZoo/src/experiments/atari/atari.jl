using .ArcadeLearningEnvironment
using .ReinforcementLearningEnvironments
using BSON
using Flux: Chain

function atari_env_factory(
    name,
    state_size,
    n_frames,
    max_episode_steps = 100_000;
    seed = nothing,
    repeat_action_probability = 0.25,
    n_replica = 1,
)
    init(seed) =
        AtariEnv(;
            name = string(name),
            grayscale_obs = true,
            noop_max = 30,
            frame_skip = 4,
            terminal_on_life_loss = false,
            repeat_action_probability = repeat_action_probability,
            max_num_frames_per_episode = n_frames * max_episode_steps,
            color_averaging = false,
            full_action_space = false,
            seed = seed,
        ) |>
        env ->
            StateOverriddenEnv(
                env,
                Chain(ResizeImage(state_size...), StackFrames(state_size..., n_frames)),
            ) |>
            StateCachedEnv |>
            env -> RewardOverriddenEnv(env, r -> clamp(r, -1, 1))

    if n_replica == 1
        init(seed)
    else
        envs = [init(hash(seed + i)) for i in 1:n_replica]
        states = Flux.batch(state.(envs))
        rewards = reward.(envs)
        terminals = is_terminated.(envs)
        A = Space([action_space(x) for x in envs])
        S = Space(fill(0..255, size(states)))
        MultiThreadEnv(envs, states, rewards, terminals, A, S, nothing)
    end
end

"Total reward per episode before reward reshaping"
Base.@kwdef mutable struct TotalOriginalRewardPerEpisode <: AbstractHook
    rewards::Vector{Float64} = Float64[]
    reward::Float64 = 0.0
end

function (hook::TotalOriginalRewardPerEpisode)(
    ::PostActStage,
    agent,
    env::RewardOverriddenEnv,
)
    hook.reward += reward(env.env)
end

function (hook::TotalOriginalRewardPerEpisode)(::PostEpisodeStage, agent, env)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
end

"Total reward of each inner env per episode before reward reshaping"
struct TotalBatchOriginalRewardPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}}
    reward::Vector{Float64}
end

function TotalBatchOriginalRewardPerEpisode(batch_size::Int)
    TotalBatchOriginalRewardPerEpisode([Float64[] for _ in 1:batch_size], zeros(batch_size))
end

function (hook::TotalBatchOriginalRewardPerEpisode)(
    ::PostActStage,
    agent,
    env::MultiThreadEnv{<:RewardOverriddenEnv},
)
    for (i, e) in enumerate(env.envs)
        hook.reward[i] += reward(e.env)
        if is_terminated(e)
            push!(hook.rewards[i], hook.reward[i])
            hook.reward[i] = 0.0
        end
    end
end


for f in readdir(@__DIR__)
    if f != splitdir(@__FILE__)[2]
        include(f)
    end
end
