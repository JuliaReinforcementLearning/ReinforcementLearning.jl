export AbstractHook,
    ComposedHook,
    EmptyHook,
    StepsPerEpisode,
    RewardsPerEpisode,
    TotalRewardPerEpisode,
    TotalBatchRewardPerEpisode,
    CumulativeReward,
    TimePerStep

"""
A hook is called at different stage duiring a [`run`](@ref) to allow users to inject customized runtime logic.
By default, a `AbstractHook` will do nothing. One can override the behavior by implementing the following methods:

- `(hook::YourHook)(::PreActStage, agent, env, obs, action)`, note that there's an extra argument of `action`.
- `(hook::YourHook)(::PostActStage, agent, env, obs)`
- `(hook::YourHook)(::PreEpisodeStage, agent, env, obs)`
- `(hook::YourHook)(::PostEpisodeStage, agent, env, obs)`
"""
abstract type AbstractHook end

(hook::AbstractHook)(args...) = nothing

# https://github.com/JuliaLang/julia/issues/14919
# function (f::Function)(stage::T, args...;kw...) where T<: AbstractStage end

#####
# ComposedHook
#####

"""
    ComposedHook(hooks::AbstractHook...)

Compose different hooks into a single hook.
"""
struct ComposedHook{T<:Tuple} <: AbstractHook
    hooks::T
    ComposedHook(hooks...) = new{typeof(hooks)}(hooks)
end

function (hook::ComposedHook)(stage::AbstractStage, args...; kw...)
    for h in hook.hooks
        h(stage, args...; kw...)
    end
end

Base.getindex(hook::ComposedHook, inds...) = getindex(hook.hooks, inds...)

#####
# EmptyHook
#####

"""
Do nothing
"""
struct EmptyHook <: AbstractHook end

const EMPTY_HOOK = EmptyHook()

#####
# StepsPerEpisode
#####

"""
    StepsPerEpisode(; steps = Int[], count = 0, tag = "TRAINING")

Store steps of each episode in the field of `steps`.
"""
Base.@kwdef mutable struct StepsPerEpisode <: AbstractHook
    steps::Vector{Int} = Int[]
    count::Int = 0
    tag::String = "TRAINING"
end

function (hook::StepsPerEpisode)(::PostActStage, args...)
    hook.count += 1
end

function (hook::StepsPerEpisode)(
    ::Union{PostEpisodeStage,PostExperimentStage},
    agent,
    env,
    obs,
)
    push!(hook.steps, hook.count)
    hook.count = 0
    @debug hook.tag STEPS_PER_EPISODE = hook.steps[end]
end

#####
# RewardsPerEpisode
#####

"""
    RewardsPerEpisode(; rewards = Vector{Vector{Float64}}(), tag = "TRAINING")

Store each reward of each step in every episode in the field of `rewards`.
"""
Base.@kwdef mutable struct RewardsPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    tag::String = "TRAINING"
end

function (hook::RewardsPerEpisode)(::PreEpisodeStage, agent, env, obs)
    push!(hook.rewards, [])
end

function (hook::RewardsPerEpisode)(::PostActStage, agent, env, obs)
    push!(hook.rewards[end], get_reward(obs))
end

function (hook::RewardsPerEpisode)(::PostEpisodeStage, agent, env, obs)
    @debug hook.tag REWARDS_PER_EPISODE = hook.rewards[end]
end

#####
# TotalRewardPerEpisode
#####

"""
    TotalRewardPerEpisode(; rewards = Float64[], reward = 0.0, tag = "TRAINING")

Store the total rewards of each episode in the field of `rewards`.
"""
Base.@kwdef mutable struct TotalRewardPerEpisode <: AbstractHook
    rewards::Vector{Float64} = Float64[]
    reward::Float64 = 0.0
    tag::String = "TRAINING"
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env, obs)
    hook.reward += get_reward(obs)
end

function (hook::TotalRewardPerEpisode)(
    ::Union{PostEpisodeStage,PostExperimentStage},
    agent,
    env,
    obs,
)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
    @debug hook.tag REWARD_PER_EPISODE = hook.rewards[end]
end

#####
# TotalBatchRewardPerEpisode 
#####
struct TotalBatchRewardPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}}
    reward::Vector{Float64}
    tag::String
end

"""
    TotalBatchRewardPerEpisode(batch_size::Int;tag="TRAINING")

Similar to [`TotalRewardPerEpisode`](@ref), but will record total rewards per episode in [`BatchObs`](@ref).
"""
function TotalBatchRewardPerEpisode(batch_size::Int;tag="TRAINING")
    TotalBatchRewardPerEpisode(
        [Float64[] for _ in 1:batch_size],
        zeros(batch_size),
        tag
    )
end

function (hook::TotalBatchRewardPerEpisode)(::PostActStage, agent, env, obs::BatchObs)
    for i in 1:length(obs)
        hook.reward[i] += get_reward(obs[i])
        if get_terminal(obs[i])
            push!(hook.rewards[i], hook.reward[i])
            hook.reward[i] = 0.
        end
    end
end

#####
# CumulativeReward
#####

"""
    CumulativeReward(rewards::Vector{Float64} = [0.0], tag::String = "TRAINING")

Store cumulative rewards since the beginning to the field of `rewards`.
"""
Base.@kwdef struct CumulativeReward <: AbstractHook
    rewards::Vector{Float64} = [0.0]
    tag::String = "TRAINING"
end

function (hook::CumulativeReward)(::PostActStage, agent, env, obs)
    push!(hook.rewards, get_reward(obs) + hook.rewards[end])
    @debug hook.tag CUMULATIVE_REWARD = hook.rewards[end]
end

#####
# TimePerStep
#####

"""
    TimePerStep(;max_steps=100)
    TimePerStep(times::CircularArrayBuffer{Float64}, t::UInt64)

Store time cost of the latest `max_steps` in the `times` field.
"""
mutable struct TimePerStep <: AbstractHook
    times::CircularArrayBuffer{Float64,1}
    t::UInt64
end

TimePerStep(; max_steps = 100) =
    TimePerStep(CircularArrayBuffer{Float64}(max_steps), time_ns())

function (hook::TimePerStep)(::PostActStage, agent, env, obs)
    push!(hook.times, (time_ns() - hook.t) / 1e9)
    hook.t = time_ns()
end
