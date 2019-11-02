export AbstractStage,
       PreEpisodeStage,
       PRE_EPISODE_STAGE,
       PostEpisodeStage,
       POST_EPISODE_STAGE,
       PreActStage,
       PRE_ACT_STAGE,
       PostActStage,
       POST_ACT_STAGE,
       AbstractHook,
       ComposedHook,
       EmptyHook,
       StepsPerEpisode,
       RewardsPerEpisode,
       TotalRewardPerEpisode,
       CumulativeReward,
       TimePerStep

abstract type AbstractStage end

struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end
struct PreActStage <: AbstractStage end
struct PostActStage <: AbstractStage end

const PRE_EPISODE_STAGE = PreEpisodeStage()
const POST_EPISODE_STAGE = PostEpisodeStage()
const PRE_ACT_STAGE = PreActStage()
const POST_ACT_STAGE = PostActStage()

"""
A hook is called at different stage duiring a [`run`](@ref). One can inject customized runtime logic in it.
"""
abstract type AbstractHook end

(hook::AbstractHook)(::T, agent, env, obs) where {T<:AbstractStage} = nothing

# https://github.com/JuliaLang/julia/issues/14919
# function (f::Function)(stage::T, args...;kw...) where T<: AbstractStage end

#####
# ComposedHook
#####

"""
    ComposedHook(hooks...)

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
mutable struct StepsPerEpisode <: AbstractHook
    steps::Vector{Int}
    count::Int
    tag::String
end

function StepsPerEpisode(; steps = Int[], count = 0, tag = "TRAINING")
    StepsPerEpisode(steps, count, tag)
end

function (hook::StepsPerEpisode)(::PostActStage, agent, env, action_obs)
    hook.count += 1
end

function (hook::StepsPerEpisode)(::PostEpisodeStage, agent, env, obs)
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
mutable struct RewardsPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}}
    tag::String
end

function RewardsPerEpisode(; rewards = Vector{Vector{Float64}}(), tag = "TRAINING")
    RewardsPerEpisode(rewards, tag)
end

function (hook::RewardsPerEpisode)(::PreEpisodeStage, agent, env, obs)
    push!(hook.rewards, [])
end

function (hook::RewardsPerEpisode)(::PostActStage, agent, env, action_obs)
    action, obs = action_obs
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
mutable struct TotalRewardPerEpisode <: AbstractHook
    rewards::Vector{Float64}
    reward::Float64
    tag::String
end

function TotalRewardPerEpisode(; rewards = Float64[], reward = 0.0, tag = "TRAINING")
    TotalRewardPerEpisode(rewards, reward, tag)
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env, action_obs)
    action, obs = action_obs
    hook.reward += get_reward(obs)
end

function (hook::TotalRewardPerEpisode)(::PostEpisodeStage, agent, env, obs)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
    @debug hook.tag REWARD_PER_EPISODE = hook.rewards[end]
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

function (hook::CumulativeReward)(::PostActStage, agent, env, action_obs)
    action, obs = action_obs
    push!(hook.rewards, get_reward(obs) + hook.rewards[end])
    @debug hook.tag CUMULATIVE_REWARD = hook.rewards[end]
end

#####
# TimePerStep
#####

"""
    TimePerStep(;max_steps=100)
    TimePerStep(times::CircularArrayBuffer{Float64, 1}, t::UInt64)

Store time cost of the latest `max_steps` in the `times` field.
"""
mutable struct TimePerStep <: AbstractHook
    times::CircularArrayBuffer{Float64, 1}
    t::UInt64
end

TimePerStep(;max_steps=100) = TimePerStep(CircularArrayBuffer{Float64}(max_steps), time_ns())

function (hook::TimePerStep)(::PostActStage, agent, env, obs_action)
    push!(hook.times, (time_ns() - hook.t)/1e9)
    hook.t = time_ns()
end