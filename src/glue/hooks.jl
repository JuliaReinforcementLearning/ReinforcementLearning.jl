abstract type AbstractStage end

struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end
struct PreActStage <: AbstractStage end
struct PostActStage <: AbstractStage end

const PRE_EPISODE_STAGE = PreEpisodeStage()
const POST_EPISODE_STAGE = PostEpisodeStage()
const PRE_ACT_STAGE = PreActStage()
const POST_ACT_STAGE = PostActStage()

abstract type AbstractHook end

(hook::AbstractHook)(::T, args; kw...) where T<:AbstractStage = nothing

# https://github.com/JuliaLang/julia/issues/14919
# function (f::Function)(stage::T, args...;kw...) where T<: AbstractStage end

#####
# EmptyHook
#####

struct EmptyHook <: AbstractHook end

const EMPTY_HOOK = EmptyHook()

#####
# StepsPerEpisode
#####

mutable struct StepsPerEpisode <: AbstractHook
    steps::Vector{Int}
    count::Int
    tag::String
end

function StepsPerEpisode(;steps=Int[], count=0, tag="TRAINING")
    StepsPerEpisode(steps, count, tag)
end

function (hook::StepsPerEpisode)(::PostActStage, agent, env, obs)
    hook.count += 1
end

function (hook::StepsPerEpisode)(::PostEpisodeStage, agent, env, obs)
    push!(hook.steps, hook.count)
    hook.count = 0
    @debug hook.tag STEPS_PER_EPISODE=hook.steps[end]
end

#####
# RewardsPerEpisode
#####

mutable struct RewardsPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}}
    tag::String
end

function RewardsPerEpisode(;rewards=Vector{Vector{Float64}}(), tag="TRAINING")
    RewardsPerEpisode(reward, tag)
end

function (hook::RewardsPerEpisode)(::PreEpisodeStage, agent, env, obs)
    push!(hook.rewards, [])
end

function (hook::RewardsPerEpisode)(::PostActStage, agent, env, obs)
    push!(hook.rewards[end], reward(obs))
end

function (hook::RewardsPerEpisode)(::PostEpisodeStage, agent, env, obs)
    @debug hook.tag REWARDS_PER_EPISODE=hook.rewards[end]
end

#####
# TotalRewardPerEpisode
#####

mutable struct TotalRewardPerEpisode <: AbstractHook
    rewards::Vector{Float64}
    reward::Float64
    tag::String
end

function TotalRewardPerEpisode(;rewards=Float64[], reward=0.0, tag="TRAINING")
    TotalRewardPerEpisode(rewards, reward, tag)
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env, obs)
    hook.reward += reward(obs)
end

function (hook::TotalRewardPerEpisode)(::PostEpisodeStage, agent, env, obs)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
    @debug hook.tag REWARD_PER_EPISODE=hook.rewards[end]
end