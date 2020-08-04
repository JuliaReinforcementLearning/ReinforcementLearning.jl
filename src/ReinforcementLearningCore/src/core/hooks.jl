export AbstractHook,
    ComposedHook,
    EmptyHook,
    StepsPerEpisode,
    RewardsPerEpisode,
    TotalRewardPerEpisode,
    TotalBatchRewardPerEpisode,
    BatchStepsPerEpisode,
    CumulativeReward,
    TimePerStep,
    DoEveryNEpisode,
    DoEveryNStep

"""
A hook is called at different stage duiring a [`run`](@ref) to allow users to inject customized runtime logic.
By default, a `AbstractHook` will do nothing. One can override the behavior by implementing the following methods:

- `(hook::YourHook)(::PreActStage, agent, env, action)`, note that there's an extra argument of `action`.
- `(hook::YourHook)(::PostActStage, agent, env)`
- `(hook::YourHook)(::PreEpisodeStage, agent, env)`
- `(hook::YourHook)(::PostEpisodeStage, agent, env)`
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
# display
#####

Base.display(::AbstractStage, agent, env, args...; kwargs...) = display(env)

#####
# StepsPerEpisode
#####

"""
    StepsPerEpisode(; steps = Int[], count = 0)

Store steps of each episode in the field of `steps`.
"""
Base.@kwdef mutable struct StepsPerEpisode <: AbstractHook
    steps::Vector{Int} = Int[]
    count::Int = 0
end

(hook::StepsPerEpisode)(::PostActStage, args...) = hook.count += 1

function (hook::StepsPerEpisode)(::Union{PostEpisodeStage,PostExperimentStage}, agent, env)
    push!(hook.steps, hook.count)
    hook.count = 0
end

#####
# RewardsPerEpisode
#####

"""
    RewardsPerEpisode(; rewards = Vector{Vector{Float64}}())

Store each reward of each step in every episode in the field of `rewards`.
"""
Base.@kwdef mutable struct RewardsPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
end

function (hook::RewardsPerEpisode)(::PreEpisodeStage, agent, env)
    push!(hook.rewards, [])
end

function (hook::RewardsPerEpisode)(::PostActStage, agent, env)
    push!(hook.rewards[end], get_reward(env))
end

function (hook::RewardsPerEpisode)(::PostActStage, agent, env::RewardOverriddenEnv)
    push!(hook.rewards[end], get_reward(env.env))
end

#####
# TotalRewardPerEpisode
#####

"""
    TotalRewardPerEpisode(; rewards = Float64[], reward = 0.0)

Store the total rewards of each episode in the field of `rewards`.

!!! note
    If the environment is a [`RewardOverriddenenv`](@ref), then the original reward is recorded.
"""
Base.@kwdef mutable struct TotalRewardPerEpisode <: AbstractHook
    rewards::Vector{Float64} = Float64[]
    reward::Float64 = 0.0
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env)
    hook.reward += get_reward(env)
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env::RewardOverriddenEnv)
    hook.reward += get_reward(env.env)
end

function (hook::TotalRewardPerEpisode)(
    ::Union{PostEpisodeStage,PostExperimentStage},
    agent,
    env,
)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
end

#####
# TotalBatchRewardPerEpisode 
#####
struct TotalBatchRewardPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}}
    reward::Vector{Float64}
end

"""
    TotalBatchRewardPerEpisode(batch_size::Int)

Similar to [`TotalRewardPerEpisode`](@ref), but will record total rewards per episode in [`MultiThreadEnv`](@ref).

!!! note
    If the environment is a [`RewardOverriddenEnv`](@ref), then the original reward is recorded.
"""
function TotalBatchRewardPerEpisode(batch_size::Int)
    TotalBatchRewardPerEpisode([Float64[] for _ in 1:batch_size], zeros(batch_size))
end

function (hook::TotalBatchRewardPerEpisode)(
    ::PostActStage,
    agent,
    env::MultiThreadEnv{T},
) where {T}
    for i in 1:length(env)
        if T <: RewardOverriddenEnv
            hook.reward[i] += get_reward(env[i].env)
        else
            hook.reward[i] += get_reward(env[i])
        end
        if get_terminal(env[i])
            push!(hook.rewards[i], hook.reward[i])
            hook.reward[i] = 0.0
        end
    end
end

struct BatchStepsPerEpisode <: AbstractHook
    steps::Vector{Vector{Int}}
    step::Vector{Int}
end

"""
    BatchStepsPerEpisode(batch_size::Int; tag = "TRAINING")

Similar to [`StepsPerEpisode`](@ref), but only work for [`MultiThreadEnv`](@ref)
"""
function BatchStepsPerEpisode(batch_size::Int)
    BatchStepsPerEpisode([Int[] for _ in 1:batch_size], zeros(Int, batch_size))
end

function (hook::BatchStepsPerEpisode)(::PostActStage, agent, env::MultiThreadEnv)
    for i in 1:length(env)
        hook.step[i] += 1
        if get_terminal(env[i])
            push!(hook.steps[i], hook.step[i])
            hook.step[i] = 0
        end
    end
end

#####
# CumulativeReward
#####

"""
    CumulativeReward(rewards::Vector{Float64} = [0.0])

Store cumulative rewards since the beginning to the field of `rewards`.

!!! note
    If the environment is a [`RewardOverriddenEnv`](@ref), then the original reward is recorded instead.
"""
Base.@kwdef struct CumulativeReward <: AbstractHook
    rewards::Vector{Float64} = [0.0]
end

function (hook::CumulativeReward)(::PostActStage, agent, env::T) where {T}
    if T <: RewardOverriddenEnv
        r = get_reward(env.env)
    else
        r = get_reward(env)
    end
    push!(hook.rewards, r + hook.rewards[end])
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

function (hook::TimePerStep)(::PostActStage, agent, env)
    push!(hook.times, (time_ns() - hook.t) / 1e9)
    hook.t = time_ns()
end

"""
    DoEveryNStep(f; n=1, t=0)

Execute `f(agent, env)` every `n` step.
`t` is a counter of steps.
"""
Base.@kwdef mutable struct DoEveryNStep{F} <: AbstractHook
    f::F
    n::Int = 1
    t::Int = 0
end

DoEveryNStep(f, n = 1, t = 0) = DoEveryNStep(f, n, t)

function (hook::DoEveryNStep)(::PostActStage, agent, env)
    hook.t += 1
    if hook.t % hook.n == 0
        hook.f(hook.t, agent, env)
    end
end

"""
    DoEveryNEpisode(f; n=1, t=0)

Execute `f(agent, env)` every `n` episode.
`t` is a counter of steps.
"""
Base.@kwdef mutable struct DoEveryNEpisode{F} <: AbstractHook
    f::F
    n::Int = 1
    t::Int = 0
end

DoEveryNEpisode(f, n = 1, t = 0) = DoEveryNEpisode(f, n, t)

function (hook::DoEveryNEpisode)(::PostEpisodeStage, agent, env)
    hook.t += 1
    if hook.t % hook.n == 0
        hook.f(hook.t, agent, env)
    end
end
