export AbstractHook,
    EmptyHook,
    StepsPerEpisode,
    RewardsPerEpisode,
    TotalRewardPerEpisode,
    TotalBatchRewardPerEpisode,
    BatchStepsPerEpisode,
    TimePerStep,
    DoEveryNEpisode,
    DoEveryNStep,
    DoOnExit

using UnicodePlots: lineplot, lineplot!
using Statistics: mean, std
using CircularArrayBuffers: CircularVectorBuffer

"""
A hook is called at different stage duiring a [`run`](@ref) to allow users to inject customized runtime logic.
By default, an `AbstractHook` will do nothing. One can customize the behavior by implementing the following methods:

- `(hook::YourHook)(::PreActStage, agent, env)`
- `(hook::YourHook)(::PostActStage, agent, env)`
- `(hook::YourHook)(::PreEpisodeStage, agent, env)`
- `(hook::YourHook)(::PostEpisodeStage, agent, env)`
- `(hook::YourHook)(::PostExperimentStage, agent, env)`

By convention, the `Base.getindex(h::YourHook)` is implemented to extract the metrics we are interested in.
Users can compose different `AbstractHook`s with `+`.
"""
abstract type AbstractHook end

(hook::AbstractHook)(args...) = nothing

struct ComposedHook{H} <: AbstractHook
    hooks::H
end

Base.:(+)(h1::AbstractHook, h2::AbstractHook) = ComposedHook((h1, h2))
Base.:(+)(h1::ComposedHook, h2::AbstractHook) = ComposedHook((h1.hooks..., h2))
Base.:(+)(h1::AbstractHook, h2::ComposedHook) = ComposedHook((h1, h2.hooks...))
Base.:(+)(h1::ComposedHook, h2::ComposedHook) = ComposedHook((h1.hooks..., h2.hooks...))

(h::ComposedHook)(args...) = map(h -> h(args...), h.hooks)

#####
# EmptyHook
#####

"""
Nothing but a placeholder.
"""
struct EmptyHook <: AbstractHook end

const EMPTY_HOOK = EmptyHook()

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

Base.getindex(h::StepsPerEpisode) = h.steps

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
struct RewardsPerEpisode{T} <: AbstractHook where {T<:Number}
    rewards::Vector{Vector{T}}
    empty_vect::Vector{T}

    function RewardsPerEpisode{T}() where {T<:Number}
        new{T}(Vector{Vector{T}}(), Vector{T}())
    end

    function RewardsPerEpisode()
        new{Float64}(Vector{Vector{Float64}}(), Vector{Float64}())
    end
end

Base.getindex(h::RewardsPerEpisode) = h.rewards

(h::RewardsPerEpisode)(::PreEpisodeStage, agent, env) = push!(h.rewards, h.empty_vect)
(h::RewardsPerEpisode)(::PostActStage, agent, env) = push!(h.rewards[end], reward(env))

#####
# TotalRewardPerEpisode
#####

"""
    TotalRewardPerEpisode(; rewards = Float64[], reward = 0.0, is_display_on_exit = true)

Store the total reward of each episode in the field of `rewards`. If
`is_display_on_exit` is set to `true`, a unicode plot will be shown at the [`PostExperimentStage`](@ref).
"""
mutable struct TotalRewardPerEpisode{T} <: AbstractHook where {T<:Union{Bool,Nothing}}
    rewards::Vector{Float64}
    reward::Float64
    is_display_on_exit::Bool

    function TotalRewardPerEpisode(; is_display_on_exit::Bool = true)
        struct_type = is_display_on_exit ? Bool : Nothing
        new{struct_type}(Float64[], 0.0, is_display_on_exit)
    end
end

Base.getindex(h::TotalRewardPerEpisode) = h.rewards

(h::TotalRewardPerEpisode)(::PostActStage, agent, env) =
    h.reward += reward(env)

function (hook::TotalRewardPerEpisode)(
    ::PostEpisodeStage,
    agent,
    env,
)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
end

function (hook::TotalRewardPerEpisode{Bool})(
    ::PostExperimentStage,
    agent,
    env,
)
    println(
        lineplot(
            hook.rewards,
            title = "Total reward per episode",
            xlabel = "Episode",
            ylabel = "Score",
        ),
    )
end

#####
# TotalBatchRewardPerEpisode
#####
struct TotalBatchRewardPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}}
    reward::Vector{Float64}
    is_display_on_exit::Bool
end

Base.getindex(h::TotalBatchRewardPerEpisode) = h.rewards

"""
    TotalBatchRewardPerEpisode(batch_size::Int; is_display_on_exit=true)

Similar to [`TotalRewardPerEpisode`](@ref), but is specific to environments
which return a `Vector` of rewards (a typical case with `MultiThreadEnv`).
If `is_display_on_exit` is set to `true`, a ribbon plot will be shown to reflect
the mean and std of rewards.
"""
function TotalBatchRewardPerEpisode(batch_size::Int; is_display_on_exit = true)
    TotalBatchRewardPerEpisode(
        [Float64[] for _ = 1:batch_size],
        zeros(batch_size),
        is_display_on_exit,
    )
end

function (hook::TotalBatchRewardPerEpisode)(
    ::PostActStage,
    agent,
    env,
)
    R = reward(env)
    for (i, (t, r)) in enumerate(zip(is_terminated(env), R))
        hook.reward[i] += r
        if t
            push!(hook.rewards[i], hook.reward[i])
            hook.reward[i] = 0.0
        end
    end
end

function (hook::TotalBatchRewardPerEpisode)(
    ::PostExperimentStage,
    agent,
    env,
)
    if hook.is_display_on_exit
        n = minimum(map(length, hook.rewards))
        m = mean([@view(x[1:n]) for x in hook.rewards])
        s = std([@view(x[1:n]) for x in hook.rewards])
        p = lineplot(
            m,
            title = "Avg total reward per episode",
            xlabel = "Episode",
            ylabel = "Score",
        )
        lineplot!(p, m .- s)
        lineplot!(p, m .+ s)
        println(p)
    end
end

#####
# BatchStepsPerEpisode
#####

struct BatchStepsPerEpisode <: AbstractHook
    steps::Vector{Vector{Int}}
    step::Vector{Int}
end

Base.getindex(h::BatchStepsPerEpisode) = h.steps

"""
    BatchStepsPerEpisode(batch_size::Int; tag = "TRAINING")

Similar to [`StepsPerEpisode`](@ref), but is specific to environments
which return a `Vector` of rewards (a typical case with `MultiThreadEnv`).
"""
function BatchStepsPerEpisode(batch_size::Int)
    BatchStepsPerEpisode([Int[] for _ = 1:batch_size], zeros(Int, batch_size))
end

function (hook::BatchStepsPerEpisode)(
    ::PostActStage,
    agent,
    env,
)
    for (i, t) in enumerate(is_terminated(env))
        hook.step[i] += 1
        if t
            push!(hook.steps[i], hook.step[i])
            hook.step[i] = 0
        end
    end
end

#####
# TimePerStep
#####

"""
    TimePerStep(;max_steps=100)
    TimePerStep(times::CircularVectorBuffer{Float64}, t::Float64)

Store time cost in seconds of the latest `max_steps` in the `times` field.
"""
struct TimePerStep{T} <: AbstractHook where {T<:AbstractFloat}
    times::CircularVectorBuffer{T}
    t::Vector{Float64}

    function TimePerStep(; max_steps = 100)
        new{Float64}(CircularVectorBuffer{Float64}(max_steps), Float64[time()])
    end

    function TimePerStep{T}(; max_steps = 100) where {T<: AbstractFloat}
        new{T}(CircularVectorBuffer{T}(max_steps), Float64[time()])
    end
end

Base.getindex(h::TimePerStep) = h.times

function (hook::TimePerStep)(::PostActStage, agent, env)
    push!(hook.times, (time() - hook.t[1]))
    hook.t[1] = time()
    return
end

"""
    DoEveryNStep(f; n=1, t=0)

Execute `f(t, agent, env)` every `n` step.
`t` is a counter of steps.
"""
mutable struct DoEveryNStep{F} <: AbstractHook
    f::F
    n::Int
    t::Int
end

DoEveryNStep(f; n = 1, t = 0) = DoEveryNStep(f, n, t)

function (hook::DoEveryNStep)(::PostActStage, agent, env)
    hook.t += 1
    if hook.t % hook.n == 0
        hook.f(hook.t, agent, env)
    end
end

"""
    DoEveryNEpisode(f; n=1, t=0)

Execute `f(t, agent, env)` every `n` episode.
`t` is a counter of episodes.
"""
mutable struct DoEveryNEpisode{S<:Union{PreEpisodeStage,PostEpisodeStage},F} <: AbstractHook
    f::F
    n::Int
    t::Int
end

DoEveryNEpisode(f::F; n = 1, t = 0, stage::S = PostEpisodeStage()) where {S,F} =
    DoEveryNEpisode{S,F}(f, n, t)

function (hook::DoEveryNEpisode{S})(::S, agent, env) where {S}
    hook.t += 1
    if hook.t % hook.n == 0
        hook.f(hook.t, agent, env)
    end
end

"""
    DoOnExit(f)

Call the lambda function `f` at the end of an [`Experiment`](@ref).
"""
struct DoOnExit{F} <: AbstractHook
    f::F
end

function (h::DoOnExit)(::PostExperimentStage, agent, env)
    h.f(agent, env)
end
