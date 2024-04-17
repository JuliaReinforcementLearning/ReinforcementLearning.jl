export AbstractHook,
    EmptyHook,
    ComposedHook,
    StepsPerEpisode,
    RewardsPerEpisode,
    TotalRewardPerEpisode,
    BatchStepsPerEpisode,
    TimePerStep,
    DoEveryNEpisodes,
    DoEveryNSteps,
    DoOnExit

using UnicodePlots: lineplot, lineplot!
using Statistics: mean, std
using CircularArrayBuffers: CircularVectorBuffer
import ReinforcementLearningBase: RLBase
import Base.push!

"""
A hook is called at different stage during a [`run`](@ref) to allow users to inject customized runtime logic.
By default, an `AbstractHook` will do nothing. One can customize the behavior by implementing the following methods:

- `Base.push!(hook::YourHook, ::PreActStage, agent, env)`
- `Base.push!(hook::YourHook, ::PostActStage, agent, env)`
- `Base.push!(hook::YourHook, ::PreEpisodeStage, agent, env)`
- `Base.push!(hook::YourHook, ::PostEpisodeStage, agent, env)`
- `Base.push!(hook::YourHook, ::PostExperimentStage, agent, env)`

By convention, the `Base.getindex(h::YourHook)` is implemented to extract the metrics we are interested in.
Users can compose different `AbstractHook`s with `+`.
"""
abstract type AbstractHook end

# Pushing to hooks is a noop unless `push!` for the stage has been specified
Base.push!(::AbstractHook, ::AbstractStage, ::AbstractPolicy, ::AbstractEnv) = nothing

struct ComposedHook{T<:Tuple} <: AbstractHook
    hooks::T
    ComposedHook(hooks...) = new{typeof(hooks)}(hooks)
end

Base.:(+)(h1::AbstractHook, h2::AbstractHook) = ComposedHook(h1, h2)
Base.:(+)(h1::ComposedHook, h2::AbstractHook) = ComposedHook(h1.hooks..., h2)
Base.:(+)(h1::AbstractHook, h2::ComposedHook) = ComposedHook(h1, h2.hooks...)
Base.:(+)(h1::ComposedHook, h2::ComposedHook) = ComposedHook(h1.hooks..., h2.hooks...)

@inline function _push!(stage::AbstractStage, policy::P, env::E, hook::H, hook_tuple...) where {P <: AbstractPolicy, E <: AbstractEnv, H <: AbstractHook}
    push!(hook, stage, policy, env)
    _push!(stage, policy, env, hook_tuple...)
end

_push!(stage::AbstractStage, policy::P, env::E) where {P <: AbstractPolicy, E <: AbstractEnv} = nothing

function Base.push!(composed_hook::ComposedHook{T},
                            stage::AbstractStage,
                            policy::P,
                            env::E
                            ) where {T <: Tuple, P <: AbstractPolicy, E <: AbstractEnv}
    _push!(stage, policy, env, composed_hook.hooks...)
end

Base.getindex(hook::ComposedHook, inds...) = getindex(hook.hooks, inds...)

#####
# EmptyHook
#####

"""
Nothing but a placeholder.
"""
struct EmptyHook <: AbstractHook end

const EMPTY_HOOK = EmptyHook()


#####
# MultiPlayer default behavior
#####
Base.push!(hook::AbstractHook, stage::AbstractStage, agent::AbstractPolicy, env::AbstractEnv, ::AbstractPlayer) = push!(hook, stage, agent, env)

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

Base.push!(hook::StepsPerEpisode, ::PostActStage, agent::AbstractPolicy, env::AbstractEnv) = hook.count += 1

Base.push!(hook::StepsPerEpisode, stage::PostEpisodeStage, agent::AbstractPolicy, env::AbstractEnv, ::Player) = push!(hook, stage, agent, env)

function Base.push!(hook::StepsPerEpisode, ::PostEpisodeStage, agent::AbstractPolicy, env::AbstractEnv)
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

    function RewardsPerEpisode{T}() where {T<:Number}
        new{T}(Vector{Vector{T}}())
    end

    function RewardsPerEpisode()
        RewardsPerEpisode{Float64}()
    end
end

Base.getindex(h::RewardsPerEpisode) = h.rewards

function Base.push!(h::RewardsPerEpisode{T}, ::PreEpisodeStage, agent::AbstractPolicy, env::AbstractEnv) where {T<:Number}
    push!(h.rewards, T[])
end

Base.push!(h::RewardsPerEpisode, ::PostActStage, agent::P, env::E) where {P <: AbstractPolicy, E <: AbstractEnv} = push!(last(h.rewards), reward(env))
Base.push!(h::RewardsPerEpisode, ::PostActStage, agent::Policy, env::E, player::Player) where {Policy <: AbstractPolicy, E <: AbstractEnv, Player <: AbstractPlayer} = push!(last(h.rewards), reward(env, player))

#####
# TotalRewardPerEpisode
#####

"""
    TotalRewardPerEpisode(; is_display_on_exit = true)

Store the total reward of each episode in the field of `rewards`. If
`is_display_on_exit` is set to `true`, a unicode plot will be shown at the [`PostExperimentStage`](@ref).
"""
mutable struct TotalRewardPerEpisode{T,F} <: AbstractHook where {T<:Union{Val{true},Val{false}},F<:Number}
    rewards::Vector{F}
    reward::F
    is_display_on_exit::Bool

    function TotalRewardPerEpisode{F}(; is_display_on_exit::Bool=true) where {F<:Number}
        new{Val{is_display_on_exit},F}(Vector{F}(), 0.0, is_display_on_exit)
    end

    function TotalRewardPerEpisode(; is_display_on_exit::Bool=true)
        TotalRewardPerEpisode{Float64}(; is_display_on_exit=is_display_on_exit)
    end
end

Base.getindex(h::TotalRewardPerEpisode) = h.rewards

Base.push!(h::TotalRewardPerEpisode, ::PostActStage, agent::P, env::E) where {P <: AbstractPolicy, E <: AbstractEnv} = h.reward += reward(env)
Base.push!(h::TotalRewardPerEpisode, ::PostActStage, agent::P, env::E, player::Player) where {P <: AbstractPolicy, E <: AbstractEnv, Player <: AbstractPlayer} = h.reward += reward(env, player)

function Base.push!(hook::TotalRewardPerEpisode,
    ::PostEpisodeStage,
    agent::AbstractPolicy,
    env::AbstractEnv,
)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
    return
end

function Base.show(io::IO, hook::TotalRewardPerEpisode{true, F}) where {F<:Number}
    if length(hook.rewards) > 0
        println(io, lineplot(
            hook.rewards,
            title="Total reward per episode",
            xlabel="Episode",
            ylabel="Score",
        ))
    else
        println(io, typeof(hook))
    end
    return
end

function Base.push!(hook::TotalRewardPerEpisode{true, F}, 
    ::PostExperimentStage,
    agent::AbstractPolicy,
    env::AbstractEnv,
) where {F<:Number}
    display(hook)
    return
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
    BatchStepsPerEpisode(batchsize::Int; tag = "TRAINING")

Similar to [`StepsPerEpisode`](@ref), but is specific to environments
which return a `Vector` of rewards (a typical case with `MultiThreadEnv`).
"""
function BatchStepsPerEpisode(batchsize::Int)
    BatchStepsPerEpisode([Int[] for _ = 1:batchsize], zeros(Int, batchsize))
end

function Base.push!(hook::BatchStepsPerEpisode, 
    ::PostActStage,
    agent::AbstractPolicy,
    env::AbstractEnv,
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

    function TimePerStep(; max_steps=100)
        new{Float64}(CircularVectorBuffer{Float64}(max_steps), Float64[time()])
    end

    function TimePerStep{T}(; max_steps=100) where {T<:AbstractFloat}
        new{T}(CircularVectorBuffer{T}(max_steps), Float64[time()])
    end
end

Base.getindex(h::TimePerStep) = h.times

function Base.push!(hook::TimePerStep, ::PostActStage, agent::AbstractPolicy, env::AbstractEnv)
    push!(hook.times, (time() - hook.t[1]))
    hook.t[1] = time()
    return
end

"""
    DoEveryNSteps(f; n=1, t=0)

Execute `f(t, agent, env)` every `n` step.
`t` is a counter of steps.
"""
mutable struct DoEveryNSteps{F} <: AbstractHook where {F}
    f::F
    n::Int
    t::Int
    
    function DoEveryNSteps(f::F; n=1, t=0) where {F}
        new{F}(f, n, t)
    end
end

function Base.push!(hook::DoEveryNSteps, ::PostActStage, agent::AbstractPolicy, env::AbstractEnv)
    hook.t += 1
    if hook.t % hook.n == 0
        hook.f(hook.t, agent, env)
    end
    return
end

"""
    DoEveryNEpisodes(f; n=1, t=0)

Execute `f(t, agent, env)` every `n` episode.
`t` is a counter of episodes.
"""
mutable struct DoEveryNEpisodes{S<:Union{PreEpisodeStage,PostEpisodeStage},F} <: AbstractHook
    f::F
    n::Int
    t::Int
end

DoEveryNEpisodes(f::F; n=1, t=0, stage::S=PostEpisodeStage()) where {S<:AbstractStage,F} =
    DoEveryNEpisodes{S,F}(f, n, t)

function Base.push!(hook::DoEveryNEpisodes{S}, ::S, agent::AbstractPolicy, env::AbstractEnv) where {S<:AbstractStage}
    hook.t += 1
    if hook.t % hook.n == 0
        hook.f(hook.t, agent, env)
    end
    return
end

"""
    DoOnExit(f)

Call the lambda function `f` at the end of an `Experiment`.
"""
struct DoOnExit{F} <: AbstractHook
    f::F
end

function Base.push!(h::DoOnExit, ::PostExperimentStage, agent::AbstractPolicy, env::AbstractEnv)
    h.f(agent, env)
end
