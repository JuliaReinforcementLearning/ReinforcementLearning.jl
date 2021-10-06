export AbstractHook,
    ComposedHook,
    EmptyHook,
    StepsPerEpisode,
    RewardsPerEpisode,
    TotalRewardPerEpisode,
    TotalBatchRewardPerEpisode,
    BatchStepsPerEpisode,
    TimePerStep,
    DoEveryNEpisode,
    DoEveryNStep,
    DoOnExit,
    UploadTrajectoryEveryNStep,
    MultiAgentHook

using UnicodePlots: lineplot, lineplot!
using Statistics

"""
A hook is called at different stage duiring a [`run`](@ref) to allow users to inject customized runtime logic.
By default, a `AbstractHook` will do nothing. One can override the behavior by implementing the following methods:

- `(hook::YourHook)(::PreActStage, agent, env, action)`, note that there's an extra argument of `action`.
- `(hook::YourHook)(::PostActStage, agent, env)`
- `(hook::YourHook)(::PreEpisodeStage, agent, env)`
- `(hook::YourHook)(::PostEpisodeStage, agent, env)`
- `(hook::YourHook)(::PostExperimentStage, agent, env)`
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
Base.@kwdef mutable struct RewardsPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
end

Base.getindex(h::RewardsPerEpisode) = h.rewards

function (hook::RewardsPerEpisode)(::PreEpisodeStage, agent, env)
    push!(hook.rewards, [])
end

function (hook::RewardsPerEpisode)(::PostActStage, agent, env)
    push!(hook.rewards[end], reward(env))
end

function (hook::RewardsPerEpisode)(::PostActStage, agent::NamedPolicy, env)
    push!(hook.rewards[end], reward(env, nameof(agent)))
end

#####
# TotalRewardPerEpisode
#####

"""
    TotalRewardPerEpisode(; rewards = Float64[], reward = 0.0, is_display_on_exit = true)

Store the total reward of each episode in the field of `rewards`. If
`is_display_on_exit` is set to `true`, a unicode plot will be shown at the [`PostExperimentStage`](@ref).
"""
Base.@kwdef mutable struct TotalRewardPerEpisode <: AbstractHook
    rewards::Vector{Float64} = Float64[]
    reward::Float64 = 0.0
    is_display_on_exit::Bool = true
end

Base.getindex(h::TotalRewardPerEpisode) = h.rewards

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env)
    hook.reward += reward(env)
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent::NamedPolicy, env)
    hook.reward += reward(env, nameof(agent))
end

function (hook::TotalRewardPerEpisode)(::PostEpisodeStage, agent, env)
    push!(hook.rewards, hook.reward)
    hook.reward = 0
end

function (hook::TotalRewardPerEpisode)(::PostExperimentStage, agent, env)
    if hook.is_display_on_exit
        println(
            lineplot(
                hook.rewards,
                title = "Total reward per episode",
                xlabel = "Episode",
                ylabel = "Score",
            ),
        )
    end
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
        [Float64[] for _ in 1:batch_size],
        zeros(batch_size),
        is_display_on_exit,
    )
end

function (hook::TotalBatchRewardPerEpisode)(::PostActStage, agent, env)
    R = agent isa NamedPolicy ? reward(env, nameof(agent)) : reward(env)
    for (i, (t, r)) in enumerate(zip(is_terminated(env), R))
        hook.reward[i] += r
        if t
            push!(hook.rewards[i], hook.reward[i])
            hook.reward[i] = 0.0
        end
    end
end

function (hook::TotalBatchRewardPerEpisode)(::PostExperimentStage, agent, env)
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
    BatchStepsPerEpisode([Int[] for _ in 1:batch_size], zeros(Int, batch_size))
end

function (hook::BatchStepsPerEpisode)(::PostActStage, agent, env)
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
    TimePerStep(times::CircularArrayBuffer{Float64}, t::UInt64)

Store time cost of the latest `max_steps` in the `times` field.
"""
mutable struct TimePerStep <: AbstractHook
    times::CircularArrayBuffer{Float64,1}
    t::UInt64
end

Base.getindex(h::TimePerStep) = h.times

TimePerStep(; max_steps = 100) =
    TimePerStep(CircularArrayBuffer{Float64}(max_steps), time_ns())

function (hook::TimePerStep)(::PostActStage, agent, env)
    push!(hook.times, (time_ns() - hook.t) / 1e9)
    hook.t = time_ns()
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

DoEveryNEpisode(f::F; n = 1, t = 0, stage::S = POST_EPISODE_STAGE) where {S,F} =
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
    h.f()
end

"""
    UploadTrajectoryEveryNStep(;mailbox, n, sealer=deepcopy)
"""
Base.@kwdef mutable struct UploadTrajectoryEveryNStep{M,S} <: AbstractHook
    mailbox::M
    n::Int
    t::Int = -1
    sealer::S = deepcopy
end

function (hook::UploadTrajectoryEveryNStep)(::PostActStage, agent::Agent, env)
    hook.t += 1
    if hook.t > 0 && hook.t % hook.n == 0
        put!(hook.mailbox, hook.sealer(agent.trajectory))
    end
end

"""
    MultiAgentHook(player=>hook...)
"""
struct MultiAgentHook <: AbstractHook
    hooks::Dict{Any,Any}
end

MultiAgentHook(player_hook_pair::Pair...) = MultiAgentHook(Dict(player_hook_pair...))

Base.getindex(h::MultiAgentHook, p) = getindex(h.hooks, p)

function (hook::MultiAgentHook)(
    s::AbstractStage,
    m::MultiAgentManager,
    env::AbstractEnv,
    args...,
)
    for (p, h) in zip(values(m.agents), values(hook.hooks))
        h(s, p, env, args...)
    end
end
