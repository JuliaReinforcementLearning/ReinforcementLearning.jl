export StopAfterStep,
    StopAfterEpisode,
    StopWhenDone,
    ComposedStopCondition,
    StopSignal,
    StopAfterNoImprovement,
    StopAfterNSeconds

using ProgressMeter
using CircularArrayBuffers: CircularArrayBuffer, isfull

const update! = ReinforcementLearningBase.update!

#####
# ComposedStopCondition
#####

"""
    ComposedStopCondition(stop_conditions...; reducer = any)

The result of `stop_conditions` is reduced by `reducer`.
"""
struct ComposedStopCondition{S,T}
    stop_conditions::S
    reducer::T
    function ComposedStopCondition(stop_conditions...; reducer = any)
        new{typeof(stop_conditions),typeof(reducer)}(stop_conditions, reducer)
    end
end

function (s::ComposedStopCondition)(args...)
    s.reducer(sc(args...) for sc in s.stop_conditions)
end

#####
# StopAfterStep
#####
"""
    StopAfterStep(step; cur = 1, is_show_progress = true)

Return `true` after being called `step` times.
"""
mutable struct StopAfterStep{Tl}
    step::Int
    cur::Int
    progress::Tl
end

function StopAfterStep(step; cur = 1, is_show_progress = true)
    if is_show_progress
        progress = Progress(step, 1)
        ProgressMeter.update!(progress, cur)
    else
        progress = nothing
    end
    StopAfterStep(step, cur, progress)
end

function (s::StopAfterStep)(args...)
    if !isnothing(s.progress)
        # https://github.com/timholy/ProgressMeter.jl/pull/131
        # next!(s.progress; showvalues = [(Symbol(s.tag, "/", :STEP), s.cur)])
        next!(s.progress)
    end

    @debug s.tag STEP = s.cur

    res = s.cur >= s.step
    s.cur += 1
    res
end

#####
# StopAfterEpisode
#####

"""
    StopAfterEpisode(episode; cur = 0, is_show_progress = true)

Return `true` after being called `episode`. If `is_show_progress` is `true`, the `ProgressMeter` will be used to show progress.
"""
mutable struct StopAfterEpisode{Tl}
    episode::Int
    cur::Int
    progress::Tl
end

function StopAfterEpisode(episode; cur = 0, is_show_progress = true)
    if is_show_progress
        progress = Progress(episode, 1)
        ProgressMeter.update!(progress, cur)
    else
        progress = nothing
    end
    StopAfterEpisode(episode, cur, progress)
end

function (s::StopAfterEpisode)(agent, env)
    if is_terminated(env)
        s.cur += 1
        if !isnothing(s.progress)
            next!(s.progress;)
        end
    end

    s.cur >= s.episode
end

"""
StopAfterNoImprovement()

Stop training when a monitored metric has stopped improving.

Parameters:

fn: a closure, return a scalar value, which indicates the performance of the policy (the higher the better)
e.g.
1. () -> reward(env)
1. () -> total_reward_per_episode.reward

patience: Number of epochs with no improvement after which training will be stopped.

δ: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.

Return `true` after the monitored metric has stopped improving.
"""
mutable struct StopAfterNoImprovement{T<:Number,F}
    fn::F
    patience::Int
    δ::T
    peak::T
    counter::Int
end

function StopAfterNoImprovement(fn, patience::Int, δ::T = 0.0f0) where {T<:Number}
    StopAfterNoImprovement(fn, patience, δ, typemin(T), 1)
end

function (s::StopAfterNoImprovement)(agent, env)::Bool
    is_terminated(env) || return false # post episode stage
    val = s.fn()
    if s.δ < val - s.peak
        s.counter = 1
        s.peak = max(val, s.peak)
        return false
    else
        s.counter += 1
        return s.counter > s.patience
    end
    return false
end

#####
# StopWhenDone
#####

"""
    StopWhenDone()

Return `true` if the environment is terminated.
"""
struct StopWhenDone end

(s::StopWhenDone)(agent, env) = is_terminated(env)

#####
# StopSignal
#####

"""
    StopSignal()

Create a stop signal initialized with a value of `false`.
You can manually set it to `true` by `s[] = true` to stop
the running loop at any time.
"""
Base.@kwdef struct StopSignal
    is_stop::Ref{Bool} = Ref(false)
end

Base.getindex(s::StopSignal) = s.is_stop[]
Base.setindex!(s::StopSignal, v::Bool) = s.is_stop[] = v

(s::StopSignal)(agent, env) = s[]

"""
StopAfterNSeconds

parameter:
1. time budget

stop training after N seconds

"""
Base.@kwdef mutable struct StopAfterNSeconds
    budget::Float64
    deadline::Float64 = 0.0
end
function RLBase.reset!(s::StopAfterNSeconds)
    s.deadline = time() + s.budget
    s
end
function StopAfterNSeconds(budget::Float64)
    s = StopAfterNSeconds(; budget)
    RLBase.reset!(s)
end
(s::StopAfterNSeconds)(_...) = time() > s.deadline
