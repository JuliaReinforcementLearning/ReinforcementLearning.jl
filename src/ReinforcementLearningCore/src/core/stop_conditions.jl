export StopAfterStep, StopAfterEpisode, StopWhenDone, ComposedStopCondition, StopSignal

using ProgressMeter

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
        new{typeof(stop_conditions), typeof(reducer)}(stop_conditions, reducer)
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
    if get_terminal(env)
        s.cur += 1
        if !isnothing(s.progress)
            next!(s.progress;)
        end
    end

    s.cur >= s.episode
end

(s::StopAfterEpisode)(agent, env::MultiThreadEnv) =
    @error "MultiThreadEnv is not supported!"

#####
# StopWhenDone
#####

"""
    StopWhenDone()

Return `true` if the environment is terminated.
"""
struct StopWhenDone end

(s::StopWhenDone)(agent, env) = get_terminal(env)

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