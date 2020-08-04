export StopAfterStep, StopAfterEpisode, StopWhenDone, ComposedStopCondition

using ProgressMeter

const update! = ReinforcementLearningBase.update!

#####
# ComposedStopCondition
#####

"""
    ComposedStopCondition(stop_conditions; reducer = any)

The result of `stop_conditions` is reduced by `reducer`.
"""
struct ComposedStopCondition{T<:Function}
    stop_conditions::Vector{Any}
    reducer::T
end

ComposedStopCondition(stop_conditions; reducer = any) =
    ComposedStopCondition(stop_conditions, reducer)

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
