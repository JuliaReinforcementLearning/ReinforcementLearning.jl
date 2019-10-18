export StopAfterStep, StopAfterEpisode, StopWhenDone, ComposedStopCondition

using ProgressMeter

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
    StopAfterStep(step; cur = 1, is_show_progress = true, tag = "TRAINING")

Return `true` after being called for `step`.
"""
mutable struct StopAfterStep{Tl}
    step::Int
    cur::Int
    progress::Tl
    tag::String
end

function StopAfterStep(step; cur = 1, is_show_progress = true, tag = "TRAINING")
    if is_show_progress
        progress = Progress(step)
        ProgressMeter.update!(progress, cur)
    else
        progress = nothing
    end
    StopAfterStep(step, cur, progress, tag)
end

function (s::StopAfterStep)(args...)
    !isnothing(s.progress) && next!(
        s.progress;
        # showvalues = [(Symbol(s.tag, "/", :STEP), s.cur)],  # https://github.com/timholy/ProgressMeter.jl/pull/131
    )
    @debug s.tag STEP = s.cur

    res = s.cur >= s.step
    s.cur += 1
    res
end

#####
# StopAfterEpisode
#####

"""
    StopAfterEpisode(episode; cur = 0, is_show_progress = true, tag = "TRAINING")

Return `true` after being called `episode`. If `is_show_progress` is `true`, the `ProgressMeter` will be used to show progress.
"""
mutable struct StopAfterEpisode{Tl}
    episode::Int
    cur::Int
    progress::Tl
    tag::String
end

function StopAfterEpisode(episode; cur = 0, is_show_progress = true, tag = "TRAINING")
    if is_show_progress
        progress = Progress(episode)
        ProgressMeter.update!(progress, cur)
    else
        progress = nothing
    end
    StopAfterEpisode(episode, cur, progress, tag)
end

function (s::StopAfterEpisode)(agent, env, obs)
    !isnothing(s.progress) && next!(
        s.progress;
        # showvalues = [(Symbol(s.tag, "/", :EPISODE), s.cur)],  # https://github.com/timholy/ProgressMeter.jl/pull/131
    )
    @debug s.tag EPISODE = s.cur

    get_terminal(obs) && (s.cur += 1)
    s.cur >= s.episode
end

#####
# StopWhenDone
#####

"""
    StopWhenDone()

Return `true` if the `terminal` field of an observation is `true`.
"""
struct StopWhenDone end

(s::StopWhenDone)(agent, env, obs) = get_terminal(obs)