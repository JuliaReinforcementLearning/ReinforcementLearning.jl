using ProgressMeter

#####
# StopAfterStep
#####
mutable struct StopAfterStep{Tl}
    step::Int
    cur::Int
    progress::Tl
    tag::String
end

function StopAfterStep(step; cur=0, is_show_progress, tag="TRAINING")
    progress=Progress(step)
    ProgressMeter.update!(progress, cur)
    StopAfterStep(step, cur, progress, tag)
end

function (s::StopAfterStep)(args...)
    !isnothing(s.progress) && next!(s.progress; showvalues=[(Symbol(s.tag, "/", :STEP), s.cur)])
    @debug s.tag STEP=s.cur

    res = s.step >= s.cur
    s.cur += 1
    res
end

#####
# StopAfterEpisode
#####

mutable struct StopAfterEpisode{Tl}
    episode::Int
    cur::Int
    progress::Tl
    tag::String
end

function StopAfterEpisode(episode; cur=0, is_show_progress, tag="TRAINING")
    progress=Progress(episode)
    ProgressMeter.update!(progress, cur)
    StopAfterEpisode(episode, cur, progress, tag)
end

function (s::StopAfterEpisode)(args...)
    !isnothing(s.progress) && next!(s.progress; showvalues=[(Symbol(s.tag, "/", :EPISODE), s.cur)])
    @debug s.tag EPISODE=s.cur

    res = s.episode >= s.cur
    s.cur += 1
    res
end

#####
# StopWhenDone
#####

struct StopWhenDone
end

(s::StopWhenDone)(agent, env, obs) = terminal(obs)