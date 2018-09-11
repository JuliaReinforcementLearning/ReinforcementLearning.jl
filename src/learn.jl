"""
    learn!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) with learning.
"""
function learn!(rlsetup)
    @unpack learner, policy, fillbuffer, preprocessor, buffer, environment, stoppingcriterion = rlsetup
    obs = reset!(environment)
    while true
        s = preprocessstate(preprocessor, obs)
        a = policy(s)
        next_obs, r, isdone = interact!(environment, a)
        next_s = preprocessstate(preprocessor, next_obs)
        next_a = policy(next_s)

        fillbuffer &&  push!(buffer, Turn(s,a,r,isdone,next_s, next_a))
        rlsetup.islearning && update!(learner, buffer)

        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, obs, a, r, isdone)
        end

        isbreak!(stoppingcriterion, next_obs, a, r, isdone) && break
        obs = isdone ? reset!(environment) : next_obs
    end
end

"""
    run!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) without learning.
"""
function run!(rlsetup; fillbuffer = false)
    @unpack islearning = rlsetup
    rlsetup.islearning = false
    tmp = rlsetup.fillbuffer
    rlsetup.fillbuffer = fillbuffer
    learn!(rlsetup)
    rlsetup.islearning = islearning
    rlsetup.fillbuffer = tmp
end

export learn!, run!
