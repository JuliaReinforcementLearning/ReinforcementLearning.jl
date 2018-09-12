"""
    learn!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) with learning.
"""
function learn!(rlsetup)
    @unpack learner, policy, fillbuffer, preprocessor, buffer, environment, stoppingcriterion = rlsetup
    if isempty(buffer) || buffer[end].isdone
        obs = reset!(environment)
        s = preprocessstate(preprocessor, obs)
        a = policy(s)
    else
        lastturn = buffer[end]
        s, a = lastturn.nextstate, lastturn.nextaction
    end
    while true
        next_obs, r, isdone = interact!(environment, a)
        next_s = preprocessstate(preprocessor, next_obs)
        next_a = policy(next_s)

        fillbuffer &&  push!(buffer, Turn(s,a,r,isdone,next_s, next_a))
        rlsetup.islearning && update!(learner, buffer)

        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, s, a, r, isdone)
        end

        isbreak!(stoppingcriterion, next_obs, a, r, isdone) && break

        if isdone
            obs = reset!(environment)
            s = preprocessstate(preprocessor, obs)
            a = policy(s)
        else
            s, a = next_s, next_a  # TODO: to handle async env, use `getstate(env)`
        end
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
