"""
    learn!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) with learning.
"""
function learn!(rlsetup)
    @unpack learner, policy, fillbuffer, preprocessor, buffer, environment, stoppingcriterion = rlsetup
    # s, a = firststateaction!(rlsetup) #TODO: callbacks don't see first state action
    # s = preprocess(preprocessor, reset!(environment))
    obs = reset!(environment)
    while true
        s = preprocessstate(preprocessor, obs)
        a = policy(s)
        next_obs, r, isdone = interact!(environment, a)
        fillbuffer && (pushstate!(buffer, s); pushaction!(buffer, a); pushreturn!(buffer, r, isdone)) 
        rlsetup.islearning && (update!(learner, buffer);)

        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, obs, a, r, isdone)
        end

        isbreak!(stoppingcriterion, obs, a, r, isdone) && break

        if isdone
            obs = reset!(environment)
        else
            obs = next_obs
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
