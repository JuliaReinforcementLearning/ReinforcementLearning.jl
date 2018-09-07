@inline function step!(rlsetup, a)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    s0, r0, done0 = interact!(environment, a)
    s, r, done = preprocess(preprocessor, s0, r0, done0)
    # if fillbuffer; pushreturn!(buffer, r, done) end
    if done
        s0 = reset!(environment)
        s = preprocessstate(preprocessor, s0) 
    end
    # if fillbuffer; pushstate!(buffer, s) end
    a = policy(s)
    # if fillbuffer pushaction!(buffer, a) end
    s0, a, r, done
end
@inline function firststateaction!(rlsetup)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    if isempty(buffer.actions)
        sraw, done = getstate(environment)
        if done; sraw = reset!(environment); end
        s = preprocessstate(preprocessor, sraw)
        if fillbuffer; pushstate!(buffer, s) end
        a = policy(s)
        if fillbuffer; pushaction!(buffer, a) end
        (s, a)
    else
        (buffer.states[end], buffer.actions[end])
    end
end

"""
    learn!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) with learning.
"""
function learn!(rlsetup)
    @unpack learner, policy, fillbuffer, preprocessor, buffer, environment, stoppingcriterion = rlsetup
    s, a = firststateaction!(rlsetup) #TODO: callbacks don't see first state action
    # s = preprocess(preprocessor, reset!(environment))
    while true
        s_nxt_raw, r, isdone = interact!(environment, a)
        fillbuffer && pushreturn!(buffer, r, isdone)
        isdone && (s_nxt_raw = reset!(environment);)
        s = preprocessstate(preprocessor, s_nxt_raw)
        a = policy(s)
        fillbuffer && (pushstate!(buffer, s); pushaction!(buffer, a))

        if rlsetup.islearning; update!(learner, buffer); end
        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, s_nxt_raw, a, r, isdone)
        end

        isbreak!(stoppingcriterion, s_nxt_raw, a, r, isdone) && break
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
