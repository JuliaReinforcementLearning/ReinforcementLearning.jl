@inline function step!(rlsetup, a)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    s0, r0, done0 = interact!(environment, a)
    s, r, done = preprocess(preprocessor, s0, r0, done0)
    if fillbuffer; pushreturn!(buffer, r, done) end
    if done
        s0, = reset!(environment)
        s = preprocessstate(preprocessor, s0) 
    end
    if fillbuffer; pushstate!(buffer, s) end
    a = policy(s)
    if fillbuffer pushaction!(buffer, a) end
    s0, a, r, done
end
@inline function firststateaction!(rlsetup)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    if isempty(buffer.actions)
        sraw, done = getstate(environment)
        if done; sraw, = reset!(environment); end
        s = preprocessstate(preprocessor, sraw)
        if fillbuffer; pushstate!(buffer, s) end
        a = policy(s)
        if fillbuffer; pushaction!(buffer, a) end
        a
    else
        buffer.actions[end]
    end
end

"""
    learn!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) with learning.
"""
function learn!(rlsetup)
    @unpack learner, buffer = rlsetup
    a = firststateaction!(rlsetup) #TODO: callbacks don't see first state action
    while true
        sraw, a, r, done = step!(rlsetup, a)
        if rlsetup.islearning; update!(learner, buffer); end
        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, sraw, a, r, done)
        end
        if isbreak!(rlsetup.stoppingcriterion, sraw, a, r, done); break; end
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
