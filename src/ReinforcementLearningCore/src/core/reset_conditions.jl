export ResetAtTerminal, ResetAfterNSteps

"""
    ResetAtTerminal()

A reset condition that resets if is_terminated(env) is true.
"""
struct ResetAtTerminal end

(::ResetAtTerminal)(policy, env) = is_terminated(env)

"""
    ResetAfterNSteps(n)

A reset condition that resets after `n` steps in the environment.
"""
mutable struct ResetAfterNSteps
    t::Int
    n::Int
end

ResetAfterNSteps(n::Int) = ResetAfterNSteps(0, n)

function (r::ResetAfterNSteps)(policy, env) 
    r.t += 1
    stop = r.t >= r.n
    if stop
        r.t = 0
        return true
    else
        return false
    end
end
