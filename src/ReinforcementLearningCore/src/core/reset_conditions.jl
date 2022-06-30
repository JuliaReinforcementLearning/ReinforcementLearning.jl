export ResetAtTerminal, ResetAfterNSteps

"""
    ResetAtTerminal()

A reset condition that resets the environemnt if is_terminated(env) is true.
"""
struct ResetAtTerminal end

(::ResetAtTerminal)(policy, env) = is_terminated(env)

"""
    ResetAfterNSteps(n)

A reset condition that resets the environment after `n` steps.
"""
mutable struct ResetAfterNSteps
    t::Int
    n::Int
end

ResetAfterNSteps(n::Int) = ResetAfterNSteps(0, n)

function (r::ResetAfterNSteps)(policy, env) 
    stop = r.t >= r.n
    r.t += 1
    if stop
        r.t = 0
        return true
    else
        return false
    end
end
