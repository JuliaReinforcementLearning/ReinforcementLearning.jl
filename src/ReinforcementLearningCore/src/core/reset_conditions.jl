export ResetAtTerminal, ResetAfterNSteps

struct ResetAtTerminal end

(::ResetAtTerminal)(policy, env) = is_terminal(env)

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
