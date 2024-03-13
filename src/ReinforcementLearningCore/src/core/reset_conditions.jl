export AbstractResetCondition, ResetIfEnvTerminated, ResetAfterNSteps
  
abstract type AbstractResetCondition end

"""
ResetIfEnvTerminated()

A reset condition that resets the environment if is_terminated(env) is true.
"""
struct ResetIfEnvTerminated <: AbstractResetCondition end

check!(::ResetIfEnvTerminated, policy::AbstractPolicy, env::AbstractEnv) = is_terminated(env)

"""
    ResetAfterNSteps(n)

A reset condition that resets the environment after `n` steps.
"""
mutable struct ResetAfterNSteps <: AbstractResetCondition
    t::Int
    n::Int
end

ResetAfterNSteps(n::Int) = ResetAfterNSteps(0, n)

function check!(r::ResetAfterNSteps, policy::AbstractPolicy, env::AbstractEnv)
    stop = r.t >= r.n
    r.t += 1
    if stop
        r.t = 0
        return true
    else
        return false
    end
end
