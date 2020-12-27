export NamedPolicy

import Functors: functor
using Setfield: @set

"""
    NamedPolicy(name=>policy)

A policy wrapper to provide a name. Mostly used in multi-agent environments.
"""
struct NamedPolicy{P,N} <: AbstractPolicy
    name::N
    policy::P
end

NamedPolicy((name, policy)) = NamedPolicy(name, policy)

functor(x::NamedPolicy) = (policy = x.policy,), y -> @set x.policy = y.policy

Base.nameof(agent::NamedPolicy) = agent.name

function check(agent::NamedPolicy, env::AbstractEnv)
    check(agent.policy, env)
end

RLBase.update!(p::NamedPolicy, args...) = update!(p.policy, args...)

(p::NamedPolicy)(env::AbstractEnv) = p.policy(env)
