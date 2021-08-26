export NamedPolicy

import Functors: functor
using Setfield: @set

"""
    NamedPolicy(name=>policy)

A policy wrapper to provide a name. Mostly used in multi-agent environments.
"""
Base.@kwdef struct NamedPolicy{P,N} <: AbstractPolicy
    name::N
    policy::P
end

NamedPolicy((name, policy)) = NamedPolicy(name, policy)

functor(x::NamedPolicy) = (policy = x.policy,), y -> @set x.policy = y.policy

Base.nameof(agent::NamedPolicy) = agent.name

function check(agent::NamedPolicy, env::AbstractEnv)
    check(agent.policy, env)
end

function RLBase.update!(
    p::NamedPolicy,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    update!(p.policy, t, e, s)
end

function RLBase.update!(
    p::NamedPolicy,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::PreActStage,
)
    update!(p.policy, t, e, s)
end


(p::NamedPolicy)(env::AbstractEnv) = DynamicStyle(env) == SEQUENTIAL ? p.policy(env) : p.policy(env, p.name)
(p::NamedPolicy)(s::AbstractStage, env::AbstractEnv) = p.policy(s, env)
(p::NamedPolicy)(s::PreActStage, env::AbstractEnv, action) = p.policy(s, env, action)
