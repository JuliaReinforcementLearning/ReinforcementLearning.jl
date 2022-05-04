export Agent

import Functors: functor
using Setfield: @set

"""
    Agent(;kwargs...)

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between an agent and an environment
"""
Base.@kwdef struct Agent{P,T} <: AbstractPolicy
    policy::P
    trajectory::T
    function Agent(p, t) end
end

functor(x::Agent) = (policy = x.policy,), y -> @set x.policy = y.policy

(agent::Agent)(env) = agent.policy(env)

(agent::Agent)(::PreActStage, env, action) =
    push!(agent.trajectory; state = state(env), action = action)

(agent::Agent)(::PostActStage, env) =
    push!(agent.trajectory; reward = reward(env), terminal = is_terminated(env))

function (agent::Agent{P,<:Trajectory})(::PreActStage, env, action) where {P}
    push!(agent.trajectory; state = state(env), action = action)
    optimise!(agent.policy, agent.trajectory)
end

function optimise!(p::AbstractPolicy, t::AbstractTrajectory)
    for batch in t
        optimise!(p, batch)
    end
end