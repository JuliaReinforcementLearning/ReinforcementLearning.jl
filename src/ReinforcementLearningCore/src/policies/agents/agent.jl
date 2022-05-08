export Agent

using Base.Threads
import Functors: functor
using Setfield: @set

"""
    Agent(;policy, trajectory)

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`Trajectory`](@ref): used to store intractions between an agent and an environment
"""
Base.@kwdef struct Agent{P,T} <: AbstractPolicy
    policy::P
    trajectory::T

    function Agent(p::P, t::T) where {P,T}
        agent = new{P,T}(p, t)
        bind(t, @spawn(optimise!(agent)))
        agent
    end
end

functor(x::Agent) = (policy = x.policy,), y -> @set x.policy = y.policy

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
(agent::Agent)(env) = agent.policy(env)

(agent::Agent)(::PreActStage, env, action) =
    push!(agent.trajectory; state = state(env), action = action)

(agent::Agent)(::PostActStage, env) =
    push!(agent.trajectory; reward = reward(env), terminal = is_terminated(env))

function (agent::Agent)(::PreActStage, env, action)
    push!(agent.trajectory; state = state(env), action = action)
    if TrajectoryStyle(agent.trajectory) === SyncTrajectoryStyle()
        optimise!(agent)
    end
end

function optimise!(agent::Agent)
    for batch in agent.trajectory
        optimise!(agent.policy, batch)
    end
end
