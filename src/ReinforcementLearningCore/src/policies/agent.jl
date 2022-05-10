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
        if TrajectoryStyle(t) === AsyncTrajectoryStyle
            t = @spawn optimise!(p, t)
            bind(agent.trajectory, t)
        end
        agent
    end
end

optimise!(::AbstractPolicy) = nothing
optimise!(agent::Agent) = optimise!(TrajectoryStyle(agent.trajectory), agent)
optimise!(::SyncTrajectoryStyle, agent::Agent) = optimise!(agent.policy, agent.trajectory)
optimise!(::AsyncTrajectoryStyle, agent::Agent) = nothing

function optimise!(policy::AbstractPolicy, trajectory::Trajectory)
    for batch in trajectory
        optimise!(policy, batch)
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

(agent::Agent)(::PreActStage, env, action) =
    push!(agent.trajectory; state = state(env), action = action)
