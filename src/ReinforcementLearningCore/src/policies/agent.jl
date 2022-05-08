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
    task_ref::Ref{Task}

    function Agent(p::P, t::T) where {P,T}
        agent = new{P,T}(p, t, Ref{Task}())
        optimise!(agent)
        agent
    end
end

optimise!(::AbstractPolicy) = nothing

function optimise!(agent::Agent)
    if TrajectoryStyle(agent.trajectory) isa SyncTrajectoryStyle
        optimise!(agent.policy, agent.trajectory)
    else
        if !isassigned(agent.task_ref)
            t = @spawn optimise!(agent.policy, agent.trajectory)
            bind(agent.trajectory, t)
            agent.task_ref[] = t
        end
    end
end

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
