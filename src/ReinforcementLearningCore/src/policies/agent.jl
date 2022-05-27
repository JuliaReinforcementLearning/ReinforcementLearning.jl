export Agent, optimise!

using Base.Threads
import Functors: functor
using ReinforcementLearningTrajectories

"""
    Agent(;policy, trajectory)

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`Trajectory`](@ref): used to store intractions between an agent and an environment
"""
mutable struct Agent{P,T} <: AbstractPolicy
    policy::P
    trajectory::T
    cache::NamedTuple # trajectory do not support partial inserting

    function Agent(p::P, t::T, cache = NamedTuple()) where {P,T}
        agent = new{P,T}(p, t, cache)
        if TrajectoryStyle(t) === AsyncTrajectoryStyle
            t = @spawn optimise!(p, t)
            bind(agent.trajectory, t)
        end
        agent
    end
end

Agent(; policy, trajectory, cache = NamedTuple()) = Agent(policy, trajectory, cache)

optimise!(agent::Agent) = optimise!(TrajectoryStyle(agent.trajectory), agent)
optimise!(::SyncTrajectoryStyle, agent::Agent) = optimise!(agent.policy, agent.trajectory)

# already spawn a task to optimise inner policy when initializing the agent
optimise!(::AsyncTrajectoryStyle, agent::Agent) = nothing

function optimise!(policy::AbstractPolicy, trajectory::Trajectory)
    for batch in trajectory
        optimise!(policy, batch)
    end
end

functor(x::Agent) = (policy = x.policy,), y -> Agent(y.policy, x.trajectory, x.cache)

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function (agent::Agent)(env::AbstractEnv)
    action = agent.policy(env)
    push!(agent.trajectory, (agent.cache..., action = action))
    agent.cache = NamedTuple()
    action
end

(agent::Agent)(::PreActStage, env::AbstractEnv) =
    agent.cache = (agent.cache..., state = state(env))

(agent::Agent)(::PostActStage, env::AbstractEnv) =
    agent.cache = (agent.cache..., reward = reward(env), terminal = is_terminated(env))
