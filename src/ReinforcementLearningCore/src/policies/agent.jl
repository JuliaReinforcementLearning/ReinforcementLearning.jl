export Agent

using Base.Threads: @spawn

using Functors: @functor

"""
    Agent(;policy, trajectory) <: AbstractPolicy

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages. Agent
is a Callable and its call method accepts varargs and keyword arguments to be
passed to the policy. 

"""
mutable struct Agent{P,T} <: AbstractPolicy
    policy::P
    trajectory::T
    cache::NamedTuple # trajectory do not support partial inserting

    function Agent(policy::P, trajectory::T, cache=NamedTuple()) where {P,T}
        agent = new{P,T}(policy, trajectory, cache)
        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(; policy, trajectory, cache=NamedTuple()) = Agent(policy, trajectory, cache)

RLBase.optimise!(agent::Agent) = optimise!(TrajectoryStyle(agent.trajectory), agent)
RLBase.optimise!(::SyncTrajectoryStyle, agent::Agent) =
    optimise!(agent.policy, agent.trajectory)

# already spawn a task to optimise inner policy when initializing the agent
RLBase.optimise!(::AsyncTrajectoryStyle, agent::Agent) = nothing

function RLBase.optimise!(policy::AbstractPolicy, trajectory::Trajectory)
    for batch in trajectory
        optimise!(policy, batch)
    end
end

@functor Agent (policy,)

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function (agent::Agent)(env::AbstractEnv, args...; kwargs...)
    action = agent.policy(env, args...; kwargs...)
    push!(agent.trajectory, (agent.cache..., action=action))
    agent.cache = (;)
    action
end

(agent::Agent)(::PreActStage, env::AbstractEnv) =
    agent.cache = (agent.cache..., state=state(env))

(agent::Agent)(::PostActStage, env::AbstractEnv) =
    agent.cache = (agent.cache..., reward=reward(env), terminal=is_terminated(env))
