export Agent

using Base.Threads: @spawn

using Functors: @functor
import Base.push!
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

    function Agent(policy::P, trajectory::T) where {P,T}
        agent = new{P,T}(policy, trajectory)

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end

    function Agent(policy::P, trajectory::T) where {P,T,C}
        agent = new{P,T,C}(policy, trajectory)

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(;policy, trajectory) = Agent(policy, trajectory)

RLBase.optimise!(agent::Agent, stage::S) where {S<:AbstractStage} = RLBase.optimise!(TrajectoryStyle(agent.trajectory), agent, stage)
RLBase.optimise!(::SyncTrajectoryStyle, agent::Agent, stage::S) where {S<:AbstractStage} = RLBase.optimise!(agent.policy, stage, agent.trajectory)

# already spawn a task to optimise inner policy when initializing the agent
RLBase.optimise!(::AsyncTrajectoryStyle, agent::Agent, stage::S) where {S<:AbstractStage} = nothing

#by default, optimise does nothing at all stage
function RLBase.optimise!(policy::AbstractPolicy, stage::AbstractStage, trajectory::Trajectory) end

@functor Agent (policy,)

function Base.push!(agent::Agent, ::PreEpisodeStage, env::AbstractEnv)
    push!(agent.trajectory, (state = state(env),))
end

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function RLBase.plan!(agent::Agent{P,T,C}, env::AbstractEnv) where {P,T,C}
    RLBase.plan!(agent.policy, env)
end

# Multiagent Version
function RLBase.plan!(agent::Agent{P,T,C}, env::E, p::Symbol) where {P,T,C,E<:AbstractEnv}
    action = RLBase.plan!(agent.policy, env, p)
    push!(agent.trajectory, agent.cache, action)
    action
end

function Base.push!(agent::Agent{P,T,C}, ::PostActStage, env::E, action) where {P,T,C,E<:AbstractEnv}
    next_state = state(env)
    push!(agent.trajectory, (state = next_state, action = action, reward = reward(env), terminal = is_terminated(env)))
end
