export Agent

using Base.Threads: @spawn
using Flux
import Base.push!

abstract type AbstractAgent <: AbstractPolicy end

"""
    Agent(;policy, trajectory) <: AbstractPolicy

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages. Agent
is a Callable and its call method accepts varargs and keyword arguments to be
passed to the policy. 

"""
mutable struct Agent{P,T} <: AbstractAgent
    policy::P
    trajectory::T

    function Agent(policy::P, trajectory::T) where {P<:AbstractPolicy, T<:Trajectory}
        agent = new{P,T}(policy, trajectory)

        if TrajectoryStyle(trajectory) === AsyncTrajectoryStyle()
            bind(trajectory, @spawn(optimise!(policy, trajectory)))
        end
        agent
    end
end

Agent(;policy, trajectory) = Agent(policy, trajectory)

RLBase.optimise!(agent::AbstractAgent, stage::S) where {S<:AbstractStage} = RLBase.optimise!(TrajectoryStyle(agent.trajectory), agent, stage)
RLBase.optimise!(::SyncTrajectoryStyle, agent::AbstractAgent, stage::S) where {S<:AbstractStage} = RLBase.optimise!(agent.policy, stage, agent.trajectory)

# already spawn a task to optimise inner policy when initializing the agent
RLBase.optimise!(::AsyncTrajectoryStyle, agent::AbstractAgent, stage::S) where {S<:AbstractStage} = nothing

#by default, optimise does nothing at all stages
function RLBase.optimise!(policy::AbstractPolicy, stage::AbstractStage, trajectory::Trajectory) end

Flux.@layer Agent trainable=(policy,)

function Base.push!(agent::Agent, ::PreEpisodeStage, env::AbstractEnv)
    push!(agent.trajectory, (state = state(env),))
end

# !!! TODO: In async scenarios, parameters of the policy may still be updating
# (partially), which will result to incorrect action. This should be addressed
# in Oolong.jl with a wrapper
function RLBase.plan!(agent::AbstractAgent, env::AbstractEnv)
    RLBase.plan!(agent.policy, env)
end

function Base.push!(agent::Agent, ::PostActStage, env::AbstractEnv, action)
    next_state = state(env)
    push!(agent.trajectory, (state = next_state, action = action, reward = reward(env), terminal = is_terminated(env)))
end

function Base.push!(agent::Agent, ::PostEpisodeStage, env::AbstractEnv)
    if haskey(agent.trajectory, :next_action) 
        action = RLBase.plan!(agent.policy, env)
        push!(agent.trajectory, PartialNamedTuple((action = action, )))
    end
end
