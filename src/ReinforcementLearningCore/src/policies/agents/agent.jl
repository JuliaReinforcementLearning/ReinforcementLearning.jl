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
Base.@kwdef struct Agent{P<:AbstractPolicy,T<:AbstractTrajectory} <: AbstractPolicy
    policy::P
    trajectory::T
end

functor(x::Agent) = (policy = x.policy,), y -> @set x.policy = y.policy

(agent::Agent)(env) = agent.policy(env)

function (agent::Agent)(stage::AbstractStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    update!(agent.policy, agent.trajectory, env, stage)
end

function (agent::Agent)(stage::PreActStage, env::AbstractEnv)
    action = update!(agent.trajectory, agent.policy, env, stage)
    update!(agent.policy, agent.trajectory, env, stage)
    action
end

RLBase.update!(::AbstractPolicy, ::AbstractTrajectory, ::AbstractEnv, ::AbstractStage) = nothing
RLBase.update!(p::AbstractPolicy, t::AbstractTrajectory, ::AbstractEnv, ::PreActStage) = update!(p, t)

## update trajectory

RLBase.update!(::AbstractTrajectory, ::AbstractPolicy, ::AbstractEnv, ::AbstractStage) = nothing

function RLBase.update!(
    trajectory::Union{CircularArraySARTTrajectory, PrioritizedTrajectory{<:CircularArraySARTTrajectory}},
    ::AbstractPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    if length(trajectory) > 0
        pop!(trajectory[:state])
        pop!(trajectory[:action])
    end
end

function RLBase.update!(
    trajectory::Union{CircularArraySLARTTrajectory, PrioritizedTrajectory{<:CircularArraySLARTTrajectory}},
    ::AbstractPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    if length(trajectory) > 0
        pop!(trajectory[:state])
        pop!(trajectory[:action])
        pop!(trajectory[:legal_actions_mask])
    end
end

function RLBase.update!(
    trajectory::Union{CircularArraySARTTrajectory,PrioritizedTrajectory{<:CircularArraySARTTrajectory}},
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::Union{PreActStage, PostEpisodeStage},
)
    action = policy(env)
    push!(trajectory[:state], get_state(env))
    push!(trajectory[:action], action)
    action
end

function RLBase.update!(
    trajectory::Union{CircularArraySLARTTrajectory,PrioritizedTrajectory{<:CircularArraySLARTTrajectory}},
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::Union{PreActStage, PostEpisodeStage},
)
    action = policy(env)
    push!(trajectory[:state], get_state(env))
    push!(trajectory[:action], action)
    push!(trajectory[:legal_actions_mask], get_legal_actions_mask(env))
    action
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    ::AbstractPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    push!(trajectory[:reward], get_reward(env))
    push!(trajectory[:terminal], get_terminal(env))
end
