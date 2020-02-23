export Agent

"""
    Agent(;kwargs...)

One of the most commonly used [`AbstractAgent`](@ref).

Generally speaking, it does nothing but

1. Pass observation to the policy to generate an action
1. Update the trajectory using the `observation => action` pair
1. Update the policy with the newly updated trajectory

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between agent and environment
- `role=DEFAULT_PLAYER`: used to distinguish different agents
"""
Base.@kwdef mutable struct Agent{P<:AbstractPolicy,T<:AbstractTrajectory,R} <: AbstractAgent
    policy::P
    trajectory::T
    role::R = DEFAULT_PLAYER
end

RLBase.get_role(agent::Agent) = agent.role

#####
# EpisodicCompactSARTSATrajectory
#####
function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PreEpisodeStage,
    obs,
)
    empty!(agent.trajectory)
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PreActStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end

function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PostActStage,
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PostEpisodeStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end

#####
# Union{CircularCompactSARTSATrajectory, CircularCompactPSARTSATrajectory}
#####

function (
    agent::Agent{
        <:AbstractPolicy,
        <:Union{CircularCompactSARTSATrajectory,CircularCompactPSARTSATrajectory},
    }
)(
    ::PreEpisodeStage,
    obs,
)
    if length(agent.trajectory) > 0
        pop!(agent.trajectory, :state, :action)
    end
    nothing
end

function (
    agent::Agent{
        <:AbstractPolicy,
        <:Union{CircularCompactSARTSATrajectory,CircularCompactPSARTSATrajectory},
    }
)(
    ::PreActStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end

function (
    agent::Agent{
        <:AbstractPolicy,
        <:Union{CircularCompactSARTSATrajectory,CircularCompactPSARTSATrajectory},
    }
)(
    ::PostActStage,
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    nothing
end

function (
    agent::Agent{
        <:AbstractPolicy,
        <:Union{CircularCompactSARTSATrajectory,CircularCompactPSARTSATrajectory},
    }
)(
    ::PostEpisodeStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end

#####
# VectorialCompactSARTSATrajectory
#####

function (agent::Agent{<:AbstractPolicy,<:VectorialCompactSARTSATrajectory})(
    ::PreEpisodeStage,
    obs,
)
    if length(agent.trajectory) > 0
        pop!(agent.trajectory, :state, :action)
    end
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:VectorialCompactSARTSATrajectory})(
    ::PreActStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end

function (agent::Agent{<:AbstractPolicy,<:VectorialCompactSARTSATrajectory})(
    ::PostActStage,
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:VectorialCompactSARTSATrajectory})(
    ::PostEpisodeStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end
