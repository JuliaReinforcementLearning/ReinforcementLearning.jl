export Agent, @agent_str

using Flux
using BSON
using JLD
using FileIO
using Setfield

"""
    Agent(;kwargs...)

One of the most commonly used [`AbstractAgent`](@ref).

Generally speaking, it does nothing but update the trajectory and policy appropriately in different stages.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between an agent and an environment
- `role=:DEFAULT_PLAYER`: used to distinguish different agents
"""
Base.@kwdef mutable struct Agent{P<:AbstractPolicy,T<:AbstractTrajectory,R} <: AbstractAgent
    policy::P
    trajectory::T
    role::R = :DEFAULT_PLAYER
    is_training::Bool = true
end

# avoid polluting trajectory
(agent::Agent)(obs) = agent.policy(obs)

Flux.functor(x::Agent) = (policy = x.policy,), y -> @set x.policy = y.policy

function FileIO.save(dir::String, agent::Agent)
    mkpath(dir)

    policy = cpu(agent.policy)  # avoid gpu relate problems
    trajectory = agent.trajectory
    role = agent.role
    is_training = agent.is_training

    @info "saving agent to $dir ..."
    t = @elapsed begin
        BSON.@save joinpath(dir, "policy.bson") policy
        JLD.@save joinpath(dir, "trajectory.jld") trajectory
        BSON.@save joinpath(dir, "role.bson") role
        BSON.@save joinpath(dir, "is_training.bson") is_training
    end
    @info "finished saving agent in $t seconds"
end

macro agent_str(dir)
    Agent(dir)
end

"remember to call `gpu` if necessary"
function Agent(dir::String)
    @info "loading agent from $dir"
    BSON.@load joinpath(dir, "policy.bson") policy
    JLD.@load joinpath(dir, "trajectory.jld") trajectory
    BSON.@load joinpath(dir, "role.bson") role
    BSON.@load joinpath(dir, "is_training.bson") is_training
    Agent(policy, trajectory, role, is_training)
end

get_role(agent::Agent) = agent.role

function Flux.testmode!(agent::Agent, mode = true)
    agent.is_training = !mode
    testmode!(agent.policy, mode)
end

(agent::Agent)(stage::AbstractStage, obs) =
    agent.is_training ? agent(Training(stage), obs) : agent(Testing(stage), obs)

(agent::Agent)(::Testing, obs) = nothing
(agent::Agent)(::Testing{PreActStage}, obs) = agent.policy(obs)

#####
# EpisodicCompactSARTSATrajectory
#####
function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::Training{PreEpisodeStage},
    obs,
)
    empty!(agent.trajectory)
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::Training{PreActStage},
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end

function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::Training{PostActStage},
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::Training{PostEpisodeStage},
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
    ::Training{PreEpisodeStage},
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
    ::Training{PreActStage},
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
    ::Training{PostActStage},
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
    ::Training{PostEpisodeStage},
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
    ::Training{PreEpisodeStage},
    obs,
)
    if length(agent.trajectory) > 0
        pop!(agent.trajectory, :state, :action)
    end
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:VectorialCompactSARTSATrajectory})(
    ::Training{PreActStage},
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end

function (agent::Agent{<:AbstractPolicy,<:VectorialCompactSARTSATrajectory})(
    ::Training{PostActStage},
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    nothing
end

function (agent::Agent{<:AbstractPolicy,<:VectorialCompactSARTSATrajectory})(
    ::Training{PostEpisodeStage},
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.policy, agent.trajectory)
    action
end
