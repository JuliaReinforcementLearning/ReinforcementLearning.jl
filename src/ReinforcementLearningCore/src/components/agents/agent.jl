export Agent

using Flux
using BSON
using JLD
using Setfield

"""
    Agent(;kwargs...)

One of the most commonly used [`AbstractAgent`](@ref).

Generally speaking, it does nothing but update the trajectory and policy appropriately in different stages.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between an agent and an environment
- `role=RLBase.DEFAULT_PLAYER`: used to distinguish different agents
"""
Base.@kwdef mutable struct Agent{P<:AbstractPolicy,T<:AbstractTrajectory,R} <: AbstractAgent
    policy::P
    trajectory::T = DUMMY_TRAJECTORY
    role::R = RLBase.DEFAULT_PLAYER
    is_training::Bool = true
end

# avoid polluting trajectory
(agent::Agent)(env) = agent.policy(env)

Flux.functor(x::Agent) = (policy = x.policy,), y -> @set x.policy = y.policy

function save(dir::String, agent::Agent; is_save_trajectory = true)
    mkpath(dir)
    @info "saving agent to $dir ..."

    t = @elapsed begin
        save(joinpath(dir, "policy.bson"), agent.policy)
        if is_save_trajectory
            JLD.save(joinpath(dir, "trajectory.jld"), "trajectory", agent.trajectory)
        else
            @warn "trajectory is skipped since you set `is_save_trajectory` to false"
        end
        BSON.bson(
            joinpath(dir, "agent_meta.bson"),
            Dict(
                :role => agent.role,
                :is_training => agent.is_training,
                :policy_type => typeof(agent.policy),
            ),
        )
    end

    @info "finished saving agent in $t seconds"
end

function load(dir::String, ::Type{<:Agent})
    @info "loading agent from $dir"
    BSON.@load joinpath(dir, "agent_meta.bson") role is_training policy_type
    policy = load(joinpath(dir, "policy.bson"), policy_type)
    JLD.@load joinpath(dir, "trajectory.jld") trajectory
    Agent(policy, trajectory, role, is_training)
end

get_role(agent::Agent) = agent.role

function Flux.testmode!(agent::Agent, mode = true)
    agent.is_training = !mode
    testmode!(agent.policy, mode)
end

(agent::Agent)(stage::AbstractStage, env) =
    agent.is_training ? agent(Training(stage), env) : agent(Testing(stage), env)

(agent::Agent)(::Testing, env) = nothing
(agent::Agent)(::Testing{PreActStage}, env) = agent.policy(env)

#####
# DummyTrajectory
#####

(agent::Agent{<:AbstractPolicy,<:DummyTrajectory})(stage::AbstractStage, env) = nothing
(agent::Agent{<:AbstractPolicy,<:DummyTrajectory})(stage::PreActStage, env) =
    agent.policy(env)

#####
# default behavior
#####

function (agent::Agent)(::Training{PreEpisodeStage}, env)
    if nframes(agent.trajectory[:full_state]) > 0
        pop!(agent.trajectory, :full_state)
    end
    if nframes(agent.trajectory[:full_action]) > 0
        pop!(agent.trajectory, :full_action)
    end
    if ActionStyle(env) === FULL_ACTION_SET &&
       nframes(agent.trajectory[:full_legal_actions_mask]) > 0
        pop!(agent.trajectory, :full_legal_actions_mask)
    end
end

function (agent::Agent)(::Training{PreActStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory)
    action
end

function (agent::Agent)(::Training{PostActStage}, env)
    push!(agent.trajectory; reward = get_reward(env), terminal = get_terminal(env))
    nothing
end

function (agent::Agent)(::Training{PostEpisodeStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory)
    action
end

#####
# EpisodicCompactSARTSATrajectory
#####
function (agent::Agent{<:AbstractPolicy,<:EpisodicTrajectory})(
    ::Training{PreEpisodeStage},
    env,
)
    empty!(agent.trajectory)
    nothing
end
