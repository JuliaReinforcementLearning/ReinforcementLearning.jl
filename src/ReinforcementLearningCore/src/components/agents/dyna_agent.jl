export DynaAgent

Base.@kwdef struct DynaAgent{P<:AbstractPolicy, B<:AbstractTrajectory, M<:AbstractEnvironmentModel, R} <: AbstractAgent
    policy::P
    model::M
    trajectory::B
    role::R = DEFAULT_PLAYER
    plan_step::Int = 10
end

RLBase.get_role(agent::DynaAgent) = agent.role

function (agent::DynaAgent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PreEpisodeStage,
    obs,
)
    empty!(agent.trajectory)
    nothing
end

function (agent::DynaAgent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PreActStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.model, agent.trajectory, agent.policy)  # model learning
    update!(agent.policy, agent.trajectory)  # direct learning
    update!(agent.policy, agent.model, agent.trajectory, agent.plan_step)  # policy learning
    action
end

function (agent::DynaAgent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PostActStage,
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    nothing
end

function (agent::DynaAgent{<:AbstractPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PostEpisodeStage,
    obs,
)
    action = agent.policy(obs)
    push!(agent.trajectory; state = get_state(obs), action = action)
    update!(agent.model, agent.trajectory, agent.policy)  # model learning
    update!(agent.policy, agent.trajectory)  # direct learning
    update!(agent.policy, agent.model, agent.trajectory, agent.plan_step)  # policy learning
    action
end

"By default, only use trajectory to update model"
RLBase.update!(model::AbstractEnvironmentModel, t::AbstractTrajectory, π::AbstractPolicy) =
    update!(model, t)

function RLBase.update!(model::AbstractEnvironmentModel, buffer::AbstractTrajectory)
    transitions = extract_experience(buffer, model)
    isnothing(transitions) || update!(model, transitions)
end