export DynaAgent

"""
    DynaAgent(;kwargs...)

`DynaAgent` is first introduced in: *Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting." ACM Sigart Bulletin 2.4 (1991): 160-163.*

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `model`::[`AbstractEnvironmentModel`](@ref): describe the environment to interact with
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between agent and environment
- `role=:DEFAULT`: used to distinguish different agents
- `plan_step::Int=10`: the count of planning steps

The main difference between [`DynaAgent`](@ref) and [`Agent`](@ref) is that an environment model is involved. It is best described in the book: *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*

![](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/img/RL_book_fig_8_1.png)
![](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/img/RL_book_fig_8_2.png)
"""
Base.@kwdef struct DynaAgent{
    P<:AbstractPolicy,
    B<:AbstractTrajectory,
    M<:AbstractEnvironmentModel,
    R,
} <: AbstractAgent
    policy::P
    model::M
    trajectory::B
    role::R = :DEFAULT_PLAYER
    plan_step::Int = 10
end

get_role(agent::DynaAgent) = agent.role

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
