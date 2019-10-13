export DynaAgent

"""
    DynaAgent(;kwargs...)

`DynaAgent` is first introduced in: *Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting." ACM Sigart Bulletin 2.4 (1991): 160-163.*

# Keywords & Fields

- `π`::[`AbstractPolicy`](@ref): the policy to use
- `model`::[`AbstractEnvironmentModel`](@ref): describe the environment to interact with
- `buffer`::[`AbstractTurnBuffer`](@ref): used to store transitions between agent and environment
- `role=:DEFAULT`: used to distinguish different agents
- `plan_step::Int=10`: the count of planning steps

The main difference between [`DynaAgent`](@ref) and [`Agent`](@ref) is that an environment model is involved. It is best described in the book: *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*

![](../assets/img/RL_book_fig_8_1.png)

![](../assets/img/RL_book_fig_8_2.png)
"""
Base.@kwdef struct DynaAgent{P<:AbstractPolicy, B<:AbstractTurnBuffer, M<:AbstractEnvironmentModel, R} <: AbstractAgent
    π::P
    model::M
    buffer::B
    role::R = :DEFAULT
    plan_step::Int = 10
end

(agent::DynaAgent)(obs::Observation) = agent.π(obs)

function update!(agent::DynaAgent, experience::Pair)
    π, model, buffer = agent.π, agent.model, agent.buffer

    push!(buffer, experience)
    update!(model, buffer, π)  # model learning
    update!(π, buffer, model; plan_step = agent.plan_step)
end

update!(model::AbstractEnvironmentModel, buffer::AbstractTurnBuffer, π::AbstractPolicy) =
    update!(model, buffer)

function update!(model::AbstractEnvironmentModel, buffer::AbstractTurnBuffer)
    transitions = extract_transitions(buffer, model)
    if !isnothing(transitions)
        update!(model, transitions)
    end
end

function update!(
    model::PrioritizedSweepingSampleModel,
    buffer::AbstractTurnBuffer,
    π::AbstractPolicy,
)
    transition = extract_transitions(buffer, model)
    if !isnothing(transition)
        update!(model, transition, priority(transition, π))
    end
end

function update!(
    π::AbstractPolicy,
    buffer::AbstractTurnBuffer,
    model::AbstractEnvironmentModel;
    plan_step = 1,
)
    update!(π, buffer)  # direct RL
    update!(π, model; plan_step = plan_step)  # planning
end