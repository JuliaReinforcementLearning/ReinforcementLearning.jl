export DynaAgent

Base.@kwdef struct DynaAgent{P,B,M,R} <: AbstractAgent
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