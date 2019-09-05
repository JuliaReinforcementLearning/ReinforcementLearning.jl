export Agent, update!

mutable struct Agent{P, B, R} <: AbstractAgent
    π::P
    buffer::B
    role::R
end

Agent(π, buffer; role=:DEFAULT) = Agent(π, buffer, role)

(agent::Agent)(obs::Observation) = agent.π(obs)

function update!(agent::Agent, experience::Pair)
    push!(agent.buffer, experience)
    transitions = extract_transitions(agent.buffer, agent.π)
    if !isnothing(transitions)
        update!(agent.π, transitions)
    end
end

function update!(agent::Agent{<:QBasedPolicy{<:Union{PrioritizedDQNLearner, RainbowLearner}}}, experience::Pair)
    push!(priority(agent.buffer), agent.π.learner.default_priority)
    push!(agent.buffer, experience)
    indexed_batch = extract_transitions(agent.buffer, agent.π)
    if !isnothing(indexed_batch)
        inds, batch = indexed_batch
        priorities = update!(agent.π, batch)
        isnothing(priorities) || (priority(agent.buffer)[inds] .= priorities)
    end
end