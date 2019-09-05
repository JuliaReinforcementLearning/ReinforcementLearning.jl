export Agent, update!

mutable struct Agent{P, B} <: AbstractAgent
    π::P
    buffer::B
end

(agent::Agent)(obs::Observation) = agent.π(obs)

function update!(agent::Agent, experience::Pair)
    push!(agent.buffer, experience)
    transitions = extract_transitions(buffer, agent.π)
    if !isnothing(trainsitions)
        update!(agent.π, transitions)
    end
end

function update!(agent::Agent{<:QBasedPolicy{<:PrioritizedDQNLearner}}, experience::Pair)
    push!(agent.buffer, experience)
    transitions = extract_transitions(buffer, agent.π)
    if !isnothing(trainsitions)
        inds, priorities = update!(agent.π, transitions)
        isnothing(priorities) || (priority(agent.buffer)[inds] .= priorities)
    end
end