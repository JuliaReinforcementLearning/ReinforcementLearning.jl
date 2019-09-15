export Agent

Base.@kwdef mutable struct Agent{P,B,R} <: AbstractAgent
    π::P
    buffer::B
    role::R = :DEFAULT
end

Agent(π, buffer; role = :DEFAULT) = Agent(π, buffer, role)

(agent::Agent)(obs::Observation) = agent.π(obs)

function update!(agent::Agent, experience::Pair)
    update!(agent.buffer, experience, agent)
    update!(agent.π, agent.buffer)
end

function update!(buffer::AbstractTurnBuffer, experience::Pair, agent::Agent)
    push!(agent.buffer, experience)
end

function update!(
    buffer::AbstractTurnBuffer,
    experience::Pair,
    agent::Agent{<:QBasedPolicy{<:Union{PrioritizedDQNLearner,RainbowLearner}}},
)
    push!(priority(agent.buffer), agent.π.learner.default_priority)
    push!(agent.buffer, experience)
end