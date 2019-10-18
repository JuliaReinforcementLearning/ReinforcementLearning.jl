export Agent

"""
    Agent(;kwargs...)

One of the most commonly used [`AbstractAgent`](@ref).

Generally speaking, it does nothing but

1. Pass observation to the policy to generate an action
1. Update the buffer using the `observation => action` pair
1. Update the policy with the newly updated buffer

# Keywords & Fields

- `π`::[`AbstractPolicy`](@ref): the policy to use
- `buffer`::[`AbstractTurnBuffer`](@ref): used to store transitions between agent and environment
- `role=:DEFAULT`: used to distinguish different agents
"""
Base.@kwdef mutable struct Agent{P<:AbstractPolicy, B<:AbstractTurnBuffer, R} <: AbstractAgent
    π::P
    buffer::B
    role::R = :DEFAULT
end

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