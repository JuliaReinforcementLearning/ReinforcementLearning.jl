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
- `buffer`::[`AbstractTrajectory`](@ref): used to store transitions between agent and environment
- `role=:DEFAULT`: used to distinguish different agents
"""
Base.@kwdef mutable struct Agent{P<:AbstractPolicy, B<:AbstractTrajectory, R} <: AbstractAgent
    π::P
    buffer::B
    role::R = DEFAULT_PLAYER
end

function (agent::Agent)(obs)
    action = agent.π(obs)
    push!(agent.buffer, obs => action)
    update!(agent.π, agent.buffer)
    action
end