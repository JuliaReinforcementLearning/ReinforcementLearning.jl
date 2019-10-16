export PrioritizedSweepingSampleModel, update!, sample

using DataStructures

import StatsBase: sample

"""
    PrioritizedSweepingSampleModel(θ::Float64=1e-4)

See more details at Section (8.4) on Page 168 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
mutable struct PrioritizedSweepingSampleModel <: AbstractSampleBasedModel
    experiences::Dict{Tuple{Any,Any},Tuple{Float64,Bool,Any}}
    PQueue::PriorityQueue{Tuple{Any,Any},Float64}
    predecessors::Dict{Any,Set{Tuple{Any,Any,Float64,Bool}}}
    θ::Float64
    sample_count::Int
    PrioritizedSweepingSampleModel(θ::Float64 = 1e-4) =
        new(
            Dict{Tuple{Any,Any},Tuple{Float64,Bool,Any}}(),
            PriorityQueue{Tuple{Any,Any},Float64}(Base.Order.Reverse),
            Dict{Any,Set{Tuple{Any,Any,Float64,Bool}}}(),
            θ,
            0,
        )
end

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    model::PrioritizedSweepingSampleModel,
)
    if length(buffer) > 0
        state(buffer)[end-1],
        action(buffer)[end-1],
        reward(buffer)[end],
        terminal(buffer)[end],
        state(buffer)[end]
    else
        nothing
    end
end

function update!(m::PrioritizedSweepingSampleModel, transition, P)
    s, a, r, d, s′ = transition
    m.experiences[(s, a)] = (r, d, s′)
    if P >= m.θ
        m.PQueue[(s, a)] = P
    end
    if !haskey(m.predecessors, s′)
        m.predecessors[s′] = Set{Tuple{Any,Any,Float64,Bool}}()
    end
    push!(m.predecessors[s′], (s, a, r, d))
end

function sample(m::PrioritizedSweepingSampleModel)
    if length(m.PQueue) > 0
        s, a = dequeue!(m.PQueue)
        r, d, s′ = m.experiences[(s, a)]
        m.sample_count += 1
        s, a, r, d, s′
    else
        nothing
    end
end