export PrioritizedSweepingSamplingModel

using DataStructures: PriorityQueue, dequeue!

import StatsBase: sample

"""
    PrioritizedSweepingSamplingModel(θ::Float64=1e-4)
See more details at Section (8.4) on Page 168 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
mutable struct PrioritizedSweepingSamplingModel <: AbstractEnvironmentModel
    experiences::Dict{Tuple{Any,Any},Tuple{Float64,Bool,Any}}
    PQueue::PriorityQueue{Tuple{Any,Any},Float64}
    predecessors::Dict{Any,Set{Tuple{Any,Any,Float64,Bool}}}
    θ::Float64
    sample_count::Int
    PrioritizedSweepingSamplingModel(θ::Float64 = 1e-4) = new(
        Dict{Tuple{Any,Any},Tuple{Float64,Bool,Any}}(),
        PriorityQueue{Tuple{Any,Any},Float64}(Base.Order.Reverse),
        Dict{Any,Set{Tuple{Any,Any,Float64,Bool}}}(),
        θ,
        0,
    )
end

function RLBase.update!(
    m::PrioritizedSweepingSamplingModel,
    t::AbstractTrajectory,
    p::AbstractPolicy,
    ::AbstractEnv,
    ::Union{PreActStage,PostEpisodeStage},
)
    if length(t[:terminal]) > 0
        transition = (
            t[:state][end-1],
            t[:action][end-1],
            t[:reward][end],
            t[:terminal][end],
            t[:state][end],
        )
        pri = RLBase.priority(p, transition)
        update!(m, (transition..., pri))
    end
end

function RLBase.update!(m::PrioritizedSweepingSamplingModel, transition::Tuple)
    s, a, r, d, s′, P = transition
    m.experiences[(s, a)] = (r, d, s′)
    if P >= m.θ
        m.PQueue[(s, a)] = P
    end
    if !haskey(m.predecessors, s′)
        m.predecessors[s′] = Set{Tuple{Any,Any,Float64,Bool}}()
    end
    push!(m.predecessors[s′], (s, a, r, d))
end

function sample(m::PrioritizedSweepingSamplingModel)
    if length(m.PQueue) > 0
        s, a = dequeue!(m.PQueue)
        r, d, s′ = m.experiences[(s, a)]
        m.sample_count += 1
        s, a, r, d, s′
    else
        nothing
    end
end
