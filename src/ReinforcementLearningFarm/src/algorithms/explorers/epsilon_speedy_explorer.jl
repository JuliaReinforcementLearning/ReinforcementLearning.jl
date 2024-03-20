
using ReinforcementLearningCore
using ReinforcementLearningBase
import ReinforcementLearningBase: RLBase

using FillArrays: Trues
using Random
using Distributions: Categorical
using Base

# Epsilon Greedy Explorer for AIAPC Zoo
# Note: get_ϵ function in RLCore takes: 600.045 ns (6 allocations: 192 bytes)
# This one has: 59.003 ns (1 allocation: 16 bytes)
# Well worth looking into optimizations for RLCore
# TODO evaluate performance cost of checking all values for max, perhaps only do this in the beginning?

struct EpsilonSpeedyExplorer{R} <: AbstractExplorer
    β::Float64
    β_neg::Float64
    step::Base.RefValue{Int}
    rng::R
end

function EpsilonSpeedyExplorer(β::Float64)
    EpsilonSpeedyExplorer{typeof(Random.GLOBAL_RNG)}(
        β,
        β * -1,
        Ref(1),
        Random.GLOBAL_RNG,
    )
end

function get_ϵ(s::EpsilonSpeedyExplorer)
    exp(s.β_neg * s.step[])
end

"""
    RLBase.plan!(s::EpsilonSpeedyExplorer, values; step) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned when `is_break_tie==true`.

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
function RLBase.plan!(s::EpsilonSpeedyExplorer{R}, values::A) where {I<:Real, A<:AbstractArray{I}, R<:Random.AbstractRNG}
    ϵ = get_ϵ(s)
    s.step[] += 1
    rand(s.rng) >= ϵ ? findmax(values)[2] : rand(s.rng, 1:length(values))
end

RLBase.plan!(s::EpsilonSpeedyExplorer{R}, x::A, mask::Trues) where {I<:Real, A<:AbstractArray{I}, R<:Random.AbstractRNG} = RLBase.plan!(s, x)

function RLBase.plan!(s::EpsilonSpeedyExplorer{R}, values::A, mask::M) where {I<:Real, A<:AbstractArray{I}, M<:Union{BitVector, Vector{Bool}}, R<:Random.AbstractRNG}
    ϵ = get_ϵ(s)
    s.step[] += 1
    # NOTE: takes first max element, doesn't break ties randomly
    rand(s.rng) >= ϵ ? RLCore.findmax_masked(values, mask)[2] : rand(s.rng, findall(mask))
end

"""
    prob(s::EpsilonGreedyExplorer, values) -> Categorical
    prob(s::EpsilonGreedyExplorer, values, mask) -> Categorical

Return the probability of selecting each action given the estimated `values` of each action.
"""
function RLBase.prob(s::EpsilonSpeedyExplorer, values::A) where {I<:Real, A<:AbstractArray{I}}
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    probs[findmax(values)[2]] += 1 - ϵ
    Categorical(probs; check_args=false)
end

function RLBase.prob(s::EpsilonSpeedyExplorer, values::A, action::Integer) where {I<:Real, A<:AbstractArray{I}}
    ϵ, n = get_ϵ(s), length(values)
    if action == findmax(values)[2]
        ϵ / n + 1 - ϵ
    else
        ϵ / n
    end
end

function RLBase.prob(s::EpsilonSpeedyExplorer, values::A, mask::M) where {I<:Real, A<:AbstractArray{I}, M<:Union{BitVector, Vector{Bool}}}
    ϵ, n = get_ϵ(s), length(values)
    probs = zeros(n)
    probs[mask] .= ϵ / sum(mask)
    probs[RLCore.findmax_masked(values, mask)[2]] += 1 - ϵ
    Categorical(probs; check_args=false)
end
