export TabularPolicy

using Flux: OneHotVector

"""
    TabularPolicy(table=Dict{Int,Int}(),n_action=nothing)

A `Dict` is used internally to store the mapping from state to action.
`n_action` is required if you want to calculate the probability of the
`TabularPolicy` given a state (`prob(p::TabularPolicy, s)`). 
"""
Base.@kwdef struct TabularPolicy{S,A} <: AbstractPolicy
    table::Dict{S,A} = Dict{Int,Int}()
    n_action::Union{Int, Nothing} = nothing
end

(p::TabularPolicy)(env::AbstractEnv) = p(state(env))
(p::TabularPolicy{S})(s::S) where {S} = p.table[s]

function RLBase.update!(p::TabularPolicy, target::Pair)
    p.table[first(target)] = last(target)
end

RLBase.prob(p::TabularPolicy, s, a) = p.table[s] == a ? 1.0 : 0.0

RLBase.prob(p::TabularPolicy, s) = OneHotVector(p.table[s], p.n_action)
