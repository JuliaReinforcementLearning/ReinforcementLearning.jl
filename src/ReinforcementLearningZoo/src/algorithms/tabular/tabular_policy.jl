export TabularPolicy

using Flux: onehot

Base.@kwdef struct TabularPolicy{S,A} <: AbstractPolicy
    table::Dict{S,A} = Dict{Int, Int}()
    n_action::Int
end

(p::TabularPolicy)(env::AbstractEnv) = p(state(env))
(p::TabularPolicy{S})(s::S) where S = p.table[s]

function RLBase.update!(p::TabularPolicy, target::Pair)
    p.table[first(target)] = last(target)
end

RLBase.prob(p::TabularPolicy, s, a) = p.table[s] == a ? 1.0 : 0.