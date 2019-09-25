export TabularDeterministicPolicy

Base.@kwdef struct TabularDeterministicPolicy <: AbstractPolicy
    table::Vector{Int}
    nactions::Int
end

(p::TabularDeterministicPolicy)(s::Int) = p.table[s]
(p::TabularDeterministicPolicy)(obs::Observation) = p(get_state(obs))
get_prob(p::TabularDeterministicPolicy, s::Int, a::Int) = p.table[s] == a ? 1.0 : 0.0

function update!(p::TabularDeterministicPolicy, correction)
    s, a = correction
    p.table[s] = a
end