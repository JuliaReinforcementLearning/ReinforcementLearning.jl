export TabularApproximator

"""
    TabularApproximator(table::Vector{Float64}) -> TabularApproximator
    TabularApproximator(;n_state::Int, init::Float64=0.0) -> TabularApproximator

Use a `table` of type `Vector{Float64}` of length `ns` to record the state values.
"""
struct TabularApproximator <: AbstractApproximator
    table::Vector{Float64}
end

TabularApproximator(;n_state::Int, init::Float64 = 0.0) = TabularApproximator(fill(init, n_state))

(v::TabularApproximator)(s::Int) = v.table[s]

function RLBase.update!(v::TabularApproximator, correction::Pair{Int,Float64})
    s, e = correction
    v.table[s] += e
end