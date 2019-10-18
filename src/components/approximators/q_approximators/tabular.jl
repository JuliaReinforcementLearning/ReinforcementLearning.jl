export TabularQApproximator, update!

using StatsBase: sample

"""
    TabularQApproximator(table::Array{Float64, 2})
    TabularQApproximator(ns::Int, na::Int=1, init::Float64=0.)

Use a `table` of type `Array{Float64,2}` to store the action value of each state.
`ns` is the number of states. `na` is the number of actions. `init` is the initial value of `table`.
"""
struct TabularQApproximator <: AbstractQApproximator
    table::Array{Float64,2}
end

TabularQApproximator(; n_state::Int, n_action::Int = 1, init::Float64 = 0.0) =
    TabularQApproximator(fill(init, n_state, n_action))

(Q::TabularQApproximator)(s::Int) = @view(Q.table[s, :])

(Q::TabularQApproximator)(s::Int, a::Int) = Q.table[s, a]

function update!(Q::TabularQApproximator, correction::Pair{Int,Vector{Float64}})
    s, errors = correction
    for (i, e) in enumerate(errors)
        Q.table[s, i] += e
    end
end

function update!(Q::TabularQApproximator, correction::Pair{Tuple{Int,Int},Float64})
    (s, a), e = correction
    Q.table[s, a] += e
end