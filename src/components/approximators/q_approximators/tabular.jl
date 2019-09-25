export TabularQApproximator, update!

using StatsBase: sample

"""
    struct TabularQApproximator <: AbstractQApproximator{Int, Int}
        table::Array{Float64, 2}
    end

Using a `table` of type `Array{Float64,2}` to record the action value of each state.
"""
struct TabularQApproximator <: AbstractQApproximator
    table::Array{Float64,2}
end

"""
    TabularQApproximator(ns::Int, na::Int=1, init::Float64=0.)

Initial a table of size `(ns, na)` filled with value of `init`.
"""
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