export TabularQ, update!

using StatsBase:sample

"""
    struct TabularQ <: AbstractQApproximator{Int, Int}
        table::Array{Float64, 2}
    end

Using a `table` of type `Array{Float64,2}` to record the action value of each state.
"""
struct TabularQ <: AbstractQApproximator{Int}
    table::Array{Float64, 2}
end

"""
    TabularQ(ns::Int, na::Int=1, init::Float64=0.)

Initial a table of size `(ns, na)` filled with value of `init`.
"""
TabularQ(ns::Int, na::Int=1, init::Float64=0.) = TabularQ(fill(init, ns, na))

(Q::TabularQ)(s::Int) = @view(Q.table[s, :])

(Q::TabularQ)(s::Int, a::Int) = Q.table[s, a]

function update!(Q::TabularQ, correction::Pair{Int, Vector{Float64}})
    s, errors = correction
    for (i, e) in enumerate(errors)
        Q.table[s, i] += e
    end
end

function update!(Q::TabularQ, correction::Pair{Tuple{Int, Int}, Float64})
    (s, a), e = correction
    Q.table[s, a] += e
end