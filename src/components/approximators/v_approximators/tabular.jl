export TabularV

"""
    struct TabularV <: AbstractVApproximator{Int}
        table::Vector{Float64}
    end

Using a `table` of type `Vector{Float64}` to record the state values.
"""
struct TabularV <: AbstractVApproximator{Int}
    table::Vector{Float64}
end

TabularV(ns::Int, init::Float64=0.) = TabularV(fill(init, ns))

(v::TabularV)(s::Int) = v.table[s]

function update!(v::TabularV, correction::Pair{Int, Float64})
    s, e = correction
    v.table[s] += e
end
