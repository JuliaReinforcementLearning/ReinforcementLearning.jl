export TabularApproximator

"""
    TabularApproximator(table<:AbstractArray)

For `table` of 1-d, it will create a [`VApproximator`](@ref). For `table` of 2-d, it will create a [`QApproximator`].

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
struct TabularApproximator{N,T<:AbstractArray} <: AbstractApproximator
    table::T
    function TabularApproximator(table::T) where {T<:AbstractArray}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimention of table must be <= 2"))
        new{n,T}(table)
    end
end

function TabularApproximator(; n_state, n_action = nothing, init = 0.0)
    table = isnothing(n_action) ? fill(init, n_state) : fill(init, n_action, n_state)
    TabularApproximator(table)
end

(app::TabularApproximator{1})(s::Int) = @views app.table[s]

(app::TabularApproximator{2})(s::Int) = @views app.table[:, s]
(app::TabularApproximator{2})(s::Int, a::Int) = app(s)[a]

function RLBase.update!(app::TabularApproximator{1}, correction::Pair)
    s, e = correction
    app.table[s] += e
end

function RLBase.update!(app::TabularApproximator{2}, correction::Pair)
    (s, a), e = correction
    app.table[a, s] += e
end

function RLBase.update!(Q::TabularApproximator{2}, correction::Pair{Int,Vector{Float64}})
    s, errors = correction
    for (a, e) in enumerate(errors)
        Q.table[a, s] += e
    end
end

RLBase.ApproximatorStyle(::TabularApproximator{1}) = VApproximator()
RLBase.ApproximatorStyle(::TabularApproximator{2}) = QApproximator()
