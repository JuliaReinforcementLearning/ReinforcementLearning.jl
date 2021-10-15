export TabularApproximator, TabularVApproximator, TabularQApproximator

"""
    TabularApproximator(table<:AbstractArray, opt)

For `table` of 1-d, it will serve as a state value approximator. See [`TabularVApproximator`](@ref).
For `table` of 2-d, it will serve as a state-action value approximator. See [`TabularQApproximator`](@ref).

Note that actions and states should be presented to `TabularApproximator` as integers starting from 
1 to be used as the index of the table. That is, e.g., [`RLBase.state_space`](@ref) is expected to 
return `Base.OneTo(n_state)`, where `n_state` is the number of states.

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
struct TabularApproximator{N,T<:AbstractArray{<:AbstractFloat, N},O} <: AbstractApproximator
    table::T
    optimizer::O
    function TabularApproximator(table::T, opt::O) where {T<:AbstractArray,O}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
        new{n,T,O}(table, opt)
    end
end

const TabularVApproximator = TabularApproximator{1}
const TabularQApproximator = TabularApproximator{2}

"""
    TabularVApproximator(; n_state, init = 0.0, opt = InvDecay(1.0))

A state value approximator represented by a 1-d table. `init` is the initial value of each state.
"""
TabularVApproximator(; n_state, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_state), opt)
"""
    TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0))

An action-state value approximator represented by a 2-d table. `init` is the initial value of each
pair of action-state.
"""
TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_action, n_state), opt)

(app::TabularVApproximator)(s::Int) = @views app.table[s]

(app::TabularQApproximator)(s::Int) = @views app.table[:, s]
(app::TabularQApproximator)(s::Int, a::Int) = app.table[a, s]

function RLBase.update!(app::TabularVApproximator, correction::Pair{Int,Float64})
    s, e = correction
    x = @view app.table[s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.update!(app::TabularQApproximator, correction::Pair{Tuple{Int,Int},Float64})
    (s, a), e = correction
    x = @view app.table[a, s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.update!(app::TabularQApproximator, correction::Pair{Int,Vector{Float64}})
    s, errors = correction
    x = @view app.table[:, s]
    Flux.Optimise.update!(app.optimizer, x, errors)
end
