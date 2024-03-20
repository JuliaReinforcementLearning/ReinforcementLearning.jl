export TabularApproximator, TabularVApproximator, TabularQApproximator

struct TabularApproximator{A} <: AbstractLearner where {A<:AbstractArray}
    model::A
end

const TabularQApproximator = TabularApproximator{A} where {A<:AbstractMatrix}
const TabularVApproximator = TabularApproximator{A} where {A<:AbstractVector}

"""
    TabularApproximator(table<:AbstractArray)

For `table` of 1-d, it will serve as a state value approximator.
For `table` of 2-d, it will serve as a state-action value approximator.

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
function TabularApproximator(table::A) where {A<:AbstractArray}
    n = ndims(table)
    n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
    TabularApproximator{A}(table)
end

TabularVApproximator(; n_state, init = 0.0) =
    TabularApproximator(fill(init, n_state))

TabularQApproximator(; n_state, n_action, init = 0.0) =
    TabularApproximator(fill(init, n_action, n_state))

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
forward(L::TabularVApproximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(L, x))
forward(L::TabularQApproximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(L, x))

RLCore.forward(
    app::TabularVApproximator{R},
    s::I,
) where {R<:AbstractVector,I} = @views app.model[s]

RLCore.forward(
    app::TabularQApproximator{R},
    s::I,
) where {R<:AbstractArray,I} = @views app.model[:, s]

RLCore.forward(
    app::TabularQApproximator{R},
    s::I1,
    a::I2,
) where {R<:AbstractArray,I1,I2} = @views app.model[a, s]
