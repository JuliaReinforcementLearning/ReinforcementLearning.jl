export TabularApproximator, TabularVApproximator, TabularQApproximator

const TabularApproximator = Approximator{A,O} where {A<:AbstractArray,O}
const TabularQApproximator = Approximator{A,O} where {A<:AbstractArray,O}
const TabularVApproximator = Approximator{A,O} where {A<:AbstractVector,O}

"""
    TabularApproximator(table<:AbstractArray, opt)

For `table` of 1-d, it will serve as a state value approximator.
For `table` of 2-d, it will serve as a state-action value approximator.

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
function TabularApproximator(table::A, opt::O) where {A<:AbstractArray,O}
    n = ndims(table)
    n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
    TabularApproximator{A,O}(table, opt)
end

TabularVApproximator(; n_state, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_state), opt)

TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_action, n_state), opt)

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
forward(L::TabularVApproximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(L, x))
forward(L::TabularQApproximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(L, x))

RLCore.forward(
    app::TabularVApproximator{R,O},
    s::I,
) where {R<:AbstractVector,O,I} = @views app.model[s]

RLCore.forward(
    app::TabularQApproximator{R,O},
    s::I,
) where {R<:AbstractArray,O,I} = @views app.model[:, s]

RLCore.forward(
    app::TabularQApproximator{R,O},
    s::I1,
    a::I2,
) where {R<:AbstractArray,O,I1,I2} = @views app.model[a, s]
