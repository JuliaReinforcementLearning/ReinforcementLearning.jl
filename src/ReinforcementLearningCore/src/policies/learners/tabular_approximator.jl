export TabularApproximator, TabularVApproximator, TabularQApproximator

using Flux: gpu

"""
    TabularApproximator(table<:AbstractArray, opt)

For `table` of 1-d, it will serve as a state value approximator.
For `table` of 2-d, it will serve as a state-action value approximator.

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
# TODO: add back missing AbstractApproximator
struct TabularApproximator{N,A,O} <: AbstractLearner
    table::A
    optimizer::O
    function TabularApproximator(table::A, opt::O) where {A<:AbstractArray,O}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
        new{n,A,O}(table, opt)
    end
end

TabularVApproximator(; n_state, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_state), opt)

TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_action, n_state), opt)

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
function forward(L::TabularApproximator, env::E) where {E <: AbstractEnv}
    env |> state |> (x -> forward(L, x))
end

RLCore.forward(
    app::TabularApproximator{1,R,O},
    s::I,
) where {R<:AbstractArray,O,I<:Integer} = @views app.table[s]

RLCore.forward(
    app::TabularApproximator{2,R,O},
    s::I,
) where {R<:AbstractArray,O,I<:Integer} = @views app.table[:, s]
RLCore.forward(
    app::TabularApproximator{2,R,O},
    s::I1,
    a::I2,
) where {R<:AbstractArray,O,I1<:Integer,I2<:Integer} = @views app.table[a, s]
