export FluxApproximator

using Flux

"""
    FluxApproximator(model, optimiser)

Wraps a Flux trainable model and implements the `RLBase.optimise!(::FluxApproximator, ::Gradient)` 
interface. See the RLCore documentation for more information on proper usage.
"""
struct FluxApproximator{M,O} <: AbstractLearner
    model::M
    optimiser_state::O
end


"""
    FluxApproximator(; model, optimiser, usegpu=false)

Constructs an `FluxApproximator` object for reinforcement learning.

# Arguments
- `model`: The model used for approximation.
- `optimiser`: The optimizer used for updating the model.
- `usegpu`: A boolean indicating whether to use GPU for computation. Default is `false`.

# Returns
An `FluxApproximator` object.
"""
function FluxApproximator(; model, optimiser, use_gpu=false)
    optimiser_state = Flux.setup(optimiser, model)
    if use_gpu  # Pass model to GPU (if available) upon creation
        return FluxApproximator(gpu(model), gpu(optimiser_state))
    else
        return FluxApproximator(model, optimiser_state)
    end
end

FluxApproximator(model, optimiser::Flux.Optimise.AbstractOptimiser; use_gpu=false) = FluxApproximator(model=model, optimiser=optimiser, use_gpu=use_gpu)

Flux.@layer FluxApproximator trainable=(model,)

forward(A::FluxApproximator, args...; kwargs...) = A.model(args...; kwargs...)
forward(A::FluxApproximator, env::E, player::AbstractPlayer=current_player(env)) where {E <: AbstractEnv} = env |> (x -> state(x, player)) |> (x -> forward(A, x))

RLBase.optimise!(A::FluxApproximator, grad::NamedTuple) =
    Flux.Optimise.update!(A.optimiser_state, A.model, grad.model)
