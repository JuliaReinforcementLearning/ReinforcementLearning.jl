export FluxModelApproximator

using Flux

"""
    FluxModelApproximator(model, optimiser)

Wraps a Flux trainable model and implements the `RLBase.optimise!(::FluxModelApproximator, ::Gradient)` 
interface. See the RLCore documentation for more information on proper usage.
"""
struct FluxModelApproximator{M,O} <: AbstractLearner
    model::M
    optimiser_state::O
end


"""
    FluxModelApproximator(; model, optimiser, usegpu=false)

Constructs an `FluxModelApproximator` object for reinforcement learning.

# Arguments
- `model`: The model used for approximation.
- `optimiser`: The optimizer used for updating the model.
- `usegpu`: A boolean indicating whether to use GPU for computation. Default is `false`.

# Returns
An `FluxModelApproximator` object.
"""
function FluxModelApproximator(; model, optimiser, use_gpu=false)
    optimiser_state = Flux.setup(optimiser, model)
    if use_gpu  # Pass model to GPU (if available) upon creation
        return FluxModelApproximator(gpu(model), gpu(optimiser_state))
    else
        return FluxModelApproximator(model, optimiser_state)
    end
end

FluxModelApproximator(model, optimiser::Flux.Optimise.AbstractOptimiser; use_gpu=false) = FluxModelApproximator(model=model, optimiser=optimiser, use_gpu=use_gpu)

Flux.@layer FluxModelApproximator trainable=(model,)

forward(A::FluxModelApproximator, args...; kwargs...) = A.model(args...; kwargs...)
forward(A::FluxModelApproximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(A, x))

RLBase.optimise!(A::FluxModelApproximator, grad::NamedTuple) =
    Flux.Optimise.update!(A.optimiser_state, A.model, grad.model)
