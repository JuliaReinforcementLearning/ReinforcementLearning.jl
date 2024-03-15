using Flux

"""
    Approximator(model, optimiser)

Wraps a Flux trainable model and implements the `RLBase.optimise!(::Approximator, ::Gradient)` 
interface. See the RLCore documentation for more information on proper usage.
"""
struct Approximator{M,O} <: AbstractLearner
    model::M
    optimiser_state::O
end


"""
    Approximator(; model, optimiser, usegpu=false)

Constructs an `Approximator` object for reinforcement learning.

# Arguments
- `model`: The model used for approximation.
- `optimiser`: The optimizer used for updating the model.
- `usegpu`: A boolean indicating whether to use GPU for computation. Default is `false`.

# Returns
An `Approximator` object.
"""
function Approximator(; model, optimiser::Flux.Optimise.AbstractOptimiser, use_gpu=false)
    optimiser_state = Flux.setup(optimiser, model)
    if use_gpu  # Pass model to GPU (if available) upon creation
        return Approximator(gpu(model), gpu(optimiser_state))
    else
        return Approximator(model, optimiser_state)
    end
end

Approximator(model, optimiser::Flux.Optimise.AbstractOptimiser; use_gpu=false) = Approximator(model=model, optimiser=optimiser, use_gpu=use_gpu)

Flux.@layer Approximator trainable=(model,)

forward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)
forward(A::Approximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(A, x))

RLBase.optimise!(A::Approximator, grad::NamedTuple) =
    Flux.Optimise.update!(A.optimiser_state, A.model, grad.model)
