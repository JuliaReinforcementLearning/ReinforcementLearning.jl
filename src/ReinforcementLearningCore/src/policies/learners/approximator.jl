"""
    Approximator(model, optimiser)

Wraps a Flux trainable model and implements the `RLBase.optimise!(::Approximator, ::Gradient)` 
interface. See the RLCore documentation for more information on proper usage.
"""
struct Approximator{M,O} <: AbstractLearner
    model::M
    optimiser_state::O
end

function Approximator(; model, optimiser, gpu=false)
    optimiser_state = Flux.setup(optimiser, model)
    if gpu
        return Approximator(gpu(model), gpu(optimiser_state))
    else
        Approximator(model, optimiser_state)
    end
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

forward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)

RLBase.optimise!(A::Approximator, grad) =
    Flux.Optimise.update!(A.optimiser_state, A.model, grad)
