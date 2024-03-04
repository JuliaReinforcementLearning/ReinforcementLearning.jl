export AbstractLearner, Approximator

using Flux
using Functors: @functor

abstract type AbstractLearner end

Base.show(io::IO, m::MIME"text/plain", L::AbstractLearner) = show(io, m, convert(AnnotatedStructTree, L))

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
function forward(L::Le, env::E) where {Le <: AbstractLearner, E <: AbstractEnv}
    env |> state |> Flux.gpu |> (x -> forward(L, x)) |> Flux.cpu
end

function RLBase.optimise!(::AbstractLearner, ::AbstractStage, ::Trajectory) end


"""
    Approximator(model, optimiser)

Wraps a Flux trainable model and implements the `RLBase.optimise!(::Approximator, ::Gradient)` 
interface. See the RLCore documentation for more information on proper usage.
"""
struct Approximator{M,O} <: AbstractLearner
    model::M
    optimiser_state::O
end

function Approximator(; model, optimiser)
    optimiser_state = Flux.setup(optimiser, model)
    Approximator(gpu(model), optimiser_state) # Pass model to GPU (if available) upon creation
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

forward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)

RLBase.optimise!(A::Approximator, grad) =
    Flux.Optimise.update!(A.model, A.optimiser_state, grad)
