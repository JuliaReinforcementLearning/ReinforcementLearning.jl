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



struct Approximator{M,O} <: AbstractLearner
    model::M
    optimiser::O
end

function Approximator(; model, optimiser)
    Approximator(gpu(model), optimiser) # Pass model to GPU (if available) upon creation
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

forward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)

RLBase.optimise!(A::Approximator, gs) =
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
