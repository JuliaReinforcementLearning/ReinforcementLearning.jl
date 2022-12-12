export AbstractLearner, Approximator

import Flux
using Functors: @functor

abstract type AbstractLearner end

Base.show(io::IO, m::MIME"text/plain", L::AbstractLearner) = show(io, m, convert(AnnotatedStructTree, L))

(L::AbstractLearner)(env::AbstractEnv) = env |> state |> send_to_device(L) |> L |> send_to_device(env)

Base.@kwdef mutable struct Approximator{M,O}
    model::M
    optimiser::O
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

(A::Approximator)(args...) = A.model(args...)

RLBase.optimise!(A::Approximator, gs) =
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)