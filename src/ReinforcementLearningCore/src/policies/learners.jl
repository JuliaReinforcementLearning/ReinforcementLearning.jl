export AbstractLearner, Approximator

import Flux
using Functors: @functor

abstract type AbstractLearner end

Base.show(io::IO, m::MIME"text/plain", L::AbstractLearner) = show(io, m, convert(AnnotatedStructTree, L))

estimate_reward(L::Le, env::E) where {Le <: AbstractLearner, E <: AbstractEnv} = env |> state |> send_to_device(L) |> x -> RLCore.estimate_reward(L.approximator, x) |> send_to_device(env) 

Base.@kwdef mutable struct Approximator{M,O}
    model::M
    optimiser::O
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

estimate_reward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)

RLBase.optimise!(A::Approximator, gs) =
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
