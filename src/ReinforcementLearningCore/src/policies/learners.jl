export AbstractLearner, Approximator

import Flux
using Functors: @functor

abstract type AbstractLearner end

(L::AbstractLearner)(env::AbstractEnv) = env |> state |> send_to_device(L) |> L |> send_to_device(env)

Base.@kwdef mutable struct Approximator{M,O}
    model::M
    optimiser::O
end

@functor Approximator (model,)

(A::Approximator)(args...) = A.model(args...)

RLBase.optimise!(A::Approximator, gs) =
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)