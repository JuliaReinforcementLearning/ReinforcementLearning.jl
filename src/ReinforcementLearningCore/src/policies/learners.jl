export AbstractLearner, Approximator

import Flux
import Functors

abstract type AbstractLearner end

(L::AbstractLearner)(env) = env |> state |> send_to_device(L) |> L |> send_to_device(env)

Base.@kwdef mutable struct Approximator{M,O}
    model::M
    optimiser::O
end

Functors.functor(x::Approximator) = (model = x.model,), y -> Approximator(y.model, x.state)

(A::Approximator)(x) = A.model(x)

RLBase.optimise!(A::Approximator, gs) =
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)