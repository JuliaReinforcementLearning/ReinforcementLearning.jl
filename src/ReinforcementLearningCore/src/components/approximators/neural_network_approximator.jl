export NeuralNetworkApproximator, ActorCritic

using Flux

"""
    NeuralNetworkApproximator(;kwargs)

Use a DNN model for value estimation.

# Keyword arguments

- `model`, a Flux based DNN model.
- `optimizer=Descent()`
"""
Base.@kwdef struct NeuralNetworkApproximator{M,O} <: AbstractApproximator
    model::M
    optimizer::O = Descent()
end

(app::NeuralNetworkApproximator)(x) = app.model(x)

# !!! watch https://github.com/FluxML/Functors.jl/blob/master/src/functor.jl#L2
Flux.functor(x::NeuralNetworkApproximator) =
    (model = x.model,), y -> NeuralNetworkApproximator(y.model, x.optimizer)

device(app::NeuralNetworkApproximator) = device(app.model)

RLBase.update!(app::NeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, params(app), gs)

Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, params(src))

Flux.testmode!(app::NeuralNetworkApproximator, mode = true) = testmode!(app.model, mode)

#####
# ActorCritic
#####

"""
    ActorCritic(actor, critic)

The `actor` part must return a **normalized** vector representing the action values, and the `critic` part must return a state value.

TODO: remove the **normalized** restriction!!!
"""
Base.@kwdef struct ActorCritic{A,C}
    actor::A
    critic::C
end

Flux.@functor ActorCritic

function RLBase.update!(app::ActorCritic, gs)
    update!(app.actor, gs)
    update!(app.critic, gs)
end

function Flux.testmode!(app::ActorCritic, mode = true)
    testmode!(app.actor, mode)
    testmode!(app.critic, mode)
end
