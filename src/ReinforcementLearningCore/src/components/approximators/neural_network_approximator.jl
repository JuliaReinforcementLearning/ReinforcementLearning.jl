export NeuralNetworkApproximator

using Flux

"""
    NeuralNetworkApproximator(;kwargs)

Use a DNN model for value estimation.

# Keyword arguments

- `model`, a Flux based DNN model.
- `optimizer`
- `parameters=params(model)`
"""
Base.@kwdef struct NeuralNetworkApproximator{M,O,P} <: AbstractApproximator
    model::M
    optimizer::O
    params::P = params(model)
end

(app::NeuralNetworkApproximator)(x) = app.model(x)

Flux.params(app::NeuralNetworkApproximator) = app.params

device(app::NeuralNetworkApproximator) = device(app.model)

RLBase.update!(app::NeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, app.params, gs)

Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, src.params)
