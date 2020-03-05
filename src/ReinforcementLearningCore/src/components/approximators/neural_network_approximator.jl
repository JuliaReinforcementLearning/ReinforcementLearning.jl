export NeuralNetworkApproximator

using Flux

struct NeuralNetworkApproximator{T,M,O,P} <: AbstractApproximator
    model::M
    optimizer::O
    params::P
end

"""
    NeuralNetworkApproximator(;kwargs)

Use a DNN model for value estimation.

# Keyword arguments

- `model`, a Flux based DNN model.
- `optimizer`
- `parameters=params(model)`
- `kind=Q_APPROXIMATOR`, specify the type of model.
"""
function NeuralNetworkApproximator(;
    model::M,
    optimizer::O,
    parameters::P = params(model),
    kind = Q_APPROXIMATOR,
) where {M,O,P}
    NeuralNetworkApproximator{kind,M,O,P}(model, optimizer, parameters)
end

device(app::NeuralNetworkApproximator) = device(app.model)

Flux.params(app::NeuralNetworkApproximator) = app.params

(app::NeuralNetworkApproximator)(s::AbstractArray) = app.model(s)
(app::NeuralNetworkApproximator{Q_APPROXIMATOR})(s::AbstractArray, a::Int) = app.model(s)[a]
(app::NeuralNetworkApproximator{HYBRID_APPROXIMATOR})(s::AbstractArray, ::Val{:Q}) =
    app.model(s, Val(:Q))
(app::NeuralNetworkApproximator{HYBRID_APPROXIMATOR})(s::AbstractArray, ::Val{:V}) =
    app.model(s, Val(:V))
(app::NeuralNetworkApproximator{HYBRID_APPROXIMATOR})(s::AbstractArray, a::Int) =
    app.model(s, Val(:Q))[a]


RLBase.batch_estimate(app::NeuralNetworkApproximator, states::AbstractArray) =
    app.model(states)

RLBase.batch_estimate(
    app::NeuralNetworkApproximator{HYBRID_APPROXIMATOR},
    states::AbstractArray,
    ::Val{:Q},
) = app.model(states, Val(:Q))

RLBase.batch_estimate(
    app::NeuralNetworkApproximator{HYBRID_APPROXIMATOR},
    states::AbstractArray,
    ::Val{:V},
) = app.model(states, Val(:V))

RLBase.update!(app::NeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, app.params, gs)

Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, src.params)
