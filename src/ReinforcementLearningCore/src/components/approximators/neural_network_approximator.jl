export NeuralNetworkApproximator

using Flux

struct NeuralNetworkApproximator{T,M,O,P} <: AbstractApproximator
    model::M
    optimizer::O
    params::P
end

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
RLBase.batch_estimate(app::NeuralNetworkApproximator, states::AbstractArray) =
    app.model(states)

RLBase.update!(app::NeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, app.params, gs)

Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, src.params)
