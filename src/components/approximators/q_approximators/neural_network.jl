export NeuralNetworkQ, update!

using Flux
using CuArrays: @allowscalar

"TODO: Add static size check for legal input data type."
struct NeuralNetworkQ{D, M, O, P} <: AbstractQApproximator
    model::M
    optimizer::O
    params::P
end

function NeuralNetworkQ(;model, optimizer, device=:cpu)
    if device == :cpu
        m = cpu(model)
        ps = params(m)
        NeuralNetworkQ{device, typeof(m), typeof(optimizer), typeof(ps)}(m, optimizer, ps)
    elseif device == :gpu
        if Flux.has_cuarrays()
            m = gpu(model)
            ps = params(m)
            NeuralNetworkQ{device, typeof(m), typeof(optimizer), typeof(ps)}(m, optimizer, ps)
        else
            error("the specified device is gpu, but can not find CuArrays!")
        end
    else
        throw(ArgumentError("unknown supported $(device)"))
    end
end

to_device(Q::NeuralNetworkQ{:cpu}, x) = cpu(x)
to_device(Q::NeuralNetworkQ{:gpu}, x) = gpu(x)

"get Q value of some specific action"
(Q::NeuralNetworkQ{:cpu})(s, a) = Q(to_device(Q, s))[a]
(Q::NeuralNetworkQ{:gpu})(s, a) = @allowscalar Q(to_device(Q, s))[a]

"get Q value of the batch"
function batch_estimate(Q::NeuralNetworkQ, states, actions)
    q = to_device(Q, states) |> Q.model
    q[CartesianIndex.(actions, axes(q, ndims(q)))]
end

batch_estimate(Q::NeuralNetworkQ, states) = Q.model(to_device(Q, s))

"get Q value of all actions"
(Q::NeuralNetworkQ)(s) = Q.model(to_device(Q, s)) |> cpu

function update!(Q::NeuralNetworkQ, loss)
    gs = Flux.gradient(() -> loss, Q.params)
    Flux.Optimise.update!(Q.optimizer, Q.params, gs)
end

function Base.copyto!(dest::NeuralNetworkQ, src::NeuralNetworkQ)
    Flux.loadparams!(dest.model, src.params)
end