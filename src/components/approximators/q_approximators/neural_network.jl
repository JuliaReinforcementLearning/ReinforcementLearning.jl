export NeuralNetworkQ, update!

using Flux

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

to_device(Q::NeuralNetworkQ{:cpu}, x) = x
to_device(Q::NeuralNetworkQ{:gpu}, x) = gpu(x)

"get Q value of the batch"
batch_estimate(Q::NeuralNetworkQ, states) = Q.model(states)

"get Q value of all actions"
(Q::NeuralNetworkQ)(s) = Q.model(to_device(Q, s)) |> cpu

"get Q value of some specific action"
(Q::NeuralNetworkQ)(s, a) = Q(s)[a]

function update!(Q::NeuralNetworkQ, gs)
    Flux.Optimise.update!(Q.optimizer, Q.params, gs)
end

function Base.copyto!(dest::NeuralNetworkQ, src::NeuralNetworkQ)
    Flux.loadparams!(dest.model, src.params)
end