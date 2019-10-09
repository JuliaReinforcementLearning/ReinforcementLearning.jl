export NeuralNetworkQ, update!

using Flux
using CUDAapi

import Flux:params
import Zygote:gradient
import .Utils:to_device, to_host

"TODO: Add static size check for legal input data type."
struct NeuralNetworkQ{D, M, O, P} <: AbstractQApproximator
    model::M
    optimizer::O
    params::P
end

function NeuralNetworkQ(;model::M, optimizer::O, parameters::P=params(model), device::Symbol=:cpu) where {M, O, P}
    if device == :gpu
        has_cuda() || throw(ArgumentError("the specified device is gpu but can not find cuda!"))
        device = Symbol(backend(model), "_gpu")
    end
    m = to_device(Val(device), model)
    ps = params(m)
    NeuralNetworkQ{device, typeof(m), O, typeof(ps)}(m, optimizer, ps)
end

params(Q::NeuralNetworkQ) = Q.params

to_device(Q::NeuralNetworkQ{D}, x) where D = to_device(Val(D), x)

"get Q value of the batch"
batch_estimate(Q::NeuralNetworkQ, states) = Q.model(states)

"get Q value of all actions"
(Q::NeuralNetworkQ)(s) = Q.model(to_device(Q, s)) |> to_host

"get Q value of some specific action"
(Q::NeuralNetworkQ)(s, a) = Q(s)[a]

function update!(Q::NeuralNetworkQ, gs)
    Flux.Optimise.update!(Q.optimizer, Q.params, gs)
end

Base.copyto!(dest::NeuralNetworkQ, src::NeuralNetworkQ) = Flux.loadparams!(dest.model, src.params)

gradient(f, Q::NeuralNetworkQ, args...) = gradient(f, Val(backend(Q.model)), args...)