export NeuralNetworkQ, update!

using Flux
using CUDAapi

import Flux:params
import Zygote:gradient
import .Utils:to_device, to_host

# TODO: Add static size check for legal input data type.

"""
    NeuralNetworkQ(;kwargs...) -> NeuralNetworkQ{D, M, O, P}

Use neural networks to generate estimations of state-action values.

# Keywords

- `model::M`: describes the network structure.
- `optimizer::O`: defines how to update parameters given grads.
- `parameters::P=params(model)`: the parameters of `model`.
- `device::Symbol=:cpu`: the param `D` of `NeuralNetworkQ`, specify where to run the model. Supported keywords are:
    - `:cpu`
    - `:gpu`, if the specified device is `:gpu`, then we will automatically change the device to one of the following device types according to the `backend(model)`:
        - `:Knet_gpu`, means the model is `Knet` that runs on gpu.
        - `:Zygote_gpu`, means the model is `Flux` with `Zygote` that runs on gpu.

# Fields

- `model::M`
- `optimizer::O`
- `params::P`


"""
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
(Q::NeuralNetworkQ)(s) = Q.model(to_device(Q, reshape(s, size(s)..., 1))) |> to_host |> drop_last_dim

"get Q value of some specific action"
(Q::NeuralNetworkQ)(s, a) = Q(s)[a]

function update!(Q::NeuralNetworkQ, gs)
    Flux.Optimise.update!(Q.optimizer, Q.params, gs)
end

Base.copyto!(dest::NeuralNetworkQ, src::NeuralNetworkQ) = Flux.loadparams!(dest.model, src.params)

gradient(f, Q::NeuralNetworkQ) = gradient(f, Val(backend(Q.model)), params(Q))