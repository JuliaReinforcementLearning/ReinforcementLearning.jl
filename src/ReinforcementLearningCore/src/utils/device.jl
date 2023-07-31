# TODO: watch https://github.com/JuliaGPU/Adapt.jl/pull/52

export device, send_to_device, send_to_host

using Flux
using Adapt
using Random

send_to_host(x) = send_to_device(Val(:cpu), x)

send_to_device(d) = x -> send_to_device(device(d), x)

send_to_device(::Val{:cpu}, m) = fmap(x -> adapt(Array, x), m)

# TODO: handle multi-devices

"""
    device(model)

Detect the suitable running device for the `model`.
Return `Val(:cpu)` by default.
"""
device(x) = device(Flux.trainable(x))
device(x::Function) = nothing
device(::Array) = Val(:cpu)
device(x::Tuple{}) = nothing
device(x::NamedTuple{(),Tuple{}}) = nothing
device(x::AbstractArray) = device(parent(x))

device(x::AbstractEnv) = Val(:cpu)  # TODO: we may support gpu later

device(x::Random.AbstractRNG) = Val(:cpu)

function device(x::Union{Tuple,NamedTuple})
    d1 = device(first(x))
    if isnothing(d1)
        device(Base.tail(x))
    else
        d1
    end
end

# recognize Torch.jl
# device(x::Tensor) = Val(Symbol(:gpu, x.device))

