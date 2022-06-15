# TODO: watch https://github.com/JuliaGPU/Adapt.jl/pull/52

export device, send_to_device

using Flux
using CUDA
using Adapt
using Random

import CUDA: device

send_to_device(d) = x -> send_to_device(device(d), x)

send_to_device(::Val{:cpu}, m) = fmap(x -> adapt(Array, x), m)

# TODO: handle multi-devices
send_to_device(::CuDevice, m) = fmap(CUDA.cu, m)

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

function device(x::Random.AbstractRNG)
    if x isa CUDA.CURAND.RNG
        device()
    else
        Val(:cpu)
    end
end

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

# Since v0.1.10 CircularArrayBuffer will adapt internal buffer into GPU
# But in RL.jl, we don't need that feature as far as I know

import CircularArrayBuffers: CircularArrayBuffer

send_to_device(d::CuDevice, m::CircularArrayBuffer) = send_to_device(d, collect(m))