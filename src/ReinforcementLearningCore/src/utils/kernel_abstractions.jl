# TODO: watch https://github.com/JuliaGPU/Adapt.jl/pull/52

export send_to_device, send_to_host

using Flux
using CUDA
using Adapt
using Random
using KernelAbstractions: CPU
import KernelAbstractions

send_to_host(x) = send_to_device(CPU(; static=false), x)

send_to_device(d) = x -> send_to_device(get_backend(d), x)

send_to_device(::CPU, m) = fmap(x -> adapt(Array, x), m)

# TODO: handle multi-devices
send_to_device(::CUDABackend, m) = fmap(CUDA.cu, m)

KernelAbstractions.get_backend(x) = KernelAbstractions.get_backend(Flux.trainable(x))
KernelAbstractions.get_backend(x::Function) = nothing
KernelAbstractions.get_backend(x::Tuple{}) = nothing
KernelAbstractions.get_backend(x::NamedTuple{(),Tuple{}}) = nothing
KernelAbstractions.get_backend(x::AbstractEnv) = CPU(;static=false)  # TODO: we may support gpu later

KernelAbstractions.get_backend(x::Random.AbstractRNG) = CPU(;static=false)
KernelAbstractions.get_backend(x::CUDA.CURAND.RNG) = CUDABackend()

function KernelAbstractions.get_backend(x::Union{Tuple,NamedTuple})
    d1 = KernelAbstractions.get_backend(first(x))
    if isnothing(d1)
        KernelAbstractions.get_backend(Base.tail(x))
    else
        d1
    end
end

# recognize Torch.jl
# get_backend(x::Tensor) = Val(Symbol(:gpu, x.device))

# Since v0.1.10 CircularArrayBuffer will adapt internal buffer into GPU
# But in RL.jl, we don't need that feature as far as I know

import CircularArrayBuffers: CircularArrayBuffer

send_to_device(d::CUDABackend, m::CircularArrayBuffer) = send_to_device(d, collect(m))
