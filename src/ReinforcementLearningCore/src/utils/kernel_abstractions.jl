# TODO: watch https://github.com/JuliaGPU/Adapt.jl/pull/52

export send_to_device, send_to_host

using Flux
using Adapt
using Random
using KernelAbstractions: CPU
import KernelAbstractions

send_to_host(x) = send_to_device(CPU(; static=false), x)

send_to_device(d) = x -> send_to_device(KernelAbstractions.get_backend(d), x)

send_to_device(::CPU, m) = fmap(x -> adapt(Array, x), m)

KernelAbstractions.get_backend(x) = KernelAbstractions.get_backend(Flux.trainable(x))
KernelAbstractions.get_backend(x::Function) = nothing
KernelAbstractions.get_backend(x::Tuple{}) = nothing
KernelAbstractions.get_backend(x::NamedTuple{(),Tuple{}}) = nothing
KernelAbstractions.get_backend(x::AbstractEnv) = CPU(;static=false)  # TODO: we may support gpu later

KernelAbstractions.get_backend(x::Random.AbstractRNG) = CPU(;static=false)

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
