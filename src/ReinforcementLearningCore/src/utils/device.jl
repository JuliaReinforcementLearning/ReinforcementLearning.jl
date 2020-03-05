export device, send_to_host, send_to_device

using Flux
using CuArrays
using Adapt

send_to_host(x) = send_to_device(Val(:cpu), x)
send_to_device(::Val{:cpu}, x) = x  # cpu(x) is not very efficient! So by default we do nothing here.

send_to_device(::Val{:cpu}, x::CuArray) = adapt(Array, x)
send_to_device(::Val{:gpu}, x) = Flux.fmap(a -> adapt(CuArray{Float32}, a), x)
send_to_device(::Val{:gpu}, x::SubArray{T,N,<:CircularArrayBuffer}) where {T,N} =
    CuArray{T}(x)

"""
    device(model)

Detect the suitable running device for the `model`.
Return `Val(:cpu)` by default.
"""
device(x) = device(Flux.trainable(x))
device(x::Function) = nothing
device(::CuArray) = Val(:gpu)
device(::Array) = Val(:cpu)
device(x::Tuple{}) = nothing
device(x::NamedTuple{(),Tuple{}}) = nothing

function device(x::Union{Tuple, NamedTuple})
    d1 = device(x[1])
    if isnothing(d1)
        device(Base.tail(x))
    else
        d1
    end
end
