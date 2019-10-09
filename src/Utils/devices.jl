export to_device, to_host

using CUDAapi
using Flux
using Knet
using Adapt
import Adapt: adapt, adapt_storage

to_host(x) = to_device(Val(:cpu), x)
to_device(::Val{:cpu}, x) = cpu(x)

if has_cuda()
    using CuArrays
    to_device(::Val{:Zygote_gpu}, x) = Flux.fmap(a -> adapt(CuArray, a), x)
    to_device(::Val{:Zygote_gpu}, x::SubArray) = CuArray(x)  # !!! Do not use `adapt` here! For `SubArray`, doing this will send the parent to gpu and this is not what we want for most cases regarding our implementation of buffers.
    to_device(::Val{:Zygote_gpu}, x::SubArray{T, N, P}) where {T, N, P<:CuArray} = x
end

to_device(::Val{:Knet_gpu}, x) = Flux.fmap(a -> adapt(KnetArray{Float32}, a), x)
to_device(::Val{:Knet_gpu}, x::SubArray) = KnetArray{Float32}(x)

adapt_storage(T::Type{<:KnetArray}, x::AbstractArray{<:Real}) = T(x)
adapt_storage(T::Type{<:KnetArray}, x::Knet.Param{<:AbstractArray}) = Knet.Param(T(value(x)))
adapt_storage(T::Type{<:Array}, x::KnetArray) = T(x)