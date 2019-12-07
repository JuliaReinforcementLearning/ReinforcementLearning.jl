export to_device, to_host

using CUDAapi
using Flux
using Knet
using Adapt
import Adapt: adapt, adapt_storage

to_host(x) = to_device(Val(:cpu), x)
to_device(::Val{:cpu}, x) = x  # cpu(x) is not very efficient! So by default we do nothing here.

if has_cuda()
    using CuArrays

    to_device(::Val{:Zygote_gpu}, x) = Flux.fmap(a -> adapt(CuArray{Float32}, a), x)
    to_device(::Val{:cpu}, x::Union{KnetArray, CuArray}) = adapt(Array, x)

    to_device(::Val{:Knet_gpu}, x) = adapt(KnetArray, x)
    to_device(::Val{:Knet_gpu}, x::SubArray) = KnetArray(x)
end

adapt_storage(T::Type{<:KnetArray}, x::AbstractArray{<:Real}) = T(x)
adapt_storage(T::Type{<:KnetArray}, x::Knet.Param{<:AbstractArray}) = Knet.Param(T(value(x)))