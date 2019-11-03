using Flux
using Zygote

import Zygote:gradient

gradient(f, ::Val{:Zygote}, args...) = gradient(f, args...)

Zygote.@adjoint argmax(xs; dims = :) = argmax(xs;dims=dims), _ -> nothing

# ??? can safely removed now
Zygote.@adjoint function Base.broadcasted(::typeof(relu), x::Array{T}) where T<:Real
    y = relu.(x)
    return y, Δ -> begin
        res = similar(Δ)
        for i in 1:length(res)
            if y[i] > 0
                res[i] = Δ[i]
            else
                res[i] = zero(T)
            end
        end
        (nothing, res)
    end
end