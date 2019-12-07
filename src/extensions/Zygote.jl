using Flux
using Zygote

import Zygote:gradient

gradient(f, ::Val{:Zygote}, args...) = gradient(f, args...)

Zygote.@adjoint argmax(xs; dims = :) = argmax(xs;dims=dims), _ -> nothing