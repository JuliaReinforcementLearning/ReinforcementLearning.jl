using Zygote

Zygote.@adjoint argmax(xs; dims = :) = argmax(xs; dims = dims), _ -> nothing
