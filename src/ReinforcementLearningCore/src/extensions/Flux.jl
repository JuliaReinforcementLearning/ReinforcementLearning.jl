export glorot_uniform, glorot_normal, orthogonal

import Flux: glorot_uniform, glorot_normal

using Random
using LinearAlgebra

# https://github.com/FluxML/Flux.jl/pull/1171/
# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
function orthogonal_matrix(rng::AbstractRNG, nrow, ncol)
    shape = reverse(minmax(nrow, ncol))
    a = randn(rng, Float32, shape)
    q, r = qr(a)
    q = Matrix(q) * diagm(sign.(diag(r)))
    nrow < ncol ? permutedims(q) : q
end

function orthogonal(rng::AbstractRNG, d1, rest_dims...)
    m = orthogonal_matrix(rng, d1, *(rest_dims...))
    reshape(m, d1, rest_dims...)
end

orthogonal(dims...) = orthogonal(Random.GLOBAL_RNG, dims...)
orthogonal(rng::AbstractRNG) = (dims...) -> orthogonal(rng, dims...)

function batch!(data, xs)
    for (i, x) in enumerate(xs)
        data[Flux.batchindex(data, i)...] = x
    end
    data
end
