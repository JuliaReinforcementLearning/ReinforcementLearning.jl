export glorot_uniform, glorot_normal, orthogonal

import Flux: glorot_uniform, glorot_normal

using Random
using LinearAlgebra

# watch https://github.com/FluxML/Flux.jl/issues/1274
glorot_uniform(rng::AbstractRNG, dims...) =
    (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(Flux.nfan(dims...)))
glorot_normal(rng::AbstractRNG, dims...) =
    randn(rng, Float32, dims...) .* sqrt(2.0f0 / sum(Flux.nfan(dims...)))

glorot_uniform(rng::AbstractRNG) = (dims...) -> glorot_uniform(rng, dims...)
glorot_normal(rng::AbstractRNG) = (dims...) -> glorot_normal(rng, dims...)

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
