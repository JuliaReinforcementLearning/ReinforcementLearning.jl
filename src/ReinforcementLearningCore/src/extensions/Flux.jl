export seed_glorot_normal, seed_glorot_uniform, seed_orthogonal

import Flux: glorot_uniform, glorot_normal

using Random
using LinearAlgebra

glorot_uniform(rng::AbstractRNG, dims...) =
    (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(Flux.nfan(dims...)))
glorot_normal(rng::AbstractRNG, dims...) =
    randn(rng, Float32, dims...) .* sqrt(2.0f0 / sum(Flux.nfan(dims...)))

seed_glorot_uniform(; seed = nothing) =
    (dims...) -> glorot_uniform(MersenneTwister(seed), dims...)
seed_glorot_normal(; seed = nothing) =
    (dims...) -> glorot_normal(MersenneTwister(seed), dims...)

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

seed_orthogonal(;seed = nothing) = (dims...) -> orthogonal(MersenneTwister(seed), dims...)