export seed_glorot_normal, seed_glorot_uniform

import Flux: glorot_uniform, glorot_normal

using Random

glorot_uniform(rng::AbstractRNG, dims...) =
    (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(Flux.nfan(dims...)))
glorot_normal(rng::AbstractRNG, dims...) =
    randn(rng, Float32, dims...) .* sqrt(2.0f0 / sum(Flux.nfan(dims...)))

seed_glorot_uniform(; seed = nothing) =
    (dims...) -> glorot_uniform(MersenneTwister(seed), dims...)
seed_glorot_normal(; seed = nothing) =
    (dims...) -> glorot_normal(MersenneTwister(seed), dims...)
