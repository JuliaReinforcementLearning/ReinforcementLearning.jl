export GumbelSoftmaxExplorer

using Random
using Flux: logsoftmax

struct GumbelSoftmaxExplorer <: AbstractExplorer
    rng::AbstractRNG
end

GumbelSoftmaxExplorer(;seed=nothing) = GumbelSoftmaxExplorer(MersenneTwister(seed))

function (p::GumbelSoftmaxExplorer)(v::AbstractVector{T}) where T
    logits = logsoftmax(v)
    u = rand(p.rng, T, length(logits))
    argmax(logits .- log.(-log.(u)))
end