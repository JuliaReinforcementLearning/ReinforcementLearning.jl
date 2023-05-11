export GumbelSoftmaxExplorer

using Random
using Flux: logsoftmax

struct GumbelSoftmaxExplorer <: AbstractExplorer
    rng::AbstractRNG
end

GumbelSoftmaxExplorer(; rng = Random.GLOBAL_RNG) = GumbelSoftmaxExplorer(rng)

function plan!(p::GumbelSoftmaxExplorer, v::AbstractVector{T}) where {T}
    logits = logsoftmax(v)
    u = rand(p.rng, T, length(logits))
    argmax(logits .- log.(-log.(u)))
end

function plan!(p::GumbelSoftmaxExplorer,
    v::AbstractVector{T},
    mask::AbstractVector{Bool},
) where {T}
    v[.!mask] .= typemin(T)
    p(v)
end
