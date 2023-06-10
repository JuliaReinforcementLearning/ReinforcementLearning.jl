export GumbelSoftmaxExplorer

using Random
using Flux: logsoftmax

struct GumbelSoftmaxExplorer <: AbstractExplorer
    rng::AbstractRNG
end

GumbelSoftmaxExplorer(; rng = Random.default_rng()) = GumbelSoftmaxExplorer(rng)

function RLBase.plan!(p::GumbelSoftmaxExplorer, v::AbstractVector{T}) where {T}
    logits = logsoftmax(v)
    u = rand(p.rng, T, length(logits))
    argmax(logits .- log.(-log.(u)))
end

function RLBase.plan!(p::GumbelSoftmaxExplorer,
    v::AbstractVector{T},
    mask::AbstractVector{Bool},
) where {T}
    v[.!mask] .= typemin(T)
    p(v)
end
