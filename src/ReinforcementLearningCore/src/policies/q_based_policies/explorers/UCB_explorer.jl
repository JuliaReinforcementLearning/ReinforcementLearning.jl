export UCBExplorer

using Random
using Flux

Base.@kwdef mutable struct UCBExplorer{R<:AbstractRNG} <: AbstractExplorer
    c::Float64
    actioncounts::Vector{Float64}
    step::Int
    rng::R
    is_training::Bool = true
end

"""
    UCBExplorer(na; c=2.0, ϵ=1e-10, step=1, seed=nothing)

# Arguments
- `na` is the number of actions used to create a internal counter.
- `t` is used to store current time step.
- `c` is used to control the degree of exploration.
- `seed`, set the seed of inner RNG.
- `is_training=true`, in training mode, time step and counter will not be updated.
"""
UCBExplorer(na; c = 2.0, ϵ = 1e-10, step = 1, rng = Random.GLOBAL_RNG, is_training = true) =
    UCBExplorer(c, fill(ϵ, na), 1, rng, is_training)

function (p::UCBExplorer)(values::AbstractArray)
    v, inds = find_all_max(@. values + p.c * sqrt(log(p.step + 1) / p.actioncounts))
    action = sample(p.rng, inds)
    if p.is_training
        p.actioncounts[action] += 1
        p.step += 1
    end
    action
end
