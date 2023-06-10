export UCBExplorer

using Random

Base.@kwdef mutable struct UCBExplorer{R<:AbstractRNG} <: AbstractExplorer
    c::Float64
    actioncounts::Vector{Float64}
    step::Int
    rng::R
end

"""
    UCBExplorer(na; c=2.0, ϵ=1e-10, step=1, seed=nothing)

# Arguments
- `na` is the number of actions used to create a internal counter.
- `t` is used to store current time step.
- `c` is used to control the degree of exploration.
- `seed`, set the seed of inner RNG.
"""
UCBExplorer(na; c = 2.0, ϵ = 1e-10, step = 1, rng = Random.default_rng()) =
    UCBExplorer(c, fill(ϵ, na), 1, rng)

function RLBase.plan!(p::UCBExplorer, values::AbstractArray)
    v, inds = find_all_max(@. values + p.c * sqrt(log(p.step + 1) / p.actioncounts))
    action = rand(p.rng, inds)
    p.actioncounts[action] += 1
    p.step += 1
    action
end
