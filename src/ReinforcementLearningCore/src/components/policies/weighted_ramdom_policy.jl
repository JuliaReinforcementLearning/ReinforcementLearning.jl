export WeightedRandomPolicy

using Random
using StatsBase: sample, Weights

struct WeightedRandomPolicy{N,A,W<:AbstractArray,S,R<:AbstractRNG} <: AbstractPolicy
    actions::A
    weight::W
    sums::S
    rng::R
end

function WeightedRandomPolicy(
    weight::W;
    actions = axes(weights, 1),
    seed = nothing,
) where {W<:AbstractArray}
    rng = MersenneTwister(seed)
    N = ndims(W)

    if N == 1
        sums = sum(weight)
    elseif N == 2
        sums = vec(sum(weight, dims = 1))
    end
    WeightedRandomPolicy{ndims(W),typeof(actions),W,typeof(sums),typeof(rng)}(
        actions,
        weight,
        sums,
        rng,
    )
end

Random.seed!(p::WeightedRandomPolicy, seed) = Random.seed!(p.rng, seed)

RLBase.update!(p::WeightedRandomPolicy, experience) = nothing

(p::WeightedRandomPolicy{1})(obs, ::MinimalActionSet) =
    sample(p.rng, p.actions, Weights(p.weight, p.sums))

function (p::WeightedRandomPolicy{1})(obs, ::FullActionSet)
    legal_actions = get_legal_actions(obs)
    legal_actions_mask = get_legal_actions_mask(obs)
    masked_weight = @view p.weight[legal_actions_mask]
    legal_actions[sample(p.rng, Weights(masked_weight))]
end

function (p::WeightedRandomPolicy{2})(obs, ::MinimalActionSet)
    s = get_state(obs)
    weight = @view p.weight[:, s]
    sample(p.rng, p.actions, Weights(weight, p.sums[s]))
end

function (p::WeightedRandomPolicy{2})(obs, ::FullActionSet)
    s = get_state(obs)
    legal_actions = get_legal_actions(obs)
    legal_actions_mask = get_legal_actions_mask(obs)
    masked_weight = @view p.weight[legal_actions_mask, s]
    legal_actions[sample(p.rng, Weights(masked_weight))]
end
