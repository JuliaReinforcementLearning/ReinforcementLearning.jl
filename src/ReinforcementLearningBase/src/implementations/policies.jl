export RandomPolicy

using Random

"""
    RandomPolicy(action_space, rng)

Randomly return a valid action.
"""
struct RandomPolicy{S<:AbstractSpace,R<:AbstractRNG} <: AbstractPolicy
    action_space::S
    rng::R
end

Base.show(io::IO, p::RandomPolicy) = print(io, "RandomPolicy($(p.action_space))")

Random.seed!(p::RandomPolicy, seed) = Random.seed!(p.rng, seed)

"""
    RandomPolicy(action_space; seed=nothing)
"""
RandomPolicy(s; seed = nothing) = RandomPolicy(s, MersenneTwister(seed))

"""
    RandomPolicy(env::AbstractEnv; seed=nothing)
"""
RandomPolicy(env::AbstractEnv; seed = nothing) =
    RandomPolicy(get_action_space(env), MersenneTwister(seed))

(p::RandomPolicy)(obs) = p(obs, ActionStyle(obs))

function (p::RandomPolicy)(obs, ::FullActionSet)
    legal_actions = get_legal_actions(obs)
    if length(legal_actions) == 0
        # in this case, we return an random action instead of throwing error
        rand(p.rng, p.action_space)
    else
        rand(p.rng, legal_actions)
    end
end

(p::RandomPolicy)(obs, ::MinimalActionSet) = rand(p.rng, p.action_space)
(p::RandomPolicy)(obs::BatchObs, ::MinimalActionSet) =
    [rand(p.rng, p.action_space) for _ in 1:length(obs)]

get_prob(p::RandomPolicy, obs) = get_prob(p, obs, ActionStyle(obs))

# TODO: TBD
# Ideally we should return a Categorical distribution.
# But this means we need to introduce an extra dependency of Distributions
get_prob(p::RandomPolicy, obs, ::MinimalActionSet) =
    fill(1 / length(p.action_space), length(p.action_space))

function get_prob(p::RandomPolicy, obs, ::FullActionSet)
    mask = get_legal_actions_mask(obs)
    n = sum(mask)
    prob = zeros(length(mask))
    prob[mask] .= 1 / n
    prob
end

get_prob(p::RandomPolicy, obs, a) = get_prob(p, obs, a, ActionStyle(obs))

get_prob(p::RandomPolicy, obs, a, ::MinimalActionSet) = 1 / length(p.action_space)

function get_prob(p::RandomPolicy, obs, a, ::FullActionSet)
    legal_actions = get_legal_actions(obs)
    if a in legal_actions
        1 / length(legal_actions)
    else
        0
    end
end
