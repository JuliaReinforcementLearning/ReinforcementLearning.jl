export RandomPolicy, RandomStartPolicy, TabularRandomPolicy

using Random

"""
    RandomPolicy(action_space, rng=Random.GLOBAL_RNG)

Construct a random policy with actions in `action_space`.
If `action_space` is `nothing` then the `legal_actions` at runtime
will be used to randomly sample a valid action.
"""
struct RandomPolicy{S,R<:AbstractRNG} <: AbstractPolicy
    action_space::S
    rng::R
end

Random.seed!(p::RandomPolicy, seed) = Random.seed!(p.rng, seed)

RandomPolicy(; rng = Random.GLOBAL_RNG) = RandomPolicy(nothing, rng)
RandomPolicy(s; rng = Random.GLOBAL_RNG) = RandomPolicy(s, rng)

"""
    RandomPolicy(env::AbstractEnv; rng=Random.GLOBAL_RNG)

If `env` is of [`FULL_ACTION_SET`](@ref), then action is randomly chosen at runtime
in `get_actions(env)`. Otherwise, the `env` is supposed to be of [`MINIMAL_ACTION_SET`](@ref).
The `get_actions(env)` is supposed to be static and will only be used to initialize
the random policy for once.
"""
RandomPolicy(env::AbstractEnv; rng = Random.GLOBAL_RNG) =
    RandomPolicy(ActionStyle(env), env, rng)
RandomPolicy(::MinimalActionSet, env::AbstractEnv, rng) =
    RandomPolicy(get_actions(env), rng)
RandomPolicy(::FullActionSet, env::AbstractEnv, rng) = RandomPolicy(nothing, rng)

(p::RandomPolicy{Nothing})(env) = rand(p.rng, get_legal_actions(env))
(p::RandomPolicy)(env) = rand(p.rng, p.action_space)

# TODO: TBD
# Ideally we should return a Categorical distribution.
# But this means we need to introduce an extra dependency of Distributions
# watch https://github.com/JuliaStats/Distributions.jl/issues/1139
get_prob(p::RandomPolicy, env) = fill(1 / length(p.action_space), length(p.action_space))

function get_prob(p::RandomPolicy{Nothing}, env)
    mask = get_legal_actions_mask(env)
    n = sum(mask)
    prob = zeros(length(mask))
    prob[mask] .= 1 / n
    prob
end

get_prob(p::RandomPolicy, env, a) = 1 / length(p.action_space)
get_prob(p::RandomPolicy{<:VectSpace}, env::MultiThreadEnv, a) =
    [1 / length(x) for x in p.action_space.data]

function get_prob(p::RandomPolicy{Nothing}, env, a)
    legal_actions = get_legal_actions(env)
    if a in legal_actions
        1.0 / length(legal_actions)
    else
        0.0
    end
end

#####
# RandomStartPolicy
#####

Base.@kwdef mutable struct RandomStartPolicy{P,R<:RandomPolicy} <: AbstractPolicy
    policy::P
    random_policy::R
    num_rand_start::Int
end

function (p::RandomStartPolicy)(env)
    p.num_rand_start -= 1
    if p.num_rand_start < 0
        p.policy(env)
    else
        p.random_policy(env)
    end
end

update!(p::RandomStartPolicy, experience) = update!(p.policy, experience)

for f in (:get_prob, :get_priority)
    @eval function $f(p::RandomStartPolicy, args...)
        if p.num_rand_start < 0
            $f(p.policy, args...)
        else
            $f(p.random_policy, args...)
        end
    end
end

#####
# TabularRandomPolicy
#####

"""
    TabularRandomPolicy(;table=Dict(), rng=Random.GLOBAL_RNG)

Use a table to store action probabilities.
"""
struct TabularRandomPolicy{S,T,R<:AbstractRNG} <: AbstractPolicy
    table::Dict{S, Vector{T}}
    rng::R
end

Random.seed!(p::TabularRandomPolicy, seed) = Random.seed!(p.rng, seed)

TabularRandomPolicy(;rng=Random.GLOBAL_RNG) = TabularRandomPolicy{Int, Float32}(;rng=rng)
TabularRandomPolicy{S}(;rng=Random.GLOBAL_RNG) where {S} = TabularRandomPolicy{S, Float32}(;rng=rng)
TabularRandomPolicy{S, T}(;rng=Random.GLOBAL_RNG) where {S,T} = TabularRandomPolicy(Dict{S,Vector{T}}(), rng)

(p::TabularRandomPolicy)(env::AbstractEnv) = _weighted_sample(p.rng, get_prob(p, env))

function get_prob(p::TabularRandomPolicy, env::AbstractEnv)
    s = get_state(env)
    if haskey(p.table, s)
        p.table[s]
    elseif ActionStyle(env) === FULL_ACTION_SET
        mask = get_legal_actions_mask(env)
        prob = mask ./ sum(mask)
        p.table[s] = prob
        prob
    elseif ActionStyle(env) === MINIMAL_ACTION_SET
        n = length(get_actions(env))
        prob = fill(1 / n, n)
        p.table[s] = prob
        prob
    end
end

update!(p::TabularRandomPolicy, experience::Pair{Int, <:AbstractVector}) = p.table[first(experience)] = last(experience)

"""
Directly copied from [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl/blob/0ea8e798c3d19609ed33b11311de5a2bd6ee9fd0/src/sampling.jl#L499-L510) to avoid depending on the whole package.
"""
function _weighted_sample(rng::AbstractRNG, wv)
    t = rand(rng)
    n = length(wv)
    i = 1
    cw = wv[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += wv[i]
    end
    return i
end