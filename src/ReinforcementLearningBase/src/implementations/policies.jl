export RandomPolicy, RandomStartPolicy

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
get_prob(p::RandomPolicy{<:VectSpace}, env::MultiThreadEnv) =
    [fill(1 / length(s), length(s)) for s in p.action_space]
get_prob(p::RandomPolicy, env::MultiThreadEnv) = [get_prob(p, x) for x in env]
get_prob(p::RandomPolicy, env) = fill(1 / length(p.action_space), length(p.action_space))
get_prob(p::RandomPolicy{Nothing}, env) = get_prob(p, env, ChanceStyle(env))

function get_prob(p::RandomPolicy{Nothing}, env, ::AbstractChanceStyle)
    mask = get_legal_actions_mask(env)
    n = sum(mask)
    prob = zeros(length(mask))
    prob[mask] .= 1 / n
    prob
end

function get_prob(p::RandomPolicy{Nothing}, env, ::ExplicitStochastic)
    if get_current_player(env) == get_chance_player(env)
        [x.prob for x::ActionProbPair in get_legal_actions(env)]
    else
        get_prob(p, env, DETERMINISTIC)
    end
end

get_prob(p::RandomPolicy, env, a) = 1 / length(p.action_space)
get_prob(p::RandomPolicy{<:VectSpace}, env::MultiThreadEnv, a) =
    [1 / length(x) for x in p.action_space.data]

get_prob(p::RandomPolicy{Nothing}, env, a::ActionProbPair) = a.prob

function get_prob(p::RandomPolicy{Nothing}, env, a)
    legal_actions = get_legal_actions(env)
    if a in legal_actions
        1.0 / length(legal_actions)
    else
        0.0
    end
end

update!(p::RandomPolicy, args...) = nothing

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
