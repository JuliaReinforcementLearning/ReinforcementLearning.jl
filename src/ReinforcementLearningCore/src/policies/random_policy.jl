export RandomPolicy

using Random: Random, AbstractRNG
using Distributions: Categorical
using FillArrays: Fill

"""
    RandomPolicy(action_space=nothing; rng=Random.default_rng())

If `action_space` is `nothing`, then it will use the `legal_action_space` at
runtime to randomly select an action. Otherwise, a random element within
`action_space` is selected.

!!! note
    You should always set `action_space=nothing` when dealing with environments
    of `FULL_ACTION_SET`.
"""
struct RandomPolicy{S,RNG<:AbstractRNG} <: AbstractPolicy
    action_space::S
    rng::RNG
end

RandomPolicy(s = nothing; rng = Random.default_rng()) = RandomPolicy(s, rng)

RLBase.optimise!(::RandomPolicy, x::NamedTuple) = nothing

RLBase.plan!(p::RandomPolicy{S,RNG}, ::AbstractEnv) where {S,RNG<:AbstractRNG} = rand(p.rng, p.action_space)

function RLBase.plan!(p::RandomPolicy{Nothing,RNG}, env::AbstractEnv) where {RNG<:AbstractRNG}
    legal_action_space_ = RLBase.legal_action_space(env)
    return rand(p.rng, legal_action_space_)
end

function RLBase.plan!(p::RandomPolicy{Nothing,RNG}, env::E, player::Player) where {E<:AbstractEnv, RNG<:AbstractRNG, Player <: AbstractPlayer}
    legal_action_space_ = RLBase.legal_action_space(env, player)
    return rand(p.rng, legal_action_space_)
end

#####

RLBase.prob(p::RandomPolicy, env::AbstractEnv) = prob(p, state(env))

function RLBase.prob(p::RandomPolicy{S,RNG}, s) where {S,RNG<:AbstractRNG}
    n = length(p.action_space)
    Categorical(Fill(1 / n, n); check_args = false)
end

RLBase.prob(::RandomPolicy{Nothing,RNG}, x) where {RNG<:AbstractRNG} =
    @error "no I really don't know how to calculate the prob from nothing"

#####

RLBase.prob(p::RandomPolicy{Nothing,RNG}, env::AbstractEnv) where {RNG<:AbstractRNG} =
    prob(p, env, ChanceStyle(env))

function RLBase.prob(
    ::RandomPolicy{Nothing,RNG},
    env::AbstractEnv,
    ::RLBase.AbstractChanceStyle,
) where {RNG<:AbstractRNG}
    mask = legal_action_space_mask(env)
    n = sum(mask)
    prob = zeros(length(mask))
    prob[mask] .= 1 / n
    prob
end

function RLBase.prob(
    p::RandomPolicy{Nothing,RNG},
    env::AbstractEnv,
    ::RLBase.ExplicitStochastic,
) where {RNG<:AbstractRNG}
    if current_player(env) == chance_player(env)
        prob(env, chance_player(env))
    else
        prob(p, env, DETERMINISTIC)
    end
end

#####

RLBase.prob(p::RandomPolicy{S,RNG}, env_or_state, a) where {S,RNG<:AbstractRNG} =
    1 / length(p.action_space)

function RLBase.prob(
    p::RandomPolicy{Nothing,RNG},
    env::AbstractEnv,
    a,
) where {RNG<:AbstractRNG}
    # we can safely assume s is discrete here.
    s = legal_action_space(env)
    if a in s
        1.0 / length(s)
    else
        0.0
    end
end
