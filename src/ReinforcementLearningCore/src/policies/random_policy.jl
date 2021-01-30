export RandomPolicy

using Random

"""
    RandomPolicy(action_space=nothing; rng=Random.GLOBAL_RNG)

If `action_space` is `nothing`, then it will use the `legal_action_space` at
runtime to randomly select an action. Otherwise, a random element within
`action_space` is selected. 

!!! note
    You should always set `action_space=nothing` when dealing with environments
    of `FULL_ACTION_SET`.
"""
struct RandomPolicy{S,R<:AbstractRNG} <: AbstractPolicy
    action_space::S
    rng::R
end

Random.seed!(p::RandomPolicy, seed) = Random.seed!(p.rng, seed)

RandomPolicy(s = nothing; rng = Random.GLOBAL_RNG) = RandomPolicy(s, rng)

(p::RandomPolicy{Nothing})(env) = rand(p.rng, legal_action_space(env))
(p::RandomPolicy)(env) = rand(p.rng, p.action_space)

function RLBase.prob(p::RandomPolicy{<:Union{AbstractVector,Tuple}}, env::AbstractEnv)
    prob(p, state(env))
end

function RLBase.prob(p::RandomPolicy{<:Union{AbstractVector,Tuple}}, s)
    n = length(p.action_space)
    Categorical(fill(1 / n, n); check_args = false)
end

RLBase.prob(p::RandomPolicy{Nothing}, env::AbstractEnv) = prob(p, env, ChanceStyle(env))
RLBase.prob(p::RandomPolicy{Nothing}, x) =
    @error "no I really don't know how to calculate the prob from nothing"

function RLBase.prob(
    p::RandomPolicy{Nothing},
    env::AbstractEnv,
    ::RLBase.AbstractChanceStyle,
)
    mask = legal_action_space_mask(env)
    n = sum(mask)
    prob = zeros(length(mask))
    prob[mask] .= 1 / n
    prob
end

function RLBase.prob(
    p::RandomPolicy{Nothing},
    env::AbstractEnv,
    ::RLBase.ExplicitStochastic,
)
    if current_player(env) == chance_player(env)
        prob(env, chance_player(env))
    else
        prob(p, env, DETERMINISTIC)
    end
end

RLBase.update!(p::RandomPolicy, x) = nothing

RLBase.prob(p::RandomPolicy, env_or_state, a) = 1 / length(p.action_space)

function RLBase.prob(p::RandomPolicy{Nothing}, env::AbstractEnv, a)
    # we can safely assume s is discrete here.
    s = legal_action_space(env)
    if a in s
        1.0 / length(s)
    else
        0.0
    end
end
