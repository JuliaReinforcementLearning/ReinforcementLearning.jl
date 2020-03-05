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

function (p::RandomPolicy)(obs, ::FullActionSet)
    legal_actions = get_legal_actions(obs)
    length(legal_actions) == 0 ? get_invalid_action(obs) : rand(p.rng, legal_actions)
end

(p::RandomPolicy)(obs, ::MinimalActionSet) = rand(p.rng, p.action_space)
(p::RandomPolicy)(obs::BatchObs, ::MinimalActionSet) =
    [rand(p.rng, p.action_space) for _ in 1:length(obs)]

RLBase.update!(p::RandomPolicy, experience) = nothing

RLBase.get_prob(p::RandomPolicy, s) =
    fill(1 / length(p.action_space), length(p.action_space))
RLBase.get_prob(p::RandomPolicy, s, a) = 1 / length(p.action_space)
