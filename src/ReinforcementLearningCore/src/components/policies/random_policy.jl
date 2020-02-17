export RandomPolicy

using Random

struct RandomPolicy{S<:AbstractSpace,R<:AbstractRNG} <: AbstractPolicy
    action_space::S
    rng::R
end

Base.show(io::IO, p::RandomPolicy) = print(io, "RandomPolicy($(p.action_space))")

Random.seed!(p::RandomPolicy, seed) = Random.seed!(p.rng, seed)

RandomPolicy(s; seed = nothing) = RandomPolicy(s, MersenneTwister(seed))

RandomPolicy(env::AbstractEnv; seed = nothing) =
    RandomPolicy(get_action_space(env), MersenneTwister(seed))

(p::RandomPolicy)(obs, ::FullActionSet) = rand(p.rng, get_legal_actions(obs))
(p::RandomPolicy)(obs, ::MinimalActionSet) = rand(p.rng, p.action_space)

RLBase.update!(p::RandomPolicy, experience) = nothing

RLBase.get_prob(p::RandomPolicy, s) =
    fill(1 / length(p.action_space), length(p.action_space))
RLBase.get_prob(p::RandomPolicy, s, a) = 1 / length(p.action_space)
