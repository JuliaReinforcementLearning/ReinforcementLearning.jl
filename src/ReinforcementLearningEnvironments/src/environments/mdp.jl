export POMDPEnv

using POMDPs
using Random

RLBase.get_action_space(m::Union{<:POMDP, <:MDP}) = convert(AbstractSpace, actions(m))

#####
# POMDPEnv
#####

mutable struct POMDPEnv{M<:POMDP,S,O,I,R,RNG<:AbstractRNG} <: AbstractEnv
    model::M
    state::S
    observation::O
    info::I
    reward::R
    rng::RNG
end

Random.seed!(env::POMDPEnv, seed) = seed!(env.rng, seed)

function POMDPEnv(model::POMDP; seed=nothing)
    rng = MersenneTwister(seed)
    s = initialstate(model, rng)
    a = rand(rng, actions(model))
    if :info in nodenames(DDNStructure(model))
        sp, o, r, info = gen(DDNOut(:sp, :o, :r, :info), model, s, a, rng)
    else
        (sp, o, r), info = gen(DDNOut(:sp, :o, :r), model, s, a, rng), nothing
    end
    env = POMDPEnv(model, sp, o, info, r, rng)
    reset!(env)
    env
end

# no info
function (env::POMDPEnv{<:POMDP,<:Any,<:Any,<:Nothing,<:Any,<:AbstractRNG})(a)
    sp, o, r = gen(DDNOut(:sp, :o, :r), env.model, env.state, a, env.rng)
    env.state = sp
    env.observation = o
    env.reward = r
    nothing
end

# has info
function (env::POMDPEnv)(a)
    sp, o, r, info = gen(DDNOut(:sp, :o, :r, :info), env.model, env.state, a, env.rng)
    env.state = sp
    env.observation = o
    env.info = info
    env.reward = r
    nothing
end

RLBase.observe(env::POMDPEnv) = (
    state=env.observation,
    reward=env.reward,
    terminal=isterminal(env.model, env.state),
    inner_state=env.state,
    info=env.info
)

function RLBase.reset!(env::POMDPEnv)
    env.state = initialstate(env.model, env.rng)
    env.observation = initialobs(env.model, env.state, env.rng)
    nothing
end

RLBase.get_observation_space(env::POMDPEnv) = get_observation_space(env.model)
RLBase.get_action_space(env::POMDPEnv) = get_action_space(env.model)

#####
# MDPEnv
#####

mutable struct MDPEnv{M<:MDP,S,I,R,RNG<:AbstractRNG} <: AbstractEnv
    model::M
    state::S
    info::I
    reward::R
    rng::RNG
end

Random.seed!(env::MDPEnv, seed) = seed!(env.rng, seed)

function MDPEnv(model::MDP; seed=nothing)
    rng = MersenneTwister(seed)
    s = initialstate(model, rng)
    a = rand(rng, actions(model))
    if :info in nodenames(DDNStructure(model))
        sp, r, info = gen(DDNOut(:sp, :r, :info), model, s, a, rng)
    else
        (sp, r), info = gen(DDNOut(:sp, :r), model, s, a, rng), nothing
    end
    env = MDPEnv(model, sp, info, r, rng)
    reset!(env)
    env
end

# no info
function (env::MDPEnv{<:MDP,<:Any,<:Nothing,<:Any,<:AbstractRNG})(a)
    sp, r = gen(DDNOut(:sp, :r), env.model, env.state, a, env.rng)
    env.state = sp
    env.reward = r
    nothing
end

# has info
function (env::MDPEnv)(a)
    sp, r, info = gen(DDNOut(:sp, :r, :info), env.model, env.state, a, env.rng)
    env.state = sp
    env.info = info
    env.reward = r
    nothing
end

RLBase.observe(env::MDPEnv) = (
    state=env.state,
    reward=env.reward,
    terminal=isterminal(env.model, env.state),
    info=env.info
)

function RLBase.reset!(env::MDPEnv)
    env.state = initialstate(env.model, env.rng)
    nothing
end

RLBase.get_observation_space(env::MDPEnv) = get_observation_space(env.model)
RLBase.get_action_space(env::MDPEnv) = get_action_space(env.model)
