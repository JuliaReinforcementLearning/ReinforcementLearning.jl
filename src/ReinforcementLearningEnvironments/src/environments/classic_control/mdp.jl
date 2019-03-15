@reexport module MDP
using Random, POMDPs, POMDPModels
using ..ReinforcementLearningEnvironments
const RLEnv = ReinforcementLearningEnvironments

export MDPEnv, POMDPEnv

mutable struct POMDPEnv{T,Ts,Ta, R<:AbstractRNG}
    model::T
    state::Ts
    actions::Ta
    action_space::DiscreteSpace
    observation_space::DiscreteSpace
    rng::R
end

POMDPEnv(model; rng=Random.GLOBAL_RNG) = POMDPEnv(
    model,
    initialstate(model, rng),
    actions(model),
    DiscreteSpace(n_actions(model)),
    DiscreteSpace(n_states(model)),
    rng)

mutable struct MDPEnv{T, Ts, Ta, R<:AbstractRNG}
    model::T
    state::Ts
    actions::Ta
    action_space::DiscreteSpace
    observation_space::DiscreteSpace
    rng::R
end

MDPEnv(model; rng=Random.GLOBAL_RNG) = MDPEnv(
    model,
    initialstate(model, rng),
    actions(model),
    DiscreteSpace(n_actions(model)),
    DiscreteSpace(n_states(model)),
    rng)

RLEnv.action_space(env::Union{MDPEnv, POMDPEnv}) = env.action_space
RLEnv.observation_space(env::Union{MDPEnv, POMDPEnv}) = env.observation_space

observationindex(env, o) = Int64(o) + 1

function RLEnv.interact!(env::POMDPEnv, action) 
    s, o, r = generate_sor(env.model, env.state, env.actions[action], env.rng)
    env.state = s
    (observation = observationindex(env.model, o), 
     reward = r, 
     isdone = isterminal(env.model, s))
end

function RLEnv.reset!(env::Union{POMDPEnv, MDPEnv})
    initialstate(env.model, env.rng)
    nothing
end

function RLEnv.observe(env::POMDPEnv)
    (observation = observationindex(env.model, generate_o(env.model, env.state, env.rng)),
     isdone = isterminal(env.model, env.state))
end

function RLEnv.interact!(env::MDPEnv, action)
    s = rand(env.rng, transition(env.model, env.state, env.actions[action]))
    r = reward(env.model, env.state, env.actions[action])
    env.state = s
    (observation = stateindex(env.model, s), 
     reward = r, 
     isdone = isterminal(env.model, s))
end

function RLEnv.observe(env::MDPEnv)
    (observation = stateindex(env.model, env.state), 
     isdone = isterminal(env.model, env.state))
end
end