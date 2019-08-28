export AbstractEnv, observe, reset!, interact!, action_space, observation_space, render, Observation, get_reward, get_terminal, get_state, get_legal_actions

abstract type AbstractEnv end

function observe end
function reset! end
function interact! end
function action_space end
function observation_space end
function render end

struct Observation{R, T, S, M<:NamedTuple}
    reward::R
    terminal::T
    state::S
    meta::M
end

Observation(;reward, terminal, state, kw...) = Observation(reward, terminal, state, merge(NamedTuple(), kw))

get_reward(obs::Observation) = obs.reward
get_terminal(obs::Observation) = obs.terminal
get_state(obs::Observation) = obs.state
get_legal_actions(obs::Observation) = obs.meta.legal_actions

# !!! >= julia v1.3
if VERSION >= v"1.3.0-rc1.0"
    (env::AbstractEnv)(a) = interact!(env, a)
end

action_space(env::AbstractEnv) = env.action_space
observation_space(env::AbstractEnv) = env.observation_space