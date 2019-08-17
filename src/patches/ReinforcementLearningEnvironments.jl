using ReinforcementLearningEnvironments

import ReinforcementLearningEnvironments.observe

mutable struct EnvObservation{Tr, Tt, Ts, Tm<:Dict}
    reward::Tr
    terminal::Tt
    state::Ts
    meta::Tm
end

function EnvObservation(;observation, reward, isdone, kw...)
    EnvObservation(
        reward,
        isdone,
        observation,
        Dict(kw)
    )
end

observe(env) = EnvObservation(;ReinforcementLearningEnvironments.observe(env)...)

terminal(obs::EnvObservation) = obs.terminal
reward(obs::EnvObservation) = obs.reward
state(obs::EnvObservation) = obs.state

is_terminal(obs::EnvObservation) = convert(Bool, obs.terminal)


# specific patches

observe(env::CartPoleEnv) = EnvObservation(;observation=env.state, reward=env.done ? 0.0 : 1.0, isdone=env.done)

Base.size(s::MultiContinuousSpace) = size(s.low)
Base.length(s::MultiContinuousSpace) = length(s.low)
(env::AbstractEnv)(a) = interact!(env, a)