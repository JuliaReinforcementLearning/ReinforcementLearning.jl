using ReinforcementLearningEnvironments

struct EnvObservation{Tr, Tt, Ts}
    reward::Tr
    terminal::Tt
    state::Ts
    meta::Dict{Symbol, Any}
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