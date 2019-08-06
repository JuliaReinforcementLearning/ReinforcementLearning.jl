struct EnvObservation{Tr, Tt, Ts}
    reward::Tr
    terminal::Tt
    state::Ts
    meta::Dict{Symbol, Any}
end

function EnvObservation(args::NamedTuple)
    obs, reward, isdone = args
    EnvObservation(
        reward,
        isdone,
        obs,
        Dict(collect(pairs(args))[4:end])
    )
end

state(obs::EnvObservation) = obs.state
is_terminal(obs::EnvObservation) = convert(Bool, obs.terminal)