"""
    mutable struct ForcedEpisode{Ts}
        t::Int64
        states::Ts
        dones::Array{Bool, 1}
        rewards::Array{Float64, 1}
""" 
mutable struct ForcedEpisode{Ts}
    t::Int64
    states::Ts
    dones::Array{Bool, 1}
    rewards::Array{Float64, 1}
end
export ForcedEpisode
ForcedEpisode(states, dones, rewards) = ForcedEpisode(1, states, dones, rewards)
function interact!(env::ForcedEpisode, a)
    env.t += 1
    (observation = env.states[env.t], reward = env.rewards[env.t], 
     isdone = env.dones[env.t])
end
function reset!(env::ForcedEpisode)
    env.t = 1
    (obervation = env.states[1], )
end
getstate(env::ForcedEpisode) = (observation = env.states[env.t], isdone = env.dones[env.t])
