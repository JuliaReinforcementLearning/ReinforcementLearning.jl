using ReinforcementLearningCore
using ReinforcementLearningBase
import Base.push!
import Base.getindex
using CircularArrayBuffers: CircularVectorBuffer

"""
TotalRewardPerLastNEpisodes{F}(; max_episodes = 100)

A hook that keeps track of the total reward per episode for the last `max_steps` episodes.
"""
struct TotalRewardPerLastNEpisodes{F} <: AbstractHook where {F<:AbstractFloat}
    rewards::CircularVectorBuffer{F}

    function TotalRewardPerLastNEpisodes(; max_steps = 100)
        new{Float64}(CircularVectorBuffer{Float64}(max_steps))
    end
end

Base.getindex(h::TotalRewardPerLastNEpisodes{F}, inds...) where {F<:AbstractFloat} =
    getindex(h.rewards, inds...)

Base.push!(
    h::TotalRewardPerLastNEpisodes{F},
    ::PostActStage,
    agent::P,
    env::E,
    player::Symbol,
) where {P<:AbstractPolicy,E<:AbstractEnv,F<:AbstractFloat} =
    h.rewards[end] += reward(env, player)

Base.push!(
    hook::TotalRewardPerLastNEpisodes{F},
    ::PreEpisodeStage,
    agent,
    env,
) where {F<:AbstractFloat} = Base.push!(hook.rewards, 0.0)

Base.push!(
    hook::TotalRewardPerLastNEpisodes{F},
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent,
    env,
    player::Symbol,
) where {F<:AbstractFloat} = Base.push!(hook, stage, agent, env)
