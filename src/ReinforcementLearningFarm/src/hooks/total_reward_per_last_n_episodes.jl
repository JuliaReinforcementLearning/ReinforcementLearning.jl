using ReinforcementLearning
import Base.push!
import Base.getindex
using CircularArrayBuffers: CircularVectorBuffer, CircularArrayBuffer

"""
TotalRewardPerLastNEpisodes{F}(; max_episodes = 100)

A hook that keeps track of the total reward per episode for the last `max_episodes` episodes.
"""
struct TotalRewardPerLastNEpisodes{B} <: AbstractHook where {B<:CircularArrayBuffer}
    rewards::B

    function TotalRewardPerLastNEpisodes(; max_episodes = 100)
        buffer = CircularVectorBuffer{Float64}(max_episodes)
        new{typeof(buffer)}(buffer)
    end
end

Base.getindex(h::TotalRewardPerLastNEpisodes{B}, inds...) where {B<:CircularArrayBuffer} =
    getindex(h.rewards, inds...)

Base.push!(
    h::TotalRewardPerLastNEpisodes{B},
    ::PostActStage,
    agent::P,
    env::E,
    player::Player,
) where {P<:AbstractPolicy,E<:AbstractEnv,B<:CircularArrayBuffer} =
    h.rewards[end] += reward(env, player)

Base.push!(
    hook::TotalRewardPerLastNEpisodes{B},
    ::PreEpisodeStage,
    agent::AbstractPolicy,
    env::AbstractEnv,
) where {B<:CircularArrayBuffer} = push!(hook.rewards, 0.0)

Base.push!(
    hook::TotalRewardPerLastNEpisodes{B},
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent::AbstractPolicy,
    env::AbstractEnv,
    ::AbstractPlayer,
) where {B<:CircularArrayBuffer} = push!(hook, stage, agent, env)
