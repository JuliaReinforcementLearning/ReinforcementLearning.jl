using ReinforcementLearningCore
using ReinforcementLearningBase
import Base.push!
import Base.getindex
using DataStructures: CircularBuffer

struct TotalRewardPerEpisodeLastN{F} <: AbstractHook where {F<:AbstractFloat}
    rewards::CircularBuffer{F}
    is_display_on_exit::Bool

    function TotalRewardPerEpisodeLastN(; max_steps = 100)
        new{Float64}(CircularBuffer{Float64}(max_steps))
    end
end

Base.getindex(h::TotalRewardPerEpisodeLastN{F}, inds...) where {F<:AbstractFloat} =
    getindex(h.rewards, inds...)

Base.push!(
    h::TotalRewardPerEpisodeLastN{F},
    ::PostActStage,
    agent::P,
    env::E,
    player::Symbol,
) where {P<:AbstractPolicy,E<:AbstractEnv,F<:AbstractFloat} =
    h.rewards[end] += reward(env, player)

function Base.push!(
    hook::TotalRewardPerEpisodeLastN{F},
    ::PreEpisodeStage,
    agent,
    env,
) where {F<:AbstractFloat}
    Base.push!(hook.rewards, 0.0)
    return
end

function Base.push!(
    hook::TotalRewardPerEpisodeLastN{F},
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent,
    env,
    player::Symbol,
) where {F<:AbstractFloat}
    Base.push!(hook, stage, agent, env)
    return
end
