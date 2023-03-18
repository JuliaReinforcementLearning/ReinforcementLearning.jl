mutable struct SRT{S,R,T}
    state::Union{S,Nothing}
    reward::Union{R,Nothing}
    terminal::Union{T,Nothing}

    function SRT()
        new{Any, Any, Any}(nothing, nothing, nothing)
    end

    function SRT{S,R,T}() where {S,R,T}
        new{S,R,T}(nothing, nothing, nothing)
    end
end

Base.push!(t::Trajectory, srt::SRT) = throw(ArgumentError("action must be supplied when pushing SRT to trajectory"))

function Base.push!(t::Trajectory, srt::SRT{S,R,T}, action::A) where {S,A,R,T}
    if srt.reward != nothing & srt.terminal == nothing
        push!(t, @NamedTuple{state::S, action::A, reward::R, terminal::T}((srt.state, action, srt.reward, srt.terminal)))
    else
        push!(t, @NamedTuple{state::S, action::A}((srt.state, action)))
    end
end

function update!(cache::SRT{S,R,T}, state::S) where {S,R,T}
    cache.state = state
end

function update!(cache::SRT{S,R,T}, reward::R, terminal::T) where {S,R,T}
    cache.reward = reward
    cache.terminal = terminal
end

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    update!(agent.cache, reward(env), is_terminated(env))
end

function (agent::Agent)(::PostExperimentStage, env::E) where {E <: AbstractEnv}
    RLBase.reset(agent.cache)
end

function RLBase.reset(cache::SRT)
    cache.state = nothing
    cache.reward = nothing
    cache.terminal = nothing
end
