import Base.push!

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

struct SA{S,A}
    state::S
    action::A
end

struct SART{S,A,R,T}
    state::S
    action::A
    reward::R
    terminal::T
end

# This method is used to push a state and action to a trace 
function Base.push!(ts::Union{CircularArraySARTSTraces,ElasticArraySARTSTraces}, xs::SA)
    push!(ts.traces[1].trace, xs.state)
    push!(ts.traces[2].trace, xs.action)
end

function Base.push!(ts::Union{CircularArraySARTSTraces,ElasticArraySARTSTraces}, xs::SART)
    push!(ts.traces[1].trace, xs.state)
    push!(ts.traces[2].trace, xs.action)
    push!(ts.traces[3], xs.reward)
    push!(ts.traces[4], xs.terminal)
end

Base.push!(t::Trajectory, srt::SRT) = throw(ArgumentError("action must be supplied when pushing SRT to trajectory. Use `Base.push!(t::Trajectory, srt::SRT; action::A)` to do so"))

function Base.push!(t::Trajectory, srt::SRT{S,R,T}, action::A) where {S,A,R,T}
    if isnothing(srt.reward) || isnothing(srt.terminal)
        push!(t, SA{S,A}(srt.state::S, action))
    else
        push!(t, SART{S,A,R,T}(srt.state::S, action, srt.reward::R, srt.terminal::T))
    end
end

function Base.push!(cache::SRT{S,R,T}, state::S) where {S,R,T}
    cache.state = state
end

function Base.push!(cache::SRT{S,R,T}, reward::R, terminal::T) where {S,R,T}
    cache.reward = reward
    cache.terminal = terminal
end

function RLBase.reset!(cache::SRT)
    cache.state = nothing
    cache.reward = nothing
    cache.terminal = nothing
end
