import Base.push!

mutable struct SA{S,A}
    state::Union{S,Nothing}
    action::Union{A, Nothing}

    function SA()
        new{Any, Any}(nothing, nothing)
    end

    function SA{S,A}() where {S,A}
        new{S,A}(nothing, nothing)
    end
end

struct SART{S,A,R,T}
    state::S
    action::A
    reward::R
    terminal::T
end

function Base.push!(ts::Union{CircularArraySARTTraces,ElasticArraySARTTraces}, xs::SART)
    push!(ts.traces[1].trace, xs.state)
    push!(ts.traces[2].trace, xs.action)
    push!(ts.traces[3], xs.reward)
    push!(ts.traces[4], xs.terminal)
end

Base.push!(t::Trajectory, srt::SA) = throw(ArgumentError("action must be supplied when pushing SRT to trajectory. Use `Base.push!(t::Trajectory, srt::SRT; action::A)` to do so"))

function Base.push!(t::Trajectory, sa::SA{S,A}, reward::R, terminal::T) where {S,A,R,T}
    if !isnothing(sa.state) && !isnothing(sa.action)
        push!(t, SART{S,A,R,T}(srt.state::S, action, srt.reward::R, srt.terminal::T))
    end
end

function Base.push!(cache::SA{S,A}, state::S) where {S,A}
    cache.state = state
end

function Base.push!(cache::SA{S,A}, action::A) where {S,A}
    cache.action = action
end

function RLBase.reset!(cache::SA)
    cache.state = nothing
    cache.action = nothing
end
