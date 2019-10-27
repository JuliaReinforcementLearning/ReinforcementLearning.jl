export StackedState

using Flux:@forward

"""
    StackedState(; state_size, state_eltype=Float32, n_frames=4)

Use a [`CircularArrayBuffer`](@ref) to store stacked states.

# Example
"""
struct StackedState{E, T, N} <: AbstractArray{T, N}
    states::CircularArrayBuffer{E, T, N}
end

function StackedState(; state_size, state_eltype=Float32, n_frames=4)
    states = CircularArrayBuffer{Array{state_eltype, length(state_size)}}(n_frames, state_size)
    for _ in 1:n_frames
        push!(states, zeros(state_eltype, state_size))
    end
    StackedState(states)
end

# !!! do not forward Base.length
@forward StackedState.states Base.push!, Base.getindex, Base.setindex, Base.view, Base.size, Base.lastindex

Base.push!(b::CircularArrayBuffer, s::StackedState) = push!(b, @view(s[end]))