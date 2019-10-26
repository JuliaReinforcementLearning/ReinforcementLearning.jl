export StackedState

"""
    StackedState(; state_size, state_eltype=Float32, n_frames=4)

Use a [`CircularArrayBuffer`](@ref) to store stacked states.

# Example
"""
struct StackedState{E, T, N}
    states::CircularArrayBuffer{E, T, N}
end

function StackedState(; state_size, state_eltype=Float32, n_frames=4)
    states = CircularArrayBuffer{Array{state_eltype, length(state_size)}}(n_frames, state_size)
    for _ in 1:n_frames
        push!(states, zeros(state_eltype, state_size))
    end
    StackedState(states)
end

Base.push!(s::StackedState, state) = push!(s.states, state)
Base.getindex(s::StackedState, i::Int) = getindex(s.states, i)
Base.view(s::StackedState, i::Int) = view(s.states, i)
Base.lastindex(s::StackedState) = lastindex(s.states)

Base.push!(b::CircularArrayBuffer, s::StackedState) = push!(b, @view(s[end]))