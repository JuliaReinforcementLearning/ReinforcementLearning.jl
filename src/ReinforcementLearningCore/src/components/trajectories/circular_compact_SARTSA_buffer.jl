export CircularCompactSARTSABuffer

const CircularCompactSARTSABuffer = Trajectory{SARTSA, T1, NamedTuple{RTSA, T2}} where {T1, T2<:Tuple{Vararg{<:CircularArrayBuffer}}}

function CircularCompactSARTSABuffer(;
    capacity,
    state_eltype = Int,
    state_size = (),
    action_eltype = Int,
    action_size = (),
    reward_eltype = Float64,
    reward_size = (),
    terminal_eltype = Bool,
    terminal_size = (),
)
    capacity > 0 || throw(ArgumentError("capacity must > 0"))
    CircularCompactSARTSABuffer{ 
        Tuple{
            state_eltype,
            action_eltype,
            reward_eltype,
            terminal_eltype,
            state_eltype,
            action_eltype
        },
        Tuple{
            CircularArrayBuffer{reward_eltype, length(reward_size)+1},
            CircularArrayBuffer{terminal_eltype, length(terminal_size)+1},
            CircularArrayBuffer{state_eltype, length(state_size)+1},
            CircularArrayBuffer{action_eltype, length(action_size)+1}
        }
    }(
        (
            reward = CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
            terminal = CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
            state = CircularArrayBuffer{state_eltype}(state_size..., capacity+1),
            action = CircularArrayBuffer{action_eltype}(action_size..., capacity+1)
        )
    )
end

Base.length(b::CircularCompactSARTSABuffer) = length(b[:terminal])
Base.isempty(b::CircularCompactSARTSABuffer) = all(isempty(b[s]) for s in RTSA)
RLBase.isfull(b::CircularCompactSARTSABuffer) = all(isfull(b[s]) for s in RTSA)

"Exactly the same with [`EpisodeCompactSARTSABuffer`](@ref)"
RLBase.get_trace(b::CircularCompactSARTSABuffer, s::Symbol) = _get_trace(b, Val(s))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:state}) = select_last_dim(b[:state], 1:(length(b) == 0 ? length(b[:state]) : length(b[:state])-1))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:action}) = select_last_dim(b[:action], 1:(length(b) == 0 ? length(b[:state]) : length(b[:action])-1))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:reward}) = b[:reward]
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:terminal}) = b[:terminal]
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:next_state}) = select_last_dim(b[:state], 2:length(b[:state]))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:next_action}) = select_last_dim(b[:action], 2:length(b[:action]))

function Base.getindex(b::CircularCompactSARTSABuffer, i::Int)
    (
        state = select_last_dim(b[:state], i),
        action = select_last_dim(b[:action], i),
        reward = select_last_dim(b[:reward], i),
        terminal = select_last_dim(b[:terminal], i),
        next_state = select_last_dim(b[:state], i+1),
        next_action = select_last_dim(b[:action], i+1)
    )
end

function Base.empty!(b::CircularCompactSARTSABuffer)
    for s in RTSA
        empty!(b[s])
    end
    b
end

function Base.push!(b::CircularCompactSARTSABuffer; state=nothing, action=nothing, reward=nothing, terminal=nothing, next_state=nothing, next_action=nothing)
    push!(b, state, action, reward, terminal, next_state, next_action)
end

function Base.push!(b::CircularCompactSARTSABuffer, s, a, ::Nothing, ::Nothing, ::Nothing, ::Nothing)
    if length(b) == 0
        push!(b[:state], s)
        push!(b[:action], a)
    else
        # @assert b[:terminal] "only reset state and action at the start of a new episode!"
        update!(b[:state], s)
        update!(b[:action], a)
    end
    b
end

function Base.push!(b::CircularCompactSARTSABuffer, ::Nothing, ::Nothing, r, t, s′, a′)
    # @assert length(b) > 0 && !b[:terminal] "shouldn't be called at the start of an episode"
    push!(b[:reward], r)
    push!(b[:terminal], t)
    push!(b[:state], s′)
    push!(b[:action], a′)
    b
end