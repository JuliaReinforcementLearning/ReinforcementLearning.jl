export EpisodeCompactSARTSABuffer

const EpisodeCompactSARTSABuffer = Trajectory{SARTSA, T1, NamedTuple{RTSA, T2}} where {T1, T2<:Tuple{Vararg{<:Vector}}}

function EpisodeCompactSARTSABuffer(;
    state_eltype = Int,
    action_eltype = Int,
    reward_eltype = Float32,
    terminal_eltype = Bool,
)
    EpisodeCompactSARTSABuffer{
        Tuple{
            state_eltype,
            action_eltype,
            reward_eltype,
            terminal_eltype,
            state_eltype,
            action_eltype,
        },
        Tuple{
            Vector{reward_eltype},
            Vector{terminal_eltype},
            Vector{state_eltype},
            Vector{action_eltype},
        }
    }(
        (
            reward=Vector{reward_eltype}(),
            terminal = Vector{terminal_eltype}(),
            state=Vector{state_eltype}(),
            action=Vector{action_eltype}()
        )
    )
end

Base.length(b::EpisodeCompactSARTSABuffer) = length(getfield(getfield(b, :buffers), :terminal))

RLBase.get_trace(b::EpisodeCompactSARTSABuffer, s::Symbol) = _get_trace(b, Val(s))
_get_trace(b::EpisodeCompactSARTSABuffer, ::Val{:state}) = select_last_dim(b[:state], 1:(length(b) == 0 ? length(b[:state]) : length(b[:state])-1))
_get_trace(b::EpisodeCompactSARTSABuffer, ::Val{:action}) = select_last_dim(b[:action], 1:(length(b) == 0 ? length(b[:state]) : length(b[:action])-1))
_get_trace(b::EpisodeCompactSARTSABuffer, ::Val{:reward}) = b[:reward]
_get_trace(b::EpisodeCompactSARTSABuffer, ::Val{:terminal}) = b[:terminal]
_get_trace(b::EpisodeCompactSARTSABuffer, ::Val{:next_state}) = select_last_dim(b[:state], 2:length(b[:state]))
_get_trace(b::EpisodeCompactSARTSABuffer, ::Val{:next_action}) = select_last_dim(b[:action], 2:length(b[:action]))

RLBase.isfull(b::EpisodeCompactSARTSABuffer) = (length(b[:terminal]) > 0) && b[:terminal][end]

function Base.empty!(b::EpisodeCompactSARTSABuffer)
    for s in RTSA
        empty!(b[s])
    end
    b
end

Base.isempty(b::EpisodeCompactSARTSABuffer) = all(isempty(b[s]) for s in RTSA)

function Base.push!(b::EpisodeCompactSARTSABuffer; state=nothing, action=nothing, reward=nothing, terminal=nothing, next_state=nothing, next_action=nothing)
    if isfull(b)
        empty!(b)
    end
    isnothing(state) || push!(b[:state], state)
    isnothing(action) || push!(b[:action], action)
    isnothing(reward) || push!(b[:reward], reward)
    isnothing(terminal) || push!(b[:terminal], terminal)
    isnothing(next_state) || push!(b[:state], next_state)
    isnothing(next_action) || push!(b[:action], next_action)
    b
end