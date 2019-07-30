export CircularTurnBuffer, circular_SART_buffer, capacity, isfull

struct CircularTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    buffers::Tbs
    function CircularTurnBuffer(configs::Pair{Symbol, NamedTuple{(:eltype, :capacity, :size), Tuple{DataType, Int, Tsize}}}...) where {Tsize<:NTuple{M, Int} where M}
        buffers = merge(NamedTuple(),
                        (n, CircularArrayBuffer{c.eltype}(c.capacity, c.size...)) for (n, c) in configs)
        names = Tuple(name for (name, _) in configs)
        types = Tuple{(config.eltype for (_, config) in configs)...}
        new{names, types, typeof(buffers)}(buffers)
    end
end

Base.getindex(b::CircularTurnBuffer{names, types}, i::Int) where {names, types} = NamedTuple{names, types}(Tuple(x[i] for x in b.buffers))

function circular_SART_buffer(
    ;capacity,
    state_eltype,
    state_size,
    action_eltype=Int,
    action_size=(),
    reward_eltype = Float64,
    reward_size=(),
    terminal_eltype=Bool,
    terminal_size=()
)
    CircularTurnBuffer(
        :state => (eltype=state_eltype, capacity=capacity+1, size=state_size),
        :action => (eltype=action_eltype, capacity=capacity+1, size=action_size),
        :reward => (eltype=reward_eltype, capacity=capacity, size=reward_size),
        :terminal => (eltype=terminal_eltype, capacity=capacity, size=terminal_size)
    )
end

Base.length(b::CircularTurnBuffer) = length(b.buffers.terminal)
capacity(b::CircularTurnBuffer) = capacity(b.buffers.terminal)
isfull(b::CircularTurnBuffer) = isfull(b.buffers.terminal)