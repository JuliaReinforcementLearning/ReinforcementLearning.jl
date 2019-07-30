export CircularTurnBuffer, CircularTurnBuffer, capacity, isfull

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

###
# CircularSARDBuffer
###

function CircularSARDBuffer(
    ;capacity,
    state_eltype,
    state_size,
    action_eltype=Int,
    action_size=(),
    reward_eltype = Float64,
    reward_size=(),
    isdone_eltype=Bool,
    isdone_size=()
)
    CircularTurnBuffer(
        :state => (eltype=state_eltype, capacity=capacity+1, size=state_size),
        :action => (eltype=action_eltype, capacity=capacity+1, size=action_size),
        :reward => (eltype=reward_eltype, capacity=capacity, size=reward_size),
        :isdone => (eltype=isdone_eltype, capacity=capacity, size=isdone_size)
    )
end

Base.length(b::CircularTurnBuffer{SARD}) = length(b.buffers.isdone)
capacity(b::CircularTurnBuffer{SARD}) = capacity(b.buffers.isdone)
isfull(b::CircularTurnBuffer{SARD}) = isfull(b.buffers.isdone)