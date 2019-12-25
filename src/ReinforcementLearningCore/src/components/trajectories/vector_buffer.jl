export VectorBuffer

const VectorBuffer = Trajectory{names, types, NamedTuple{names, trace_types}} where {names, types, trace_types<:Tuple{Vararg{<:Vector}}}

function VectorBuffer(configs::Pair{Symbol,DataType}...)
    names = Tuple(name for (name, _) in configs)
    types = Tuple{(type for (_, type) in configs)...}
    buffers = merge(NamedTuple(), (name, Vector{type}()) for (name, type) in configs)
    Trajectory{names,types,typeof(buffers)}(buffers)
end

RLBase.isfull(b::VectorBuffer) = false
