export VectorialTrajectory

const VectorialTrajectory = Trajectory{
    names,
    types,
    NamedTuple{names,trace_types},
} where {names,types,trace_types<:Tuple{Vararg{<:Vector}}}

function VectorialTrajectory(; kwargs...)
    names = keys(kwargs.data)
    types = values(kwargs.data)
    trajectories =
        merge(NamedTuple(), (name, Vector{type}()) for (name, type) in zip(names, types))
    VectorialTrajectory{names,Tuple{types...},typeof(values(trajectories))}(trajectories)
end
