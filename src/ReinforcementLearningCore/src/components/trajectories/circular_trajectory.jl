export CircularTrajectory

const CircularTrajectory = Trajectory{
    names,
    types,
    NamedTuple{names,trace_types},
} where {names,types,trace_types<:Tuple{Vararg{<:CircularArrayBuffer}}}

function CircularTrajectory(; capacity, kwargs...)
    names = keys(kwargs.data)
    types_and_sizes = values(kwargs.data)
    types = (t for (t, s) in types_and_sizes)
    sizes = (s for (t, s) in types_and_sizes)
    trajectories = merge(
        NamedTuple(),
        (name, CircularArrayBuffer{t}(s..., capacity))
        for (name, (t, s)) in zip(names, types_and_sizes)
    )
    CircularTrajectory{names,Tuple{types...},typeof(values(trajectories))}(trajectories)
end
